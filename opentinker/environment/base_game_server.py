#!/usr/bin/env python3
"""Base Game Server for LLM Training Environments.

This module provides an async HTTP server (FastAPI + uvicorn) that works with any
AbstractGame implementation. Includes per-step/batch metrics tracking with multi-job
statistics isolation support.

Parallelism is achieved via sharding: multiple server processes on consecutive ports.
Each shard handles requests independently with its own in-memory instance registry.

Endpoints:
    POST /reset           - Reset game instance (accepts job_id for stats isolation)
    POST /step            - Execute action (accepts job_id for stats isolation)
    GET  /health          - Health check
    GET  /stats           - Get current step statistics (accepts ?job_id=...)
    GET  /stats/step      - Get step stats only (accepts ?job_id=...)
    GET  /stats/cumulative - Get cumulative stats (accepts ?job_id=...)
    POST /stats/reset     - Reset step statistics (accepts ?job_id=...)
    POST /stats/reset_all - Reset all statistics (accepts ?job_id=... or reset all jobs)
    GET  /stats/jobs      - List all active job IDs
    DELETE /stats/job/{job_id} - Remove a job's statistics

Multi-Job Isolation:
    The server supports multiple concurrent training/inference tasks via job_id.
    Each job gets isolated statistics. Default job_id is "default" for backward compatibility.

Example:
    from base_game_server import run_game_server
    from my_game import MyGame

    run_game_server(MyGame, port=8081, board_size=9)
"""

import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional, Type

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from opentinker.environment.base_game import AbstractGame


class BaseGameStats:
    """Thread-safe base statistics tracker for game server.

    Tracks generic per-step statistics. Subclass for game-specific metrics.

    Key metrics:
    - games_in_step: Number of completed games (done=True)
    - total_samples: Total number of games started (regardless of completion)
    - incomplete_samples: Games that started but never got done=True
    - mean_final_reward: Average of FINAL rewards only (when done=True)
    - mean_sum_reward: Average of per-game total rewards
    - mean_avg_reward: Average of per-game average rewards
    - total_interactions: Total step() calls
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._current_step = 0
        self._step_stats: Dict[str, List[float]] = defaultdict(list)
        self._cumulative_stats: Dict[str, float] = defaultdict(float)
        # Track per-game reward accumulation: instance_id -> {sum_reward, step_count}
        self._game_rewards: Dict[str, Dict[str, float]] = {}
        # Track all games that started in this step (to count total samples)
        self._started_games: set = set()

    def register_game_start(self, instance_id: str, job_id: str = "default"):
        """Register that a game has started (called on /reset).

        This ensures games are counted even if they never call /step.
        """
        with self._lock:
            if instance_id not in self._started_games:
                self._started_games.add(instance_id)
                print(
                    f"[DEBUG] Game registered on reset: {instance_id} (job_id={job_id}, total: {len(self._started_games)})"
                )

    def record_game_result(
        self,
        info: Dict[str, Any],
        reward: float,
        done: bool,
        instance_id: str = "default",
        job_id: str = "default",
    ):
        """Record a game result. Override in subclass for game-specific metrics.

        Args:
            info: Game info dict
            reward: Reward for this step
            done: Whether the game is finished
            instance_id: Game instance identifier for per-game tracking
            job_id: Job identifier for logging (passed through for debug purposes)
        """
        with self._lock:
            # Track that this game has started (in case reset wasn't called)
            if instance_id not in self._started_games:
                self._started_games.add(instance_id)
                print(
                    f"[DEBUG] New game started on step: {instance_id} (job_id={job_id}, total: {len(self._started_games)})"
                )

            # Initialize game reward tracking if needed
            if instance_id not in self._game_rewards:
                self._game_rewards[instance_id] = {"sum_reward": 0.0, "step_count": 0}

            # Accumulate reward for this game
            self._game_rewards[instance_id]["sum_reward"] += reward
            self._game_rewards[instance_id]["step_count"] += 1

            if done:
                # Calculate per-game statistics
                game_data = self._game_rewards[instance_id]
                sum_reward = game_data["sum_reward"]
                step_count = game_data["step_count"]
                avg_reward = sum_reward / step_count if step_count > 0 else 0.0

                # Record completed game stats
                self._step_stats["game_completed"].append(1.0)
                self._step_stats["final_rewards"].append(reward)
                self._step_stats["sum_rewards"].append(sum_reward)
                self._step_stats["avg_rewards"].append(avg_reward)
                self._step_stats["game_step_counts"].append(float(step_count))
                self._cumulative_stats["total_games"] += 1

                # Clear this game's tracking (game is done)
                del self._game_rewards[instance_id]

            # Track ALL step rewards
            self._step_stats["all_step_rewards"].append(reward)

    def get_step_stats(self) -> Dict[str, Any]:
        """Get statistics for current step."""
        with self._lock:
            stats = {"step": self._current_step}

            completed = self._step_stats.get("game_completed", [])
            final_rewards = self._step_stats.get("final_rewards", [])
            sum_rewards = self._step_stats.get("sum_rewards", [])
            avg_rewards = self._step_stats.get("avg_rewards", [])
            game_step_counts = self._step_stats.get("game_step_counts", [])
            all_rewards = self._step_stats.get("all_step_rewards", [])

            # Total samples = all games that started
            total_samples = len(self._started_games)
            # Incomplete = games that started but haven't completed yet
            incomplete_samples = len(self._game_rewards)
            # Completed games
            completed_count = len(completed)

            stats["total_samples"] = total_samples
            stats["games_in_step"] = completed_count
            stats["incomplete_samples"] = incomplete_samples
            stats["completion_rate"] = (
                completed_count / total_samples if total_samples > 0 else 0.0
            )

            # Mean of FINAL rewards only (last reward of each game)
            if final_rewards:
                stats["mean_final_reward"] = sum(final_rewards) / len(final_rewards)
            else:
                stats["mean_final_reward"] = 0.0

            # Mean of SUM rewards (total reward per game) - COMPLETED games only
            if sum_rewards:
                stats["mean_sum_reward"] = sum(sum_rewards) / len(sum_rewards)
            else:
                stats["mean_sum_reward"] = 0.0

            # Mean of SUM rewards for ALL games (including incomplete ones)
            all_sum_rewards = list(sum_rewards)  # Start with completed games
            for game_data in self._game_rewards.values():
                all_sum_rewards.append(game_data["sum_reward"])

            if all_sum_rewards:
                stats["mean_sum_reward_all"] = sum(all_sum_rewards) / len(
                    all_sum_rewards
                )
            else:
                stats["mean_sum_reward_all"] = 0.0

            # Mean of AVG rewards (average reward per step in each game)
            if avg_rewards:
                stats["mean_avg_reward"] = sum(avg_rewards) / len(avg_rewards)
            else:
                stats["mean_avg_reward"] = 0.0

            # Mean game length
            if game_step_counts:
                stats["mean_game_steps"] = sum(game_step_counts) / len(game_step_counts)
            else:
                stats["mean_game_steps"] = 0.0

            # Keep backward compatibility
            stats["mean_reward"] = stats["mean_final_reward"]
            stats["total_interactions"] = len(all_rewards)

            return stats

    def get_cumulative_stats(self) -> Dict[str, Any]:
        """Get cumulative statistics."""
        with self._lock:
            return dict(self._cumulative_stats)

    def reset_step_stats(self):
        """Reset statistics for a new step/batch."""
        with self._lock:
            # Count incomplete games before clearing
            incomplete_count = len(self._game_rewards)
            started_count = len(self._started_games)

            # Debug logging
            print(
                f"[DEBUG] reset_step_stats called: step {self._current_step} -> {self._current_step + 1}, "
                f"clearing {started_count} started games, {incomplete_count} incomplete"
            )

            if incomplete_count > 0:
                self._cumulative_stats["total_incomplete"] += incomplete_count

            self._step_stats.clear()
            # Clear per-game tracking for incomplete games from previous step
            self._game_rewards.clear()
            # Clear started games set
            self._started_games.clear()
            self._current_step += 1
            return {
                "step": self._current_step,
                "message": "Step stats reset",
                "incomplete_cleared": incomplete_count,
            }

    def reset_all(self):
        """Reset all statistics."""
        with self._lock:
            self._step_stats.clear()
            self._cumulative_stats.clear()
            self._game_rewards.clear()
            self._started_games.clear()
            self._current_step = 0
            return {"message": "All stats reset"}


# Alias for backward compatibility
GameStats = BaseGameStats


class MultiJobGameStats:
    """Multi-job statistics manager.

    Wraps multiple BaseGameStats instances keyed by job_id, enabling
    a single game server to serve multiple training/inference tasks
    with isolated statistics.

    Usage:
        multi_stats = MultiJobGameStats(stats_class=GomokuGameStats)
        stats = multi_stats.get_job_stats("training_job_1")
        stats.record_game_result(...)
    """

    def __init__(self, stats_class: Type[BaseGameStats] = None):
        """Initialize multi-job stats manager.

        Args:
            stats_class: Stats class to instantiate per job. Defaults to BaseGameStats.
        """
        self._stats_class = stats_class or BaseGameStats
        self._job_stats: Dict[str, BaseGameStats] = {}
        self._lock = threading.Lock()

    def get_job_stats(self, job_id: str = "default") -> BaseGameStats:
        """Get or create stats instance for a specific job.

        Args:
            job_id: Job identifier. Defaults to "default" for backward compatibility.

        Returns:
            BaseGameStats (or subclass) instance for the job.
        """
        with self._lock:
            if job_id not in self._job_stats:
                self._job_stats[job_id] = self._stats_class()
                print(f"[MultiJobStats] Created new stats for job_id={job_id}")
            return self._job_stats[job_id]

    def list_jobs(self) -> List[str]:
        """List all active job IDs."""
        with self._lock:
            return list(self._job_stats.keys())

    def remove_job(self, job_id: str) -> bool:
        """Remove a job's statistics.

        Args:
            job_id: Job identifier to remove.

        Returns:
            True if job was removed, False if not found.
        """
        with self._lock:
            if job_id in self._job_stats:
                del self._job_stats[job_id]
                print(f"[MultiJobStats] Removed stats for job_id={job_id}")
                return True
            return False

    def reset_all_jobs(self) -> Dict[str, Any]:
        """Reset all statistics for all jobs."""
        with self._lock:
            job_count = len(self._job_stats)
            self._job_stats.clear()
            return {
                "message": f"All stats reset for {job_count} jobs",
                "jobs_cleared": job_count,
            }


# Pydantic models for request/response validation
class ResetRequest(BaseModel):
    instance_id: str = "default"
    job_id: str = "default"  # Job ID for statistics isolation

    # Allow extra fields for game-specific reset params
    class Config:
        extra = "allow"


class StepRequest(BaseModel):
    instance_id: str = "default"
    job_id: str = "default"  # Job ID for statistics isolation
    action: str = ""


class ResetResponse(BaseModel):
    observation: str
    system_prompt: Optional[str] = None
    game_state: Optional[Dict[str, Any]] = None


class StepResponse(BaseModel):
    observation: str
    reward: float
    done: bool
    info: Dict[str, Any]


def create_game_app(
    game_class: Type[AbstractGame],
    stats_class: Type[BaseGameStats] = None,
    **game_kwargs,
) -> FastAPI:
    """Create a FastAPI app for any AbstractGame implementation.

    Parallelism is achieved via sharding (multiple server processes on consecutive ports).
    Each shard handles requests independently with its own in-memory instance registry.

    Args:
        game_class: The AbstractGame subclass to use
        stats_class: Optional custom stats class (e.g., GomokuGameStats).
                    Defaults to BaseGameStats.
        **game_kwargs: Additional arguments to pass to game constructor
    """
    app = FastAPI(
        title=f"{game_class.__name__} Server",
        description=f"Game server for {game_class.__name__}",
    )

    # Shared state
    games: Dict[str, AbstractGame] = {}
    games_lock = threading.Lock()  # Thread-safe access to games dict

    # Use MultiJobGameStats for job isolation
    multi_stats = MultiJobGameStats(stats_class=stats_class or BaseGameStats)

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    @app.post("/reset")
    async def reset(request: ResetRequest):
        """Reset/create a game instance."""
        instance_id = request.instance_id
        job_id = request.job_id
        # Extract extra fields for game reset (exclude instance_id and job_id)
        reset_kwargs = request.model_dump(exclude={"instance_id", "job_id"})

        with games_lock:
            # Reuse existing game instance if available (avoids re-initialization)
            if instance_id in games:
                game = games[instance_id]
            else:
                game = game_class(**game_kwargs)
                games[instance_id] = game

        # Reset the game (this is the slow part)
        observation = game.reset(**reset_kwargs)

        # Track that this game has started (with job isolation)
        stats = multi_stats.get_job_stats(job_id)
        stats.register_game_start(instance_id, job_id)

        system_prompt = game.get_system_prompt()
        initial_message = game.get_initial_user_message()
        full_observation = (
            f"{initial_message}\n\n{observation}" if observation else initial_message
        )

        response = {
            "observation": full_observation,
            "system_prompt": system_prompt,
        }

        state = game.get_state()
        if state:
            response["game_state"] = state

        return response

    @app.post("/finalize")
    async def finalize(request: ResetRequest):
        """Finalize a game instance and clean up resources."""
        instance_id = request.instance_id
        with games_lock:
            if instance_id in games:
                del games[instance_id]
                return {"message": f"Instance {instance_id} removed"}
        return {"message": f"Instance {instance_id} not found", "status": "ignored"}

    @app.post("/step")
    async def step(request: StepRequest):
        """Execute a step in the game."""
        instance_id = request.instance_id
        job_id = request.job_id
        action = request.action

        if instance_id not in games:
            raise HTTPException(
                status_code=404,
                detail=f"Instance {instance_id} not found. Call /reset first.",
            )

        game = games[instance_id]
        result = game.step(action)

        # Record statistics with instance_id for per-game tracking (with job isolation)
        stats = multi_stats.get_job_stats(job_id)
        stats.record_game_result(
            result.info, result.reward, result.done, instance_id, job_id
        )

        # Clean up finished games
        if result.done:
            with games_lock:
                if instance_id in games:
                    del games[instance_id]

        return {
            "observation": result.observation,
            "reward": result.reward,
            "done": result.done,
            "info": result.info,
        }

    @app.get("/stats")
    async def get_stats(job_id: str = "default"):
        """Get step + cumulative statistics for a specific job."""
        stats = multi_stats.get_job_stats(job_id)
        result = stats.get_step_stats()
        result.update(stats.get_cumulative_stats())
        result["job_id"] = job_id
        return result

    @app.get("/stats/step")
    async def get_step_stats(job_id: str = "default"):
        """Get current step statistics only for a specific job."""
        stats = multi_stats.get_job_stats(job_id)
        result = stats.get_step_stats()
        result["job_id"] = job_id
        return result

    @app.get("/stats/cumulative")
    async def get_cumulative_stats(job_id: str = "default"):
        """Get cumulative statistics for a specific job."""
        stats = multi_stats.get_job_stats(job_id)
        result = stats.get_cumulative_stats()
        result["job_id"] = job_id
        return result

    @app.post("/stats/reset")
    async def reset_stats(job_id: str = "default"):
        """Reset step statistics for a specific job (call before each batch)."""
        stats = multi_stats.get_job_stats(job_id)
        result = stats.reset_step_stats()
        result["job_id"] = job_id
        return result

    @app.post("/stats/reset_all")
    async def reset_all_stats(job_id: Optional[str] = None):
        """Reset all statistics.

        Args:
            job_id: If provided, reset only that job's stats.
                   If None, reset stats for ALL jobs.
        """
        if job_id is not None:
            stats = multi_stats.get_job_stats(job_id)
            result = stats.reset_all()
            result["job_id"] = job_id
            return result
        else:
            return multi_stats.reset_all_jobs()

    @app.get("/stats/jobs")
    async def list_jobs():
        """List all active job IDs with statistics."""
        jobs = multi_stats.list_jobs()
        return {"jobs": jobs, "count": len(jobs)}

    @app.delete("/stats/job/{job_id}")
    async def remove_job_stats(job_id: str):
        """Remove statistics for a specific job."""
        removed = multi_stats.remove_job(job_id)
        if removed:
            return {"message": f"Statistics for job {job_id} removed", "job_id": job_id}
        else:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return app


def create_game_server(
    game_class: Type[AbstractGame],
    host: str = "0.0.0.0",
    port: int = 8081,
    stats_class: Type[BaseGameStats] = None,
    **game_kwargs,
) -> FastAPI:
    """Create a FastAPI app for any AbstractGame implementation.

    Args:
        game_class: The AbstractGame subclass to use
        host: Host address to bind to (not used, for compatibility)
        port: Port to listen on (not used, for compatibility)
        stats_class: Optional custom stats class (e.g., GomokuGameStats).
                    Defaults to BaseGameStats.
        **game_kwargs: Additional arguments to pass to game constructor

    Returns:
        FastAPI app (use run_game_server to actually run it)
    """
    return create_game_app(game_class, stats_class=stats_class, **game_kwargs)


def run_game_server(
    game_class: Type[AbstractGame],
    host: str = "0.0.0.0",
    port: int = 8081,
    stats_class: Type[BaseGameStats] = None,
    **game_kwargs,
):
    """Create and run a game server using FastAPI + uvicorn.

    Parallelism is achieved via sharding (multiple server processes on consecutive ports).
    Use the game-specific server script with --shards N to launch multiple shards.

    Args:
        game_class: The AbstractGame subclass to use
        host: Host address to bind to
        port: Port to listen on
        stats_class: Optional custom stats class (e.g., GomokuGameStats)
        **game_kwargs: Additional arguments to pass to game constructor
    """
    app = create_game_app(
        game_class,
        stats_class=stats_class,
        **game_kwargs,
    )

    print(f"\n{game_class.__name__} Server (FastAPI) running on http://{host}:{port}")
    print("Endpoints:")
    print("  POST /reset       - Reset game")
    print("  POST /step        - Execute action")
    print("  GET  /stats       - Get step + cumulative stats")
    print("  GET  /stats/step  - Get current step stats only")
    print("  POST /stats/reset - Reset step stats (call before each batch)")
    print("  GET  /health      - Health check")
    if game_kwargs:
        print(f"Game config: {game_kwargs}")
    print("\nPress Ctrl+C to stop.")

    uvicorn.run(app, host=host, port=port, log_level="warning")
