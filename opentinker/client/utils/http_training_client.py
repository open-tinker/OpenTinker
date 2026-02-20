#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions.
# limitations under the License.

import logging
import time
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import requests
from tqdm import tqdm

from verl import DataProto
from omegaconf import DictConfig, OmegaConf
from .utils import serialize_dataproto
# from environment import Environment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress repetitive asyncio warnings about socket errors
logging.getLogger("asyncio").setLevel(logging.ERROR)


class HTTPTrainingClient:
    def __init__(
        self,
        server_url: str,
        timeout: float = 6000.0,
        max_retries: int = 1000,  # Increased from 3 to 10 for server initialization
        retry_delay: float = 5.0,  # Increased from 2.0 to 5.0 seconds
    ):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = requests.Session()

        logger.info(f"HTTP Training Client initialized for server: {self.server_url}")

    def _make_request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        url = f"{self.server_url}/api/v1/{endpoint}"
        timeout = timeout or self.timeout

        for attempt in range(self.max_retries):
            try:
                if method.upper() == "GET":
                    response = self.session.get(url, timeout=timeout)
                elif method.upper() == "POST":
                    response = self.session.post(url, json=json_data, timeout=timeout)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                response.raise_for_status()
                return response.json()

            except requests.exceptions.Timeout:
                logger.warning(
                    f"Request to {endpoint} timed out (attempt {attempt + 1}/{self.max_retries})"
                )
            except requests.exceptions.ConnectionError:
                logger.warning(
                    f"Connection error for {endpoint} (attempt {attempt + 1}/{self.max_retries})"
                )
            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP error for {endpoint}: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error for {endpoint}: {e}")
                if attempt == self.max_retries - 1:
                    raise

            if attempt < self.max_retries - 1:
                # Exponential backoff up to 60 seconds max
                wait_time = min(self.retry_delay * (2**attempt), 20.0)
                logger.info(f"Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)

        raise RuntimeError(
            f"Failed to complete request to {endpoint} after {self.max_retries} attempts"
        )

    def health_check(self) -> Dict[str, Any]:
        return self._make_request("GET", "health", timeout=5.0)

    def get_status(self) -> Dict[str, Any]:
        return self._make_request("GET", "status")

    def init_workers(
        self, total_steps: int = 100, timeout: float = 600.0
    ) -> Dict[str, Any]:
        logger.info("Initializing workers on server (this may take several minutes)...")
        result = self._make_request(
            "POST",
            "init_workers",
            json_data={"total_steps": total_steps},
            timeout=timeout,
        )
        logger.info("Workers initialized successfully")
        return result

    def set_generation_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Setting generation config: {config}")
        result = self._make_request("POST", "set_generation_config", json_data=config)
        return result

    def set_config(self, config: Dict[str, Any], env=None) -> Dict[str, Any]:
        """Set configuration on server.

        Args:
            config: Configuration dictionary to send to server
            env: Optional BaseEnvironment instance. If provided, will call env.setup()
                 to upload reward functions and send environment config before
                 sending the main config.
        """
        # If environment is provided, setup reward functions first
        assert env is not None, "Environment must be provided to set_config"
        env_config = env.setup(self)

        logger.info(f"Setting config: {config}")

        if isinstance(config, DictConfig):
            assert env_config is not None, "Env config must be provided to set_config"
            config = OmegaConf.merge(
                config,
                OmegaConf.create(env_config),
            )

        config_payload = OmegaConf.to_container(config, resolve=True)
        result = self._make_request(
            "POST", "set_config", json_data={"config_overrides": config_payload}
        )
        return result

    def train_step(self, batch: DataProto) -> Dict[str, Any]:
        # Serialize batch
        batch_data = serialize_dataproto(batch)

        # Send request
        result = self._make_request(
            "POST", "train_step", json_data={"batch_data": batch_data}
        )

        return result

    def validate(self, batch: DataProto) -> Dict[str, Any]:
        batch_data = serialize_dataproto(batch)
        result = self._make_request(
            "POST", "validate", json_data={"batch_data": batch_data}
        )

        return result

    def save_checkpoint(self) -> Dict[str, Any]:
        logger.info("Requesting checkpoint save...")
        result = self._make_request("POST", "save_checkpoint", json_data={})
        logger.info(f"Checkpoint saved: {result}")
        return result

    def upload_reward_function(
        self, function_name: str, source_code: str
    ) -> Dict[str, Any]:
        """Upload custom reward function code to server.

        Args:
            function_name: Name of the reward function
            source_code: Python source code of the function

        Returns:
            Server response
        """
        logger.info(f"Uploading custom reward function: {function_name}")
        result = self._make_request(
            "POST",
            "upload_reward_function",
            json_data={"function_name": function_name, "source_code": source_code},
        )
        logger.info("Reward function uploaded successfully")
        return result


class SchedulerClient:
    """
    Client for interacting with the job scheduler.

    This class handles job submission to the scheduler and waits for
    server allocation before proceeding with training.
    """

    def __init__(
        self,
        scheduler_url: str,
        api_key: Optional[str] = None,
        timeout: float = 600.0,
        poll_interval: float = 2.0,
    ):
        """
        Initialize scheduler client.

        Args:
            scheduler_url: URL of the job scheduler
            api_key: API key for authentication (if enabled on scheduler)
            timeout: Maximum time to wait for job to start
            poll_interval: Interval between status polls
        """
        self.scheduler_url = scheduler_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.session = requests.Session()

        # Set Authorization header if API key provided
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

        logger.info(f"SchedulerClient initialized for scheduler: {self.scheduler_url}")
        if self.api_key:
            logger.info("Authentication enabled with API key")

    def submit_job(
        self,
        config: Dict[str, Any],
        enable_agent_loop: bool = False,
        wandb_key: Optional[str] = None,
        num_gpus: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Submit a training job to the scheduler.

        Args:
            config: Training configuration dict
            enable_agent_loop: Whether to enable agent loop mode
            wandb_key: WandB API key
            num_gpus: Number of GPUs to request (optional, uses scheduler default if not specified)

        Returns:
            Dict with job_id and server_url
        """
        logger.info("Submitting job to scheduler...")

        # Convert OmegaConf to dict if needed
        if hasattr(config, "__dict__") and "config" in config.__dict__:
            from omegaconf import OmegaConf

            config = OmegaConf.to_container(config, resolve=True)

        # Submit job
        response = self.session.post(
            f"{self.scheduler_url}/submit_job",
            json={
                "config": config,
                "enable_agent_loop": enable_agent_loop,
                "wandb_key": wandb_key,
                "num_gpus": num_gpus,
            },
            timeout=6000.0,
        )
        response.raise_for_status()
        result = response.json()

        job_id = result["job_id"]
        status = result["status"]

        logger.info(f"Job submitted: {job_id}, status: {status}")

        # If job is already running, return immediately
        if result.get("server_url"):
            logger.info(f"Job {job_id} started immediately on {result['server_url']}")
            return result

        # Otherwise, wait for job to start
        logger.info(f"Job {job_id} is queued, waiting for resources...")
        return self._wait_for_job_start(job_id)

    def _wait_for_job_start(self, job_id: str) -> Dict[str, Any]:
        """
        Wait for a queued job to start.

        Args:
            job_id: ID of the job

        Returns:
            Dict with job status including server_url
        """
        start_time = time.time()

        while True:
            # Check timeout
            if time.time() - start_time > self.timeout:
                raise TimeoutError(f"Job {job_id} did not start within {self.timeout}s")

            # Poll job status (increased timeout to 30s to handle scheduler processing multiple jobs)
            response = self.session.get(
                f"{self.scheduler_url}/job_status/{job_id}",
                timeout=6000.0,
            )
            response.raise_for_status()
            status_result = response.json()

            job_status = status_result["status"]

            if job_status == "RUNNING":
                logger.info(f"Job {job_id} started on {status_result['server_url']}")
                return status_result
            elif job_status in ["FAILED", "CANCELLED"]:
                error_msg = status_result.get("error_message", "Unknown error")
                raise RuntimeError(f"Job {job_id} failed: {error_msg}")

            # Still queued or starting, wait and retry
            logger.info(
                f"Job {job_id} status: {job_status}, waiting for RL server to be ready..."
            )
            time.sleep(self.poll_interval)

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """
        Cancel a job.

        Args:
            job_id: ID of the job to cancel

        Returns:
            Cancellation result
        """
        logger.info(f"Cancelling job {job_id}...")
        response = self.session.delete(
            f"{self.scheduler_url}/cancel_job/{job_id}",
            timeout=6000.0,
        )
        response.raise_for_status()
        return response.json()

    def complete_job(self, job_id: str) -> Dict[str, Any]:
        """
        Mark a job as completed.

        Args:
            job_id: ID of the job

        Returns:
            Completion result
        """
        logger.info(f"Marking job {job_id} as completed...")
        response = self.session.post(
            f"{self.scheduler_url}/complete_job/{job_id}",
            timeout=6000.0,
        )
        response.raise_for_status()
        return response.json()


class InferenceSchedulerClient:
    """
    Client for submitting inference jobs to the scheduler.

    This class handles inference job submission that launches vLLM servers
    on the scheduler, with lifecycle management support.
    """

    def __init__(
        self,
        scheduler_url: str,
        api_key: Optional[str] = None,
        timeout: float = 600.0,
        poll_interval: float = 2.0,
    ):
        """
        Initialize inference scheduler client.

        Args:
            scheduler_url: URL of the job scheduler
            api_key: API key for authentication (if enabled on scheduler)
            timeout: Maximum time to wait for job to start
            poll_interval: Interval between status polls
        """
        self.scheduler_url = scheduler_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.session = requests.Session()

        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

        logger.info(
            f"InferenceSchedulerClient initialized for scheduler: {self.scheduler_url}"
        )

    def submit_inference_job(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        tensor_parallel_size: int = 1,
        num_gpus: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        trust_remote_code: bool = True,
    ) -> Dict[str, Any]:
        """
        Submit an inference job to the scheduler.

        This will launch a vLLM server on the scheduler with the specified
        configuration and return the server URL when ready.

        Args:
            model_path: Path to model checkpoint
            tokenizer_path: Tokenizer path (defaults to model_path)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            num_gpus: Number of GPUs requested (overrides tensor_parallel_size if set)
            gpu_memory_utilization: GPU memory fraction to use
            max_model_len: Max model context length (optional)
            trust_remote_code: Whether to trust remote code

        Returns:
            Dict with job_id and vllm_server_url
        """
        logger.info("Submitting inference job to scheduler...")

        response = self.session.post(
            f"{self.scheduler_url}/submit_inference_job",
            json={
                "model_path": model_path,
                "tokenizer_path": tokenizer_path,
                "tensor_parallel_size": tensor_parallel_size,
                "num_gpus": num_gpus,
                "gpu_memory_utilization": gpu_memory_utilization,
                "max_model_len": max_model_len,
                "trust_remote_code": trust_remote_code,
            },
            timeout=6000.0,  # Short timeout for submission (now non-blocking on server)
        )
        response.raise_for_status()
        result = response.json()

        job_id = result["job_id"]
        status = result["status"]

        logger.info(f"Inference job submitted: {job_id}, status: {status}")

        # If job is already running (rare but possible), return immediately
        if status == "RUNNING" and result.get("vllm_server_url"):
            logger.info(
                f"Inference job {job_id} is already running at {result['vllm_server_url']}"
            )
            return result

        # For STARTING or QUEUED status, wait for job to become RUNNING
        # Note: With async startup, jobs now go to STARTING immediately
        logger.info(
            f"Inference job {job_id} status: {status}, waiting for server to be ready..."
        )
        return self._wait_for_job_start(job_id)

    def _wait_for_job_start(self, job_id: str) -> Dict[str, Any]:
        """
        Wait for a queued inference job to start.

        Args:
            job_id: ID of the job

        Returns:
            Dict with job status including vllm_server_url
        """
        start_time = time.time()

        while True:
            if time.time() - start_time > self.timeout:
                raise TimeoutError(
                    f"Inference job {job_id} did not start within {self.timeout}s"
                )

            response = self.session.get(
                f"{self.scheduler_url}/job_status/{job_id}",
                timeout=6000.0,
            )
            response.raise_for_status()
            status_result = response.json()

            job_status = status_result["status"]

            if job_status == "RUNNING":
                # For inference jobs, get vllm_server_url from the stored job info
                # The server_url field may contain it, or we need to construct from port
                vllm_url = status_result.get("server_url")
                if not vllm_url and status_result.get("port"):
                    vllm_url = f"http://localhost:{status_result['port']}"

                logger.info(f"Inference job {job_id} started at {vllm_url}")
                status_result["vllm_server_url"] = vllm_url
                return status_result
            elif job_status in ["FAILED", "CANCELLED"]:
                error_msg = status_result.get("error_message", "Unknown error")
                raise RuntimeError(f"Inference job {job_id} failed: {error_msg}")

            logger.info(
                f"Inference job {job_id} status: {job_status}, waiting for inferenceserver to be ready..."
            )
            time.sleep(self.poll_interval)

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get status of an inference job.

        Args:
            job_id: ID of the job

        Returns:
            Job status information
        """
        response = self.session.get(
            f"{self.scheduler_url}/job_status/{job_id}",
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """
        Cancel an inference job (stops the vLLM server).

        Args:
            job_id: ID of the job to cancel

        Returns:
            Cancellation result
        """
        logger.info(f"Cancelling inference job {job_id}...")
        response = self.session.delete(
            f"{self.scheduler_url}/cancel_job/{job_id}",
            timeout=6000.0,
        )
        response.raise_for_status()
        return response.json()

    def complete_job(self, job_id: str) -> Dict[str, Any]:
        """
        Mark an inference job as completed (stops the vLLM server and releases resources).

        This should be called when the inference task is finished to properly clean up
        the vLLM server process and release GPU resources.

        Args:
            job_id: ID of the job to complete

        Returns:
            Completion result
        """
        logger.info(f"Completing inference job {job_id}...")
        response = self.session.post(
            f"{self.scheduler_url}/complete_job/{job_id}",
            timeout=6000.0,
        )
        response.raise_for_status()
        return response.json()


class ServiceClient:
    def __init__(
        self,
        server_url: str,
        project_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
        logger_backends: Optional[List[str]] = None,
        **client_kwargs,
    ):
        self.client = HTTPTrainingClient(server_url, **client_kwargs)

        # Initialize tracking
        self.tracker = None
        if logger_backends and project_name and experiment_name:
            from verl.utils.tracking import Tracking

            self.tracker = Tracking(
                project_name=project_name,
                experiment_name=experiment_name,
                default_backend=logger_backends,
                config=None,  # Can pass config if needed
            )
            logger.info(f"Initialized tracking with backends: {logger_backends}")

    def set_config(self, args: DictConfig, env=None):
        # 7.5 define general config

        # 同一个key出现两次，后者会将前者覆盖
        server_cfg = OmegaConf.create(
            {
                "data": {
                    "max_prompt_length": args.max_prompt_tokens,
                    "max_response_length": args.max_new_tokens,
                },
                "actor_rollout_ref": {
                    "model": {
                        "path": args.tokenizer_path,
                    },
                    "rollout": {
                        "tensor_model_parallel_size": 2 if args.num_gpus > 1 else 1,
                    },
                },
                "critic": {
                    "model": {
                        "path": args.tokenizer_path,
                    },
                },
                "trainer": {
                    "n_gpus_per_node": args.num_gpus,
                },
            }
        )

        # Add multi_turn config if present in args
        if hasattr(args, "multi_turn") and args.multi_turn:
            multi_turn_cfg = OmegaConf.to_container(args.multi_turn, resolve=True)
            server_cfg = OmegaConf.merge(
                server_cfg,
                OmegaConf.create(
                    {"actor_rollout_ref": {"rollout": {"multi_turn": multi_turn_cfg}}}
                ),
            )
            print(
                f"[ServiceClient] Passing multi_turn config to server: {multi_turn_cfg}"
            )

        # Override agent num_workers if specified (important for single-resource envs like AndroidWorld)
        agent_num_workers = getattr(args, "agent_num_workers", None)
        if agent_num_workers is not None:
            server_cfg = OmegaConf.merge(
                server_cfg,
                OmegaConf.create(
                    {"actor_rollout_ref": {"rollout": {"agent": {"num_workers": agent_num_workers}}}}
                ),
            )
            print(
                f"[ServiceClient] Overriding agent num_workers to: {agent_num_workers}"
            )

        generation_config = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
        }
        self.client.set_generation_config(generation_config)
        self.client.set_config(server_cfg, env)

    def upload_reward_function(self, function_name: str, source_code: str):
        """Upload custom reward function code to server."""
        self.client.upload_reward_function(function_name, source_code)

    def fit(
        self,
        env,
        num_epochs: Optional[int] = None,
        num_steps: Optional[int] = None,
        save_freq: int = 100,
        test_freq: int = 50,
        validate_before_training: bool = True,
        verbose: bool = True,
        game_stats_client=None,
        game_stats_log_freq: int = 1,
    ):
        """
        Train the model.

        Args:
            env: Training environment
            num_epochs: Number of epochs to train (mutually exclusive with num_steps,
                       ignored if num_steps is provided)
            num_steps: Total number of training steps (takes precedence over num_epochs)
            save_freq: Checkpoint save frequency (in steps)
            test_freq: Validation frequency (in steps)
            validate_before_training: Run validation before training starts
            verbose: Show progress bar
            game_stats_client: Optional GameStatsClient for fetching per-step game metrics
            game_stats_log_freq: How often to log game stats (in steps), only used if
                                game_stats_client is provided

        Note:
            - If both num_steps and num_epochs are provided, num_steps takes precedence
            - If neither is provided, defaults to 1 epoch
        """
        self.env = env
        train_dataloader, val_dataloader = env.get_dataloader()
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # 1. Check server health
        try:
            health = self.client.health_check()
            logger.info(f"Server health: {health}")
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            raise RuntimeError(
                f"Cannot connect to server at {self.client.server_url}"
            ) from e

        # 2. Calculate total_steps and effective num_epochs
        try:
            dataloader_len = len(train_dataloader)
        except TypeError:
            raise TypeError(
                "train_dataloader must support len() to calculate total_steps. "
                "Please use a dataloader that implements __len__()."
            )

        # Determine training duration: num_steps takes precedence over num_epochs
        use_step_limit = num_steps is not None

        if use_step_limit:
            total_steps = num_steps
            effective_epochs = (total_steps + dataloader_len - 1) // dataloader_len
            logger.info(
                f"Training for {total_steps} steps (max {effective_epochs} epochs, {dataloader_len} steps per epoch)"
            )
        else:
            if num_epochs is None:
                num_epochs = 1
            effective_epochs = num_epochs
            total_steps = dataloader_len * num_epochs
            logger.info(
                f"Training for {num_epochs} epochs ({dataloader_len} steps per epoch, {total_steps} total steps)"
            )

        # 3. Initialize workers
        self.client.init_workers(total_steps)

        # 4. Run validation before training if requested
        if validate_before_training and val_dataloader:
            logger.info("Running validation before training...")
            val_metrics = self._run_validation(val_dataloader, game_stats_client)
            logger.info(f"Pre-training validation: {val_metrics}")
            if self.tracker:
                self.tracker.log(val_metrics, step=0)

        # 5. Training loop
        global_steps = 0
        last_metrics = {}
        steps_completed = 0

        progress_bar = tqdm(total=total_steps, desc="Training") if verbose else None

        try:
            for epoch in range(effective_epochs):
                if verbose:
                    logger.info(f"Starting epoch {epoch + 1}/{effective_epochs}")

                for batch_dict in train_dataloader:
                    # Reset game stats before each step (if game_stats_client provided)
                    if game_stats_client:
                        try:
                            game_stats_client.reset_step()
                        except Exception as e:
                            logger.warning(f"Failed to reset game stats: {e}")

                    # Convert to DataProto and execute training step
                    batch = DataProto.from_single_dict(batch_dict)
                    result = self.client.train_step(batch)

                    if result["status"] != "success":
                        error_msg = f"Training failed at step {global_steps}: {result.get('error')}"
                        logger.error(error_msg)
                        if "traceback" in result:
                            logger.error(f"Server traceback:\n{result['traceback']}")
                        raise RuntimeError(error_msg)

                    global_steps = result["global_steps"]
                    last_metrics = result["metrics"]

                    # Fetch and log game stats (if game_stats_client provided)
                    if game_stats_client and global_steps % game_stats_log_freq == 0:
                        try:
                            game_metrics = game_stats_client.get_step_stats()
                            if game_metrics:
                                # Filter out 'step' to avoid confusion - it can get out of sync
                                # due to validation resets. Use global_steps instead.
                                prefixed = {
                                    f"game/{k}": v
                                    for k, v in game_metrics.items()
                                    if isinstance(v, (int, float)) and k != "step"
                                }
                                last_metrics.update(prefixed)
                        except Exception as e:
                            logger.warning(f"Failed to get game stats: {e}")

                    # Log metrics to tracking backends
                    if self.tracker:
                        metrics_to_log = last_metrics.copy()
                        metrics_to_log["epoch"] = epoch + 1
                        self.tracker.log(metrics_to_log, step=global_steps)

                    # Update progress bar
                    if verbose and progress_bar:
                        # Show key metrics in progress bar (filter game/ metrics except win_rate)
                        display_metrics = {
                            k: v
                            for k, v in last_metrics.items()
                            if not k.startswith("game/") or k == "game/win_rate"
                        }
                        metrics_str = ", ".join(
                            [
                                f"{k}: {v:.4f}"
                                for k, v in list(display_metrics.items())[:5]
                            ]
                        )
                        epoch_str = (
                            f"Epoch {epoch + 1}/{effective_epochs}"
                            if not use_step_limit
                            else f"Epoch {epoch + 1}"
                        )
                        progress_bar.set_postfix_str(f"{epoch_str}, {metrics_str}")
                        progress_bar.update(1)

                    steps_completed += 1

                    # Check if we've reached the step limit
                    if use_step_limit and steps_completed >= total_steps:
                        logger.info(
                            f"Reached target of {total_steps} steps, stopping training."
                        )
                        break

                    # Validation
                    if (
                        val_dataloader
                        and test_freq > 0
                        and global_steps % test_freq == 0
                    ):
                        val_metrics = self._run_validation(
                            val_dataloader, game_stats_client
                        )
                        logger.info(
                            f"Validation @ epoch {epoch + 1}, step {global_steps}: {val_metrics}"
                        )
                        if self.tracker:
                            val_metrics_to_log = val_metrics.copy()
                            val_metrics_to_log["epoch"] = epoch + 1
                            self.tracker.log(val_metrics_to_log, step=global_steps)

                    # Checkpoint saving
                    if save_freq > 0 and global_steps % save_freq == 0:
                        ckpt_result = self.client.save_checkpoint()
                        logger.info(
                            f"Checkpoint saved @ epoch {epoch + 1}, step {global_steps}: {ckpt_result['result']['checkpoint_dir']}"
                        )

                # Check if we've reached the step limit (after inner loop)
                if use_step_limit and steps_completed >= total_steps:
                    break

        except Exception as e:
            logger.error(f"Training loop failed with exception: {e}")
            import traceback

            traceback.print_exc()
            raise
        finally:
            if progress_bar:
                progress_bar.close()

        # 6. Run final validation
        if val_dataloader:
            logger.info("Running final validation after training...")
            val_metrics = self._run_validation(val_dataloader, game_stats_client)
            logger.info(f"Final validation @ step {global_steps}: {val_metrics}")
            if self.tracker:
                val_metrics_to_log = val_metrics.copy()
                val_metrics_to_log["epoch"] = effective_epochs
                self.tracker.log(val_metrics_to_log, step=global_steps)

        if use_step_limit:
            logger.info(
                f"Training completed! {steps_completed} steps, final step: {global_steps}"
            )
        else:
            logger.info(
                f"Training completed! {num_epochs} epochs, final step: {global_steps}"
            )
        return last_metrics

    def _run_validation(
        self, val_dataloader: Iterator, game_stats_client=None
    ) -> Dict[str, float]:
        """Run validation on all batches and aggregate metrics.

        Args:
            val_dataloader: Validation data iterator
            game_stats_client: Optional GameStatsClient for fetching game metrics

        Returns:
            Aggregated validation metrics including game stats if available
        """
        all_metrics = []

        logger.info("Running validation...")

        # Reset game stats before validation (if game_stats_client provided)
        if game_stats_client:
            try:
                game_stats_client.reset_step()
            except Exception as e:
                logger.warning(f"Failed to reset game stats for validation: {e}")

        for i, batch_dict in enumerate(val_dataloader):
            batch = DataProto.from_single_dict(batch_dict)
            result = self.client.validate(batch)

            if result["status"] != "success":
                logger.warning(f"Validation batch {i} failed: {result.get('error')}")
                continue

            all_metrics.append(result["metrics"])

        if not all_metrics:
            return {}

        aggregated = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            aggregated[key] = float(np.mean(values))

        # Fetch and add game stats (if game_stats_client provided)
        if game_stats_client:
            try:
                game_metrics = game_stats_client.get_step_stats()
                if game_metrics:
                    # Filter out 'step' to avoid confusion with global training step
                    prefixed = {
                        f"val_game/{k}": v
                        for k, v in game_metrics.items()
                        if isinstance(v, (int, float)) and k != "step"
                    }
                    aggregated.update(prefixed)
                    logger.info(
                        f"Validation game stats: win_rate={game_metrics.get('win_rate', 0):.2%}"
                    )
            except Exception as e:
                logger.warning(f"Failed to get validation game stats: {e}")

        return aggregated
