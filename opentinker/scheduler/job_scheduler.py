#!/usr/bin/env python3
"""
Job Scheduler for OpenTinker Training

This module implements a centralized job scheduler that manages multiple
training jobs across GPU resources. It automatically allocates GPUs,
launches HTTP training servers, and handles job lifecycle management.
"""

import logging
import os
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import ray
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import user management
try:
    from user_management import UserManager, User
except ImportError:
    from .user_management import UserManager, User

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ==================== Utility Functions ====================


def find_available_ports(num_ports: int = 50, start_port: int = 38000) -> List[int]:
    """
    Find available ports by attempting to bind to them.

    Args:
        num_ports: Number of ports to find
        start_port: Starting port to search from

    Returns:
        List of available port numbers
    """
    import socket

    available_ports = []
    port = start_port
    max_attempts = num_ports * 10  # Don't search forever
    attempts = 0

    while len(available_ports) < num_ports and attempts < max_attempts:
        try:
            # Try to bind to the port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                # If successful, port is available
                available_ports.append(port)
                logger.debug(f"Port {port} is available")
        except OSError:
            # Port is in use, try next one
            logger.debug(f"Port {port} is in use")

        port += 1
        attempts += 1

    if len(available_ports) < num_ports:
        logger.warning(
            f"Could only find {len(available_ports)} available ports (requested {num_ports})"
        )

    return available_ports


def detect_gpu_topology() -> List[List[int]]:
    """
    Detect GPU topology using nvidia-smi to group GPUs by proximity.

    Groups GPUs based on NUMA nodes and PCIe connectivity. GPUs in the same
    group have better interconnect bandwidth (PIX/PXB) compared to cross-group
    connections (SYS).

    Returns:
        List of GPU groups, where each group is a list of GPU IDs with close topology.
        Falls back to single group with all GPUs if detection fails.
    """
    try:
        import subprocess

        # Run nvidia-smi to get topology matrix
        result = subprocess.run(
            ["nvidia-smi", "topo", "--matrix"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            logger.warning(
                "⚠️ GPU topology detection failed: nvidia-smi topo command returned error"
            )
            logger.warning(
                "   Falling back to sequential GPU allocation (may have reduced performance)"
            )
            logger.warning(
                "   Tip: Ensure nvidia-smi is properly installed and GPU drivers are loaded"
            )
            return []

        lines = result.stdout.strip().split("\n")

        # Parse the topology matrix
        # First line contains GPU headers
        header_line = None
        matrix_lines = []

        for line in lines:
            if line.strip().startswith("GPU"):
                if header_line is None:
                    header_line = line
                else:
                    matrix_lines.append(line)

        if not header_line or not matrix_lines:
            logger.warning(
                "⚠️ GPU topology detection failed: could not parse nvidia-smi output"
            )
            logger.warning("   Falling back to sequential GPU allocation")
            return []

        # Extract GPU IDs from header
        gpu_ids = []
        parts = header_line.split()
        for part in parts:
            if part.startswith("GPU"):
                try:
                    gpu_id = int(part[3:])  # Extract number from "GPU0", "GPU1", etc.
                    gpu_ids.append(gpu_id)
                except ValueError:
                    continue

        if not gpu_ids:
            logger.warning(
                "⚠️ GPU topology detection failed: no GPU IDs found in topology matrix"
            )
            logger.warning("   Falling back to sequential GPU allocation")
            return []

        logger.info(f"Detected {len(gpu_ids)} GPUs from topology: {gpu_ids}")

        # Parse NUMA affinity to group GPUs
        # Look for "NUMA Affinity" column
        numa_groups = {}

        for line in matrix_lines:
            parts = line.split()
            if not parts or not parts[0].startswith("GPU"):
                continue

            try:
                gpu_id = int(parts[0][3:])
            except (ValueError, IndexError):
                continue

            # Find NUMA affinity (second to last column before "GPU NUMA ID")
            # Format is usually like "0-23,48-71" or just a number "0" or "1"
            try:
                # NUMA affinity is typically at index -2 (before "GPU NUMA ID")
                numa_str = parts[-2]

                # Extract the first number as NUMA node ID
                if "-" in numa_str or "," in numa_str:
                    # Format like "0-23,48-71" - extract first number
                    numa_node = int(numa_str.split("-")[0].split(",")[0])
                else:
                    # Format like "0" or "1"
                    numa_node = int(numa_str)

                if numa_node not in numa_groups:
                    numa_groups[numa_node] = []
                numa_groups[numa_node].append(gpu_id)

            except (ValueError, IndexError) as e:
                logger.debug(f"Could not parse NUMA affinity for GPU{gpu_id}: {e}")
                continue

        # Convert to list of groups
        topology_groups = [sorted(gpus) for gpus in numa_groups.values()]

        if topology_groups:
            logger.info(f"Detected {len(topology_groups)} GPU topology groups:")
            for i, group in enumerate(topology_groups):
                logger.info(f"  Group {i}: GPUs {group} (NUMA node {i})")
            return topology_groups
        else:
            logger.warning("⚠️ GPU topology detection: no NUMA groups detected")
            logger.warning("   Falling back to sequential GPU allocation")
            logger.warning(
                "   Note: This may result in sub-optimal GPU placement for multi-GPU jobs"
            )
            return []

    except subprocess.TimeoutExpired:
        logger.warning(
            "⚠️ GPU topology detection failed: nvidia-smi topo command timed out after 10s"
        )
        logger.warning("   Falling back to sequential GPU allocation")
        return []
    except FileNotFoundError:
        logger.warning(
            "⚠️ GPU topology detection disabled: nvidia-smi not found in PATH"
        )
        logger.warning(
            "   Using sequential GPU allocation (this is expected in CPU-only environments)"
        )
        return []
    except Exception as e:
        logger.warning(f"⚠️ GPU topology detection failed with unexpected error: {e}")
        logger.warning("   Falling back to sequential GPU allocation")
        return []


def check_gpu_available(gpu_id: int) -> bool:
    """
    Check if a GPU is actually available (idle) by querying nvidia-smi.

    This prevents allocation conflicts when:
    - External processes are using GPUs
    - Scheduler restarts and loses allocation state
    - Jobs crash without properly releasing resources

    Args:
        gpu_id: GPU ID to check

    Returns:
        True if GPU is idle and available, False if occupied or check fails
    """
    try:
        import subprocess

        # Check 1: Query GPU memory usage and utilization
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
                f"--id={gpu_id}",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            logger.warning(
                f"GPU {gpu_id}: Failed to query status via nvidia-smi, assuming available"
            )
            return True  # Fail open to avoid breaking existing behavior

        # Parse output: "0, 123, 5" means GPU 0, 123 MB used, 5% utilization
        parts = result.stdout.strip().split(",")
        if len(parts) < 3:
            logger.warning(
                f"GPU {gpu_id}: Could not parse nvidia-smi output: {result.stdout}"
            )
            return True  # Fail open

        try:
            memory_used_mb = int(parts[1].strip())
            utilization_percent = int(parts[2].strip())
        except ValueError:
            logger.warning(
                f"GPU {gpu_id}: Could not parse usage values: {result.stdout}"
            )
            return True  # Fail open

        # Thresholds for considering a GPU "idle"
        MAX_MEMORY_MB = 10  # Allow up to 100 MB (some baseline CUDA overhead)
        MAX_UTILIZATION = 1000  # Allow up to 5% utilization

        if memory_used_mb > MAX_MEMORY_MB or utilization_percent > MAX_UTILIZATION:
            logger.warning(
                f"GPU {gpu_id}: ⚠️ OCCUPIED - Memory: {memory_used_mb} MB, "
                f"Utilization: {utilization_percent}% (thresholds: {MAX_MEMORY_MB} MB, {MAX_UTILIZATION}%)"
            )
            return False

        # Check 2: Look for running processes on this GPU
        pmon_result = subprocess.run(
            ["nvidia-smi", "pmon", "-c", "1", "-s", "um"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if pmon_result.returncode == 0:
            # Parse pmon output to check for processes on this GPU
            # Format: "# gpu   pid  type   sm  mem   enc   dec   command"
            #         "  0   12345   C    50   500     0     0   python"
            lines = pmon_result.stdout.strip().split("\n")
            for line in lines:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        gpu_idx = int(parts[0].strip())
                        if gpu_idx == gpu_id and parts[1].strip() != "-":
                            # Found a process on this GPU
                            pid = parts[1].strip()
                            logger.warning(
                                f"GPU {gpu_id}: ⚠️ OCCUPIED - Process {pid} detected via pmon"
                            )
                            return False
                    except (ValueError, IndexError):
                        continue

        # All checks passed - GPU is idle
        logger.debug(
            f"GPU {gpu_id}: ✅ Available (Memory: {memory_used_mb} MB, Utilization: {utilization_percent}%)"
        )
        return True

    except subprocess.TimeoutExpired:
        logger.warning(f"GPU {gpu_id}: nvidia-smi check timed out, assuming available")
        return True  # Fail open
    except FileNotFoundError:
        logger.debug(f"GPU {gpu_id}: nvidia-smi not found, assuming available")
        return True  # Fail open - nvidia-smi not available
    except Exception as e:
        logger.warning(
            f"GPU {gpu_id}: Occupancy check failed with error: {e}, assuming available"
        )
        return True  # Fail open


class JobStatus(str, Enum):
    """Job status enumeration"""

    QUEUED = "QUEUED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class JobType(str, Enum):
    """Job type enumeration"""

    TRAINING = "TRAINING"
    INFERENCE = "INFERENCE"


@dataclass
class JobInfo:
    """Information about a training or inference job"""

    job_id: str
    status: JobStatus
    config: Dict[str, Any]
    num_gpus: int  # Number of GPUs requested for this job
    job_type: JobType = JobType.TRAINING  # Job type (training or inference)
    user_id: Optional[str] = None  # User who submitted the job
    username: Optional[str] = None  # Username for display
    gpu_ids: Optional[List[int]] = None  # List of allocated GPU IDs
    port: Optional[int] = None
    server_url: Optional[str] = None
    vllm_server_url: Optional[str] = None  # vLLM server URL for inference jobs
    process: Optional[subprocess.Popen] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


# ==================== API Models ====================


class SubmitJobRequest(BaseModel):
    """Request model for job submission"""

    config: Dict[str, Any]
    enable_agent_loop: bool = False
    wandb_key: Optional[str] = None
    num_gpus: Optional[int] = None  # Number of GPUs requested by client


class SubmitJobResponse(BaseModel):
    """Response model for job submission"""

    job_id: str
    status: str
    message: str
    server_url: Optional[str] = None


class JobStatusResponse(BaseModel):
    """Response model for job status query"""

    job_id: str
    status: str
    gpu_ids: Optional[List[int]] = None
    port: Optional[int] = None
    server_url: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None


class ListJobsResponse(BaseModel):
    """Response model for listing jobs"""

    jobs: List[JobStatusResponse]


class CancelJobResponse(BaseModel):
    """Response model for job cancellation"""

    job_id: str
    status: str
    message: str


class SubmitInferenceJobRequest(BaseModel):
    """Request model for inference job submission"""

    model_path: str  # Path to model checkpoint
    tokenizer_path: Optional[str] = None  # Tokenizer path (defaults to model_path)
    tensor_parallel_size: int = 1  # Number of GPUs for tensor parallelism
    num_gpus: Optional[int] = (
        None  # Number of GPUs requested (overrides tensor_parallel_size)
    )
    gpu_memory_utilization: float = 0.9  # GPU memory fraction to use
    max_model_len: Optional[int] = None  # Max model context length
    trust_remote_code: bool = True  # Whether to trust remote code


class SubmitInferenceJobResponse(BaseModel):
    """Response model for inference job submission"""

    job_id: str
    status: str
    message: str
    vllm_server_url: Optional[str] = None


# ==================== Ray Actor: Job Scheduler ====================


@ray.remote
class JobSchedulerActor:
    """
    Ray actor that manages job queue, GPU allocation, and server processes.
    """

    def __init__(
        self,
        available_gpus: List[int],
        port_range: Optional[tuple] = None,
        server_script_path: str = "",
        base_dir: str = "",
        num_ports: int = 50,
        gpus_per_job: int = 4,
        logs_dir: str = "/workspace/logs",
    ):
        """
        Initialize the job scheduler actor.

        Args:
            available_gpus: List of GPU IDs available for allocation
            port_range: Optional tuple of (min_port, max_port) for spawned servers.
                       If None, will auto-detect available ports.
            server_script_path: Path to launch_http_server.py
            base_dir: Base directory for the project (for subprocess cwd)
            num_ports: Number of ports to detect if port_range is None (default: 50)
            gpus_per_job: Number of GPUs to allocate per job (default: 4)
            logs_dir: Directory to store job logs (default: /workspace/logs)
        """
        self.available_gpus: List[int] = list(available_gpus)
        self.allocated_gpus: Set[int] = set()
        self.gpus_per_job = gpus_per_job

        # Store original GPU set for validation during resource release
        # This prevents unauthorized GPUs from entering the available pool
        self._original_gpu_set: Set[int] = set(available_gpus)

        # Scheduling lock to prevent concurrent job scheduling
        self._scheduling_in_progress = False

        # Submission lock to prevent concurrent job submission (GPU allocation race condition)
        self._submission_in_progress = False

        # Configure logs directory
        self.logs_dir = Path(logs_dir)
        try:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Job logs will be stored in: {self.logs_dir}")
        except Exception as e:
            logger.error(f"Failed to create logs directory {self.logs_dir}: {e}")
            # Fall back to a safe default
            self.logs_dir = Path.cwd() / "scheduler_logs"
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Using fallback logs directory: {self.logs_dir}")

        # Detect GPU topology for smart allocation
        logger.info("=" * 60)
        logger.info("🔍 Detecting GPU topology...")
        self.gpu_topology_groups: List[List[int]] = detect_gpu_topology()

        # Filter topology groups to only include available GPUs
        if self.gpu_topology_groups:
            available_set = set(available_gpus)
            self.gpu_topology_groups = [
                [gpu for gpu in group if gpu in available_set]
                for group in self.gpu_topology_groups
            ]
            # Remove empty groups
            self.gpu_topology_groups = [g for g in self.gpu_topology_groups if g]

            if self.gpu_topology_groups:
                logger.info("✅ GPU Topology Detection: SUCCESS")
                logger.info(
                    f"   Using topology-aware GPU allocation with {len(self.gpu_topology_groups)} NUMA groups:"
                )
                for i, group in enumerate(self.gpu_topology_groups):
                    logger.info(f"   - Group {i}: GPUs {group}")
                logger.info(
                    "   Multi-GPU jobs will prefer GPUs from the same group for better performance"
                )
            else:
                logger.warning("⚠️ GPU Topology Detection: PARTIAL")
                logger.warning(
                    "   No valid topology groups after filtering available GPUs"
                )
                logger.warning(
                    "   Using sequential allocation (may have reduced multi-GPU performance)"
                )
        else:
            logger.warning("⚠️ GPU Topology Detection: DISABLED")
            logger.warning("   Using sequential GPU allocation")
            logger.warning(
                "   Tip: For optimal multi-GPU performance, ensure nvidia-smi is available"
            )
        logger.info("=" * 60)

        # Auto-detect ports or use specified range
        if port_range is None:
            logger.info(f"Auto-detecting {num_ports} available ports...")
            self.available_ports: List[int] = find_available_ports(num_ports=num_ports)
            logger.info(f"Auto-detected {len(self.available_ports)} available ports")
        else:
            self.available_ports: List[int] = list(
                range(port_range[0], port_range[1] + 1)
            )
            logger.info(
                f"Using port range: {port_range} -> {len(self.available_ports)} ports"
            )

        self.allocated_ports: Set[int] = set()

        self.server_script_path = server_script_path
        self.base_dir = base_dir

        # Job storage
        self.jobs: Dict[str, JobInfo] = {}
        self.job_queue: List[str] = []  # Queue of job IDs waiting for resources

        # Background threads for async vLLM startup
        self._startup_threads: Dict[str, threading.Thread] = {}

        # Retry mechanism for GPU availability
        self._pending_retry_count = 0
        self._max_retries = 15  # Retry up to 15 times
        self._retry_interval = 2.0  # 2 seconds between retries (total ~30s)

        logger.info(f"JobSchedulerActor initialized with GPUs: {available_gpus}")
        logger.info(f"GPUs per job: {gpus_per_job}")
        if len(self.available_ports) > 10:
            logger.info(
                f"Available ports: {self.available_ports[:5]}...{self.available_ports[-5:]}"
            )
        else:
            logger.info(f"Available ports: {self.available_ports}")

    def submit_job(
        self,
        config: Dict[str, Any],
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        enable_agent_loop: bool = False,
        wandb_key: Optional[str] = None,
        num_gpus: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Submit a new training job.

        Args:
            config: Training configuration dict
            user_id: ID of user submitting the job
            username: Username for display
            enable_agent_loop: Whether to enable agent loop mode
            wandb_key: WandB API key

        Returns:
            Dict with job_id, status, and server_url (if immediately started)
        """
        job_id = str(uuid.uuid4())[:8]  # Short UUID

        # Store job configuration
        job_config = {
            **config,
            "enable_agent_loop": enable_agent_loop,
            "wandb_key": wandb_key or "",
        }

        # Use client-requested GPU count, or fall back to scheduler's default
        requested_gpus = num_gpus if num_gpus is not None else self.gpus_per_job

        job = JobInfo(
            job_id=job_id,
            status=JobStatus.QUEUED,
            config=job_config,
            num_gpus=requested_gpus,
            user_id=user_id,
            username=username,
        )

        self.jobs[job_id] = job

        logger.info(f"Job {job_id} submitted (requesting {requested_gpus} GPUs)")

        # Use submission lock to prevent concurrent allocation race conditions
        # Wait for any ongoing submission to complete
        wait_count = 0
        while self._submission_in_progress:
            if wait_count == 0:
                logger.info(
                    f"Job {job_id}: ⏳ Waiting for concurrent submission to complete..."
                )
            wait_count += 1
            time.sleep(0.1)

        if wait_count > 0:
            logger.info(
                f"Job {job_id}: Proceeding after waiting {wait_count * 0.1:.1f}s for submission lock"
            )

        self._submission_in_progress = True
        try:
            # Try to allocate resources immediately
            if self._try_start_job(job_id):
                return {
                    "job_id": job_id,
                    "status": job.status.value,
                    "server_url": job.server_url,
                    "message": "Job started successfully",
                }
            else:
                # Add to queue
                self.job_queue.append(job_id)
                logger.info(
                    f"Job {job_id} added to queue (position {len(self.job_queue)})"
                )
                return {
                    "job_id": job_id,
                    "status": JobStatus.QUEUED.value,
                    "server_url": None,
                    "message": f"Job queued (position {len(self.job_queue)})",
                }
        finally:
            self._submission_in_progress = False

    def _try_start_job(self, job_id: str) -> bool:
        """
        Try to allocate resources and start a job.

        Args:
            job_id: ID of the job to start

        Returns:
            True if job was started, False if resources unavailable
        """
        job = self.jobs[job_id]

        # Use the job's requested GPU count
        required_gpus = job.num_gpus

        logger.info(f"Job {job_id}: Attempting to allocate {required_gpus} GPUs")
        logger.info(
            f"  Current state: {len(self.available_gpus)} available, {len(self.allocated_gpus)} allocated"
        )

        # Check if enough resources are available
        if len(self.available_gpus) < required_gpus or not self.available_ports:
            logger.info(
                f"Job {job_id}: Insufficient resources (need {required_gpus} GPUs, have {len(self.available_gpus)})"
            )
            return False

        # Allocate GPUs using topology-aware selection
        gpu_ids = self._select_gpus_by_topology(required_gpus)

        if not gpu_ids:
            logger.info(f"Job {job_id}: Could not find suitable GPU allocation")
            return False

        # Try to launch server with port retry mechanism to handle TOCTOU race conditions
        max_port_retries = 3
        used_ports = []  # Track ports we've tried

        for retry in range(max_port_retries):
            if not self.available_ports:
                logger.warning(f"Job {job_id}: No more available ports to try")
                # Release GPUs since we can't get a port
                for gpu_id in gpu_ids:
                    if gpu_id in self.allocated_gpus:
                        self.allocated_gpus.remove(gpu_id)
                        # Only add back if it belongs to original configured GPUs
                        if gpu_id in self._original_gpu_set:
                            self.available_gpus.append(gpu_id)
                        else:
                            logger.error(
                                f"Job {job_id}: ⚠️ GPU {gpu_id} not in original config, skipping release"
                            )
                return False

            port = self.available_ports.pop(0)
            self.allocated_ports.add(port)
            used_ports.append(port)

            job.gpu_ids = gpu_ids
            job.port = port
            # Use actual hostname instead of 0.0.0.0 for client connections
            # 0.0.0.0 is valid for binding but not for connecting
            import socket

            hostname = socket.gethostname()
            # Try to get a routable IP address
            try:
                # Get the IP address that would be used to connect to external networks
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                ip_address = s.getsockname()[0]
                s.close()
            except Exception:
                # Fall back to hostname resolution
                try:
                    ip_address = socket.gethostbyname(hostname)
                except Exception:
                    ip_address = "127.0.0.1"  # Last resort fallback
            job.server_url = f"http://{ip_address}:{port}"
            job.status = JobStatus.STARTING
            if retry == 0:
                job.started_at = datetime.now()

            logger.info(
                f"Job {job_id}: Allocated GPUs {gpu_ids}, trying port {port} (attempt {retry + 1}/{max_port_retries})"
            )

            # Launch server process
            try:
                process = self._launch_server(job)
                job.process = process
                logger.info(f"Job {job_id}: Server process started (PID {process.pid})")

                # Wait for server to be ready before marking as RUNNING
                if not self._wait_for_server_ready(job, timeout=60.0, interval=2.0):
                    logger.error(f"Job {job_id}: Failed to start - server not ready")
                    job.status = JobStatus.FAILED
                    job.error_message = "Server failed to become ready within timeout"
                    job.completed_at = datetime.now()

                    # Terminate the process
                    if job.process and job.process.poll() is None:
                        job.process.terminate()
                        try:
                            job.process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            job.process.kill()

                    # Release resources
                    self._release_resources(job_id)
                    return False

                job.status = JobStatus.RUNNING
                logger.info(
                    f"Job {job_id}: ✅ Server started and ready (PID {process.pid})"
                )

                # Return unused ports back to pool (keep only the one we successfully used)
                for p in used_ports[:-1]:
                    if p in self.allocated_ports:
                        self.allocated_ports.remove(p)
                        self.available_ports.append(p)

                return True

            except OSError as e:
                # Port binding error - try next port
                if (
                    "Address already in use" in str(e)
                    or "address in use" in str(e).lower()
                ):
                    logger.warning(
                        f"Job {job_id}: Port {port} already in use (TOCTOU race condition), trying next port..."
                    )
                    # Don't return this port to the pool immediately - it's genuinely in use
                    if port in self.allocated_ports:
                        self.allocated_ports.remove(port)
                    continue
                else:
                    # Other OSError - not a port issue
                    logger.error(f"Job {job_id}: Failed to start server: {e}")
                    job.status = JobStatus.FAILED
                    job.error_message = str(e)
                    job.completed_at = datetime.now()
                    self._release_resources(job_id)
                    return False

            except Exception as e:
                logger.error(f"Job {job_id}: Failed to start server: {e}")
                job.status = JobStatus.FAILED
                job.error_message = str(e)
                job.completed_at = datetime.now()
                self._release_resources(job_id)
                return False

        # All port retries exhausted
        logger.error(
            f"Job {job_id}: Failed to start after {max_port_retries} port allocation attempts"
        )
        job.status = JobStatus.FAILED
        job.error_message = f"Port allocation failed after {max_port_retries} retries (all ports in use)"
        job.completed_at = datetime.now()

        # Release GPUs
        for gpu_id in gpu_ids:
            if gpu_id in self.allocated_gpus:
                self.allocated_gpus.remove(gpu_id)
                # Only add back if it belongs to original configured GPUs
                if gpu_id in self._original_gpu_set:
                    self.available_gpus.append(gpu_id)
                else:
                    logger.error(
                        f"Job {job_id}: ⚠️ GPU {gpu_id} not in original config, skipping release"
                    )

        return False

    def _select_gpus_by_topology(self, num_gpus: int) -> List[int]:
        """
        Select GPUs based on topology, preferring GPUs from the same group.

        Now includes physical occupancy checks via nvidia-smi to prevent allocation
        conflicts with external processes or after scheduler restarts.

        Args:
            num_gpus: Number of GPUs to allocate

        Returns:
            List of selected GPU IDs, or empty list if allocation not possible
        """
        logger.debug(
            f"🔍 Selecting {num_gpus} GPUs from available: {self.available_gpus}"
        )

        # First, filter available_gpus to only include physically idle GPUs
        # This catches cases where scheduler state is stale (e.g., after restart)
        truly_available = []
        occupied_count = 0

        for gpu_id in self.available_gpus:
            if check_gpu_available(gpu_id):
                truly_available.append(gpu_id)
            else:
                occupied_count += 1
                logger.warning(
                    f"GPU {gpu_id}: Marked as available in scheduler state but is "
                    f"actually OCCUPIED. Skipping from allocation."
                )

        if occupied_count > 0:
            logger.warning(
                f"⚠️ Found {occupied_count} GPU(s) marked available but actually occupied. "
                f"This may indicate external processes or stale scheduler state."
            )

        logger.debug(
            f"🔍 Physically available GPUs after occupancy check: {truly_available}"
        )

        if not self.gpu_topology_groups:
            # Fallback to sequential allocation if no topology info
            if len(truly_available) >= num_gpus:
                selected = []
                for _ in range(num_gpus):
                    gpu_id = truly_available[0]
                    truly_available.remove(gpu_id)
                    self.available_gpus.remove(gpu_id)
                    selected.append(gpu_id)
                    self.allocated_gpus.add(gpu_id)
                logger.info(f"📌 Allocated GPUs (sequential): {selected}")
                logger.debug(f"   Remaining available: {self.available_gpus}")
                return selected
            return []

        # Try to allocate from a single topology group first
        available_set = set(truly_available)

        for group_idx, group in enumerate(self.gpu_topology_groups):
            # Find available GPUs in this group (that are also physically idle)
            available_in_group = [gpu for gpu in group if gpu in available_set]

            if len(available_in_group) >= num_gpus:
                # We can allocate all GPUs from this group
                selected = available_in_group[:num_gpus]

                # Remove from available list and add to allocated
                for gpu_id in selected:
                    self.available_gpus.remove(gpu_id)
                    self.allocated_gpus.add(gpu_id)

                logger.info(
                    f"📌 Allocated GPUs from topology group {group_idx}: {selected}"
                )
                logger.debug(f"   Remaining available: {self.available_gpus}")
                return selected

        # If no single group has enough GPUs, fall back to sequential allocation
        # (cross-group allocation)
        if len(truly_available) >= num_gpus:
            selected = []
            for _ in range(num_gpus):
                gpu_id = truly_available[0]
                truly_available.remove(gpu_id)
                self.available_gpus.remove(gpu_id)
                selected.append(gpu_id)
                self.allocated_gpus.add(gpu_id)
            logger.warning(f"⚠️ Allocated GPUs across topology groups: {selected}")
            logger.debug(f"   Remaining available: {self.available_gpus}")
            return selected

        return []

    def _wait_for_server_ready(
        self, job: JobInfo, timeout: float = 60.0, interval: float = 2.0
    ) -> bool:
        """
        Wait for HTTP server to be ready by polling health endpoint.

        Args:
            job: JobInfo object
            timeout: Maximum time to wait in seconds
            interval: Polling interval in seconds

        Returns:
            True if server is ready, False if timeout or process crashed
        """
        import requests

        start_time = time.time()
        # Use 127.0.0.1 for health check since it's local to the scheduler
        # (job.server_url uses the external IP for client connections)
        health_url = f"http://127.0.0.1:{job.port}/api/v1/health"

        logger.info(f"Job {job.job_id}: Waiting for server to be ready at {health_url}")

        while time.time() - start_time < timeout:
            # First, check if the process is still running
            if job.process and job.process.poll() is not None:
                exit_code = job.process.poll()
                logger.error(
                    f"Job {job.job_id}: ❌ Process crashed during startup (exit code {exit_code})"
                )

                # Try to read error from stderr log file
                if hasattr(job.process, "stderr_log_path"):
                    try:
                        with open(job.process.stderr_log_path, "r") as f:
                            stderr_content = f.read()
                        if stderr_content.strip():
                            logger.error(
                                f"Job {job.job_id}: Server stderr output:\n{stderr_content[-500:]}"
                            )  # Last 500 chars
                    except Exception as e:
                        logger.warning(
                            f"Job {job.job_id}: Could not read stderr log: {e}"
                        )

                return False

            # Then try health check
            try:
                response = requests.get(health_url, timeout=5.0)
                if response.status_code == 200:
                    elapsed = time.time() - start_time
                    logger.info(
                        f"Job {job.job_id}: ✅ Server ready after {elapsed:.1f}s"
                    )
                    return True
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                # Server not ready yet, continue waiting
                pass
            except Exception as e:
                logger.warning(
                    f"Job {job.job_id}: Unexpected error during health check: {e}"
                )

            time.sleep(interval)

        logger.error(
            f"Job {job.job_id}: ❌ Server health check timed out after {timeout}s"
        )
        return False

    def _launch_server(self, job: JobInfo) -> subprocess.Popen:
        """
        Launch the HTTP training server subprocess.

        Args:
            job: JobInfo object containing job configuration

        Returns:
            Popen process object
        """
        env = os.environ.copy()
        # Set CUDA_VISIBLE_DEVICES to comma-separated list of GPU IDs
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, job.gpu_ids))

        # Build command line arguments from config
        cmd = [
            "python",
            self.server_script_path,
            f"server.port={job.port}",
            f"job_id={job.job_id}",
            f"enable_agent_loop={job.config['enable_agent_loop']}",
        ]

        if job.config.get("wandb_key"):
            cmd.append(f"wandb_key={job.config['wandb_key']}")

        # Debug: log job.config keys to verify parameters are being received
        logger.info(f"Job {job.job_id}: Config keys: {list(job.config.keys())}")

        # Forward algorithm parameters from client config
        adv_estimator = job.config.get("adv_estimator")
        if adv_estimator:
            cmd.append(f"algorithm.adv_estimator={adv_estimator}")
            logger.info(f"Job {job.job_id}: ✓ Forwarding adv_estimator={adv_estimator}")
        else:
            logger.warning(f"Job {job.job_id}: ⚠ adv_estimator not found in job.config")

        # Forward rollout.n ONLY for GRPO mode (PPO/GAE doesn't use rollout.n)
        rollout_n = job.config.get("rollout_n")
        # grpo_per_step also uses rollout.n for grouped sampling
        if adv_estimator in ("grpo", "grpo_per_step") and rollout_n:
            cmd.append(f"actor_rollout_ref.rollout.n={rollout_n}")
            logger.info(
                f"Job {job.job_id}: ✓ Forwarding rollout.n={rollout_n} ({adv_estimator} mode)"
            )
        elif adv_estimator in ("grpo", "grpo_per_step"):
            logger.warning(
                f"Job {job.job_id}: ⚠ {adv_estimator} mode but rollout_n not specified, using server default"
            )
        elif rollout_n:
            logger.info(
                f"Job {job.job_id}: Ignoring rollout_n={rollout_n} (not in GRPO mode)"
            )

        # Forward LoRA parameters if enabled (lora_rank > 0)
        lora_config = job.config.get("lora", {})
        lora_rank = lora_config.get("lora_rank", 0)
        if lora_rank and lora_rank > 0:
            cmd.append(f"actor_rollout_ref.model.lora_rank={lora_rank}")

            lora_alpha = lora_config.get("lora_alpha", 16)
            cmd.append(f"actor_rollout_ref.model.lora_alpha={lora_alpha}")

            target_modules = lora_config.get("target_modules", "all-linear")
            if target_modules:
                cmd.append(f"actor_rollout_ref.model.target_modules={target_modules}")

            exclude_modules = lora_config.get("exclude_modules")
            if exclude_modules:
                cmd.append(f"actor_rollout_ref.model.exclude_modules={exclude_modules}")

            lora_adapter_path = lora_config.get("lora_adapter_path")
            if lora_adapter_path:
                cmd.append(
                    f"actor_rollout_ref.model.lora_adapter_path={lora_adapter_path}"
                )

            # Forward LoRA-specific learning rate if specified
            lora_lr = lora_config.get("lr")
            if lora_lr:
                cmd.append(f"actor_rollout_ref.actor.optim.lr={lora_lr}")
                logger.info(f"Job {job.job_id}: ✓ LoRA lr: {lora_lr}")

            logger.info(
                f"Job {job.job_id}: ✓ LoRA enabled: rank={lora_rank}, alpha={lora_alpha}, target_modules={target_modules}"
            )

        # Forward world model loss settings if specified
        actor_config = job.config.get("actor", {})
        if actor_config.get("use_world_model_loss"):
            cmd.append("actor_rollout_ref.actor.use_world_model_loss=true")
            wm_coef = actor_config.get("world_model_loss_coef", 0.1)
            cmd.append(f"actor_rollout_ref.actor.world_model_loss_coef={wm_coef}")
            logger.info(f"Job {job.job_id}: ✓ World Model Loss enabled: coef={wm_coef}")

        # Forward WM active sampling settings if specified
        if actor_config.get("wm_active_sampling"):
            cmd.append("actor_rollout_ref.actor.wm_active_sampling=true")
            wm_active_coef = actor_config.get("wm_active_sampling_coef", 0.5)
            cmd.append(
                f"actor_rollout_ref.actor.wm_active_sampling_coef={wm_active_coef}"
            )
            logger.info(
                f"Job {job.job_id}: ✓ WM Active Sampling enabled: coef={wm_active_coef}"
            )

        logger.info(f"Job {job.job_id}: Launching server with command: {' '.join(cmd)}")

        # Create log files for stdout and stderr with human-readable timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        stdout_log = self.logs_dir / f"{timestamp}_job_{job.job_id}_stdout.log"
        stderr_log = self.logs_dir / f"{timestamp}_job_{job.job_id}_stderr.log"

        logger.info(f"Job {job.job_id}: Logging stdout to {stdout_log}")
        logger.info(f"Job {job.job_id}: Logging stderr to {stderr_log}")

        # Open log files
        stdout_file = open(stdout_log, "w")
        stderr_file = open(stderr_log, "w")

        try:
            # Launch process with output redirected to files
            # close_fds=False is default on Unix, files will be inherited by child process
            process = subprocess.Popen(
                cmd,
                cwd=self.base_dir,
                env=env,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
            )

            # CRITICAL: Close file handles in parent process immediately after spawning
            # The child process has already inherited them, so we don't need them anymore
            # This prevents file handle leaks in the parent (scheduler) process
            stdout_file.close()
            stderr_file.close()

            # Store log paths for later reference (for error reading)
            process.stdout_log_path = str(stdout_log)
            process.stderr_log_path = str(stderr_log)

            # Give server a brief moment to start (reduced from 2s to 0.5s to avoid blocking)
            # This prevents the scheduler from being unresponsive when starting multiple jobs
            time.sleep(0.5)

            # Check if process is still running
            if process.poll() is not None:
                # Read error from log file
                with open(stderr_log, "r") as f:
                    stderr_content = f.read()
                raise RuntimeError(
                    f"Server process exited immediately. Stderr: {stderr_content}"
                )

            return process

        except Exception:
            # If any error occurs, ensure file handles are closed
            try:
                if not stdout_file.closed:
                    stdout_file.close()
            except:
                pass
            try:
                if not stderr_file.closed:
                    stderr_file.close()
            except:
                pass
            raise

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific job.

        Args:
            job_id: ID of the job

        Returns:
            Dict with job status information, or None if job not found
        """
        job = self.jobs.get(job_id)
        if not job:
            return None

        # Check if process is still running
        if job.process and job.status == JobStatus.RUNNING:
            exit_code = job.process.poll()
            if exit_code is not None:
                # Process has terminated - check exit code
                if exit_code == 0:
                    logger.info(
                        f"Job {job_id}: Process terminated normally (exit code 0)"
                    )
                    job.status = JobStatus.COMPLETED
                else:
                    logger.error(
                        f"Job {job_id}: Process terminated with error (exit code {exit_code})"
                    )
                    job.status = JobStatus.FAILED
                    job.error_message = (
                        f"Process exited with non-zero code: {exit_code}"
                    )

                job.completed_at = datetime.now()
                self._release_resources(job_id)
                self._schedule_next_job()

        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "job_type": job.job_type.value,
            "gpu_ids": job.gpu_ids,
            "port": job.port,
            "server_url": job.server_url
            or job.vllm_server_url,  # Use vllm_server_url for inference jobs
            "vllm_server_url": job.vllm_server_url,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "error_message": job.error_message,
        }

    def list_jobs(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all jobs, optionally filtered by user.

        Args:
            user_id: If provided, only return jobs for this user

        Returns:
            List of job status dicts
        """
        # Update all running jobs
        for job_id in list(self.jobs.keys()):
            self.get_job_status(job_id)

        # Filter by user if specified
        if user_id:
            return [
                self.get_job_status(job_id)
                for job_id in self.jobs.keys()
                if self.jobs[job_id].user_id == user_id
            ]
        else:
            # Return all jobs (admin view)
            return [self.get_job_status(job_id) for job_id in self.jobs.keys()]

    def cancel_job(self, job_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Cancel a job.

        Args:
            job_id: ID of the job to cancel
            user_id: ID of user requesting cancellation (for authorization)

        Returns:
            Dict with cancellation status
        """
        job = self.jobs.get(job_id)
        if not job:
            return {"job_id": job_id, "status": "NOT_FOUND", "message": "Job not found"}

        # Check authorization if user_id provided
        if user_id and job.user_id and job.user_id != user_id:
            return {
                "job_id": job_id,
                "status": "FORBIDDEN",
                "message": "Not authorized to cancel this job",
            }

        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return {
                "job_id": job_id,
                "status": job.status.value,
                "message": f"Job already in terminal state: {job.status.value}",
            }

        logger.info(f"Job {job_id}: Cancelling")

        # Remove from queue if queued
        if job.status == JobStatus.QUEUED and job_id in self.job_queue:
            self.job_queue.remove(job_id)

        # Cleanup resources
        self._cleanup_job(job_id)

        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now()

        # Try to schedule next job
        self._schedule_next_job()

        return {
            "job_id": job_id,
            "status": JobStatus.CANCELLED.value,
            "message": "Job cancelled successfully",
        }

    def _cleanup_job(self, job_id: str):
        """
        Cleanup resources for a job (kill process, Ray actors, etc.).

        Args:
            job_id: ID of the job to cleanup
        """
        job = self.jobs.get(job_id)
        if not job:
            return

        logger.info(f"Job {job_id}: Starting cleanup")

        # 1. Kill the server subprocess
        if job.process:
            try:
                if job.process.poll() is None:
                    logger.info(
                        f"Job {job_id}: Terminating server process (PID {job.process.pid})"
                    )
                    job.process.terminate()
                    try:
                        job.process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        logger.warning(
                            f"Job {job_id}: Process didn't terminate, killing"
                        )
                        job.process.kill()
                        job.process.wait()

                # Note: File handles are already closed in _launch_server after process spawn
                # No need to close them here

            except Exception as e:
                logger.error(f"Job {job_id}: Error killing process: {e}")

        # 2. Kill Ray actors associated with this job
        actor_suffixes = [
            "actor_rollout",
            "critic",
            "ref",
            "reward_model",
        ]

        actors_to_kill = []
        for suffix in actor_suffixes:
            actor_name = f"job_{job_id}_{suffix}"
            try:
                actor = ray.get_actor(actor_name)
                actors_to_kill.append((actor_name, actor))
                logger.info(f"Job {job_id}: Found Ray actor {actor_name} to kill")
            except ValueError:
                # Actor doesn't exist, that's fine
                logger.debug(
                    f"Job {job_id}: Actor {actor_name} not found (already terminated or never created)"
                )
            except Exception as e:
                logger.error(f"Job {job_id}: Error looking up actor {actor_name}: {e}")

        # Kill all actors in batch with no_restart=True to prevent automatic restart
        if actors_to_kill:
            logger.info(f"Job {job_id}: Killing {len(actors_to_kill)} Ray actors...")
            for actor_name, actor in actors_to_kill:
                try:
                    # no_restart=True prevents the actor from being restarted automatically
                    ray.kill(actor, no_restart=True)
                    logger.info(f"Job {job_id}: Sent kill signal to {actor_name}")
                except Exception as e:
                    logger.error(f"Job {job_id}: Error killing actor {actor_name}: {e}")

            # Give actors a brief moment to terminate (non-blocking)
            # This helps ensure resources are freed before starting next job
            time.sleep(0.5)
            logger.info(f"Job {job_id}: Ray actors cleanup complete")
        else:
            logger.info(f"Job {job_id}: No Ray actors to kill")

        # 3. Release resources
        self._release_resources(job_id)

        logger.info(f"Job {job_id}: Cleanup complete")

    def _release_resources(self, job_id: str):
        """
        Release GPU and port resources for a job.

        Args:
            job_id: ID of the job
        """
        job = self.jobs.get(job_id)
        if not job:
            return

        logger.info(f"Job {job_id}: 🔓 Releasing resources...")
        logger.info(
            f"  Before release: {len(self.available_gpus)} available, {len(self.allocated_gpus)} allocated"
        )

        # Release all allocated GPUs
        if job.gpu_ids:
            released_gpus = []
            skipped_gpus = []
            for gpu_id in job.gpu_ids:
                if gpu_id in self.allocated_gpus:
                    self.allocated_gpus.remove(gpu_id)
                    # Only add back if it belongs to original configured GPUs
                    if gpu_id in self._original_gpu_set:
                        self.available_gpus.append(gpu_id)
                        released_gpus.append(gpu_id)
                    else:
                        skipped_gpus.append(gpu_id)
                        logger.error(
                            f"Job {job_id}: ⚠️ GPU {gpu_id} not in original config ({self._original_gpu_set}), skipping release"
                        )
            if released_gpus:
                logger.info(f"Job {job_id}: Released GPUs {released_gpus}")
            if skipped_gpus:
                logger.error(
                    f"Job {job_id}: ❌ Skipped releasing unauthorized GPUs: {skipped_gpus}"
                )

        if job.port is not None and job.port in self.allocated_ports:
            self.allocated_ports.remove(job.port)
            self.available_ports.append(job.port)
            logger.info(f"Job {job_id}: Released port {job.port}")

        logger.info(
            f"  After release: {len(self.available_gpus)} available, {len(self.allocated_gpus)} allocated"
        )
        logger.info(f"  Available GPUs now: {self.available_gpus}")

    def _schedule_next_job(self):
        """
        Try to schedule the next job from the queue.
        Uses a lock to prevent concurrent scheduling operations.
        """
        # Prevent concurrent scheduling - critical for avoiding GPU allocation race conditions
        if self._scheduling_in_progress:
            logger.debug("⚠️ Scheduling already in progress, skipping duplicate call")
            return

        self._scheduling_in_progress = True
        try:
            logger.info(
                f"🔄 Starting job scheduling (queue size: {len(self.job_queue)})"
            )
            logger.info(f"   Available GPUs: {self.available_gpus}")
            logger.info(f"   Allocated GPUs: {sorted(self.allocated_gpus)}")

            scheduled_count = 0
            while self.job_queue:
                next_job_id = self.job_queue[0]

                # Remove from queue BEFORE trying to start to prevent race conditions
                # where multiple concurrent calls could try to start the same job
                self.job_queue.pop(0)
                logger.info(f"📋 Attempting to schedule job {next_job_id} from queue")

                # Verify job is still in QUEUED state (could have been cancelled)
                job = self.jobs.get(next_job_id)
                if not job or job.status != JobStatus.QUEUED:
                    logger.warning(
                        f"Job {next_job_id}: Skipping, status is {job.status if job else 'NOT_FOUND'}"
                    )
                    continue

                # Log current resource state before allocation attempt
                logger.info(
                    f"   Before allocation - Available: {self.available_gpus}, Allocated: {sorted(self.allocated_gpus)}"
                )

                # Dispatch to correct start method based on job type
                if job.job_type == JobType.INFERENCE:
                    started = self._try_start_inference_job(next_job_id)
                else:
                    started = self._try_start_job(next_job_id)

                if started:
                    scheduled_count += 1
                    logger.info(
                        f"✅ Job {next_job_id}: Started from queue (#{scheduled_count})"
                    )
                    logger.info(
                        f"   After allocation - Available: {self.available_gpus}, Allocated: {sorted(self.allocated_gpus)}"
                    )
                else:
                    # No resources available, put the job back at the front of the queue
                    self.job_queue.insert(0, next_job_id)
                    logger.info(
                        f"⏸️ Job {next_job_id}: Insufficient resources, returned to queue"
                    )

                    # Schedule a delayed retry to check GPU availability
                    self._schedule_retry()
                    break

            logger.info(
                f"✅ Scheduling complete: {scheduled_count} job(s) started, {len(self.job_queue)} remaining in queue"
            )
        finally:
            self._scheduling_in_progress = False

    def _schedule_retry(self):
        """
        Schedule a delayed retry for job scheduling.

        Uses threading.Timer to check GPU physical availability after a delay.
        This handles the case where GPUs are marked as available in scheduler state
        but the previous job's processes haven't fully released them yet.
        """
        if self._pending_retry_count >= self._max_retries:
            logger.warning(
                f"⚠️ Max retries ({self._max_retries}) reached for GPU availability check, stopping retry"
            )
            self._pending_retry_count = 0
            return

        self._pending_retry_count += 1

        def do_retry():
            """Timer callback to check GPU availability and retry scheduling."""
            # Check if queue is empty (e.g., job was cancelled)
            if not self.job_queue:
                logger.info("🔄 Retry cancelled: queue is empty")
                self._pending_retry_count = 0
                return

            # Check the first job in queue
            job = self.jobs.get(self.job_queue[0])
            if not job:
                logger.info("🔄 Retry cancelled: job not found")
                self._pending_retry_count = 0
                return

            # Check physical GPU availability
            idle_count = sum(
                1 for gpu_id in self.available_gpus if check_gpu_available(gpu_id)
            )

            logger.info(
                f"🔄 Retry {self._pending_retry_count}/{self._max_retries}: "
                f"{idle_count} GPUs physically idle, need {job.num_gpus}"
            )

            if idle_count >= job.num_gpus:
                logger.info("🔄 GPUs now available, triggering schedule")
                self._pending_retry_count = 0  # Reset before scheduling
                self._schedule_next_job()
            else:
                # GPUs still not ready, schedule another retry
                self._schedule_retry()

        timer = threading.Timer(self._retry_interval, do_retry)
        timer.daemon = True
        timer.start()

        if self._pending_retry_count == 1:  # Only log on first retry
            logger.info(
                f"⏳ Scheduled GPU availability retry in {self._retry_interval}s "
                f"(will retry up to {self._max_retries} times, ~{self._max_retries * self._retry_interval}s total)"
            )

    def complete_job(
        self, job_id: str, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Mark a job as completed (called by client when training finishes).

        Args:
            job_id: ID of the job
            user_id: ID of user requesting completion (for authorization)

        Returns:
            Dict with completion status
        """
        job = self.jobs.get(job_id)
        if not job:
            return {"job_id": job_id, "status": "NOT_FOUND", "message": "Job not found"}

        # Check authorization if user_id provided
        if user_id and job.user_id and job.user_id != user_id:
            return {
                "job_id": job_id,
                "status": "FORBIDDEN",
                "message": "Not authorized to complete this job",
            }

        logger.info(f"Job {job_id}: Marking as completed")

        self._cleanup_job(job_id)

        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now()

        # Schedule next job
        self._schedule_next_job()

        return {
            "job_id": job_id,
            "status": JobStatus.COMPLETED.value,
            "message": "Job completed successfully",
        }

    def submit_inference_job(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        tensor_parallel_size: int = 1,
        num_gpus: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        trust_remote_code: bool = True,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit a new inference job that launches a vLLM server.

        Args:
            model_path: Path to model checkpoint
            tokenizer_path: Tokenizer path (defaults to model_path)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            num_gpus: Number of GPUs requested (overrides tensor_parallel_size if set)
            gpu_memory_utilization: GPU memory fraction to use
            max_model_len: Max model context length (optional)
            trust_remote_code: Whether to trust remote code
            user_id: ID of user submitting the job
            username: Username for display

        Returns:
            Dict with job_id, status, and vllm_server_url (if immediately started)
        """
        job_id = str(uuid.uuid4())[:8]  # Short UUID

        # Use num_gpus if specified, otherwise use tensor_parallel_size
        requested_gpus = num_gpus if num_gpus is not None else tensor_parallel_size

        # Store job configuration
        job_config = {
            "model_path": model_path,
            "tokenizer_path": tokenizer_path or model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "trust_remote_code": trust_remote_code,
        }

        job = JobInfo(
            job_id=job_id,
            status=JobStatus.QUEUED,
            config=job_config,
            num_gpus=requested_gpus,
            job_type=JobType.INFERENCE,
            user_id=user_id,
            username=username,
        )

        self.jobs[job_id] = job

        logger.info(
            f"Inference Job {job_id} submitted (requesting {requested_gpus} GPUs)"
        )

        # Use submission lock to prevent concurrent allocation race conditions
        wait_count = 0
        while self._submission_in_progress:
            if wait_count == 0:
                logger.info(
                    f"Job {job_id}: ⏳ Waiting for concurrent submission to complete..."
                )
            wait_count += 1
            time.sleep(0.1)

        if wait_count > 0:
            logger.info(
                f"Job {job_id}: Proceeding after waiting {wait_count * 0.1:.1f}s for submission lock"
            )

        self._submission_in_progress = True
        try:
            # Try to allocate resources immediately
            if self._try_start_inference_job(job_id):
                return {
                    "job_id": job_id,
                    "status": job.status.value,
                    "vllm_server_url": job.vllm_server_url,
                    "message": "Inference job started successfully",
                }
            else:
                # Add to queue
                self.job_queue.append(job_id)
                logger.info(
                    f"Inference Job {job_id} added to queue (position {len(self.job_queue)})"
                )
                return {
                    "job_id": job_id,
                    "status": JobStatus.QUEUED.value,
                    "vllm_server_url": None,
                    "message": f"Job queued (position {len(self.job_queue)})",
                }
        finally:
            self._submission_in_progress = False

    def _try_start_inference_job(self, job_id: str) -> bool:
        """
        Try to allocate resources and start an inference job (vLLM server).

        This method is now async-friendly: it launches the vLLM server process
        and starts a background thread to wait for it to be ready. The method
        returns immediately with STARTING status, allowing concurrent job submissions.

        Args:
            job_id: ID of the job to start

        Returns:
            True if job was started (in STARTING state), False if resources unavailable
        """
        job = self.jobs[job_id]

        required_gpus = job.num_gpus

        logger.info(
            f"Inference Job {job_id}: Attempting to allocate {required_gpus} GPUs"
        )
        logger.info(
            f"  Current state: {len(self.available_gpus)} available, {len(self.allocated_gpus)} allocated"
        )

        # Check if enough resources are available
        if len(self.available_gpus) < required_gpus or not self.available_ports:
            logger.info(
                f"Job {job_id}: Insufficient resources (need {required_gpus} GPUs, have {len(self.available_gpus)})"
            )
            return False

        # Allocate GPUs using topology-aware selection
        gpu_ids = self._select_gpus_by_topology(required_gpus)

        if not gpu_ids:
            logger.info(f"Job {job_id}: Could not find suitable GPU allocation")
            return False

        # Allocate a port
        if not self.available_ports:
            logger.warning(f"Job {job_id}: No available ports")
            # Release GPUs
            for gpu_id in gpu_ids:
                if gpu_id in self.allocated_gpus:
                    self.allocated_gpus.remove(gpu_id)
                    if gpu_id in self._original_gpu_set:
                        self.available_gpus.append(gpu_id)
            return False

        port = self.available_ports.pop(0)
        self.allocated_ports.add(port)

        job.gpu_ids = gpu_ids
        job.port = port
        # Use actual hostname instead of 0.0.0.0 for client connections
        # 0.0.0.0 is valid for binding but not for connecting
        import socket

        hostname = socket.gethostname()
        # Try to get a routable IP address
        try:
            # Get the IP address that would be used to connect to external networks
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip_address = s.getsockname()[0]
            s.close()
        except Exception:
            # Fall back to hostname resolution
            try:
                ip_address = socket.gethostbyname(hostname)
            except Exception:
                ip_address = "127.0.0.1"  # Last resort fallback
        job.vllm_server_url = f"http://{ip_address}:{port}"
        job.status = JobStatus.STARTING
        job.started_at = datetime.now()

        logger.info(f"Inference Job {job_id}: Allocated GPUs {gpu_ids}, port {port}")

        try:
            process = self._launch_vllm_server(job)
            job.process = process
            logger.info(
                f"Inference Job {job_id}: vLLM server process started (PID {process.pid})"
            )

            # Start background thread to wait for vLLM server to be ready
            # This allows the submission to return immediately
            startup_thread = threading.Thread(
                target=self._background_wait_for_vllm_ready,
                args=(job_id,),
                daemon=True,
                name=f"vllm-startup-{job_id}",
            )
            self._startup_threads[job_id] = startup_thread
            startup_thread.start()

            logger.info(
                f"Inference Job {job_id}: Background startup thread started, returning immediately with STARTING status"
            )
            return True  # Job is now in STARTING state, background thread will update to RUNNING

        except OSError as e:
            if "Address already in use" in str(e) or "address in use" in str(e).lower():
                logger.warning(f"Job {job_id}: Port {port} already in use")
            else:
                logger.error(f"Job {job_id}: Failed to start vLLM server: {e}")
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            # Use _cleanup_job for consistent cleanup (terminates process if any, releases resources)
            self._cleanup_job(job_id)
            return False

        except Exception as e:
            logger.error(f"Job {job_id}: Failed to start vLLM server: {e}")
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            # Use _cleanup_job for consistent cleanup (terminates process if any, releases resources)
            self._cleanup_job(job_id)
            return False

    def _background_wait_for_vllm_ready(self, job_id: str) -> None:
        """
        Background thread function that waits for vLLM server to become ready.

        This runs in a separate thread so that job submission can return immediately.
        Updates the job status to RUNNING when ready, or FAILED if startup fails.

        Args:
            job_id: ID of the job to monitor
        """
        job = self.jobs.get(job_id)
        if not job:
            logger.error(f"Background startup thread: Job {job_id} not found")
            return

        logger.info(
            f"Inference Job {job_id}: Background thread waiting for vLLM server to be ready..."
        )

        try:
            # Wait for vLLM server to be ready (300s timeout for large models)
            if self._wait_for_vllm_server_ready(job, timeout=300.0, interval=3.0):
                job.status = JobStatus.RUNNING
                logger.info(
                    f"Inference Job {job_id}: ✅ vLLM server started and ready (background thread)"
                )
            else:
                logger.error(
                    f"Inference Job {job_id}: Failed to start - vLLM server not ready (background thread)"
                )
                job.status = JobStatus.FAILED
                job.error_message = "vLLM server failed to become ready within timeout"
                job.completed_at = datetime.now()

                # Use _cleanup_job to properly terminate process and release resources
                self._cleanup_job(job_id)
        except Exception as e:
            logger.error(
                f"Inference Job {job_id}: Background startup thread failed with exception: {e}"
            )
            job.status = JobStatus.FAILED
            job.error_message = f"Startup thread exception: {e}"
            job.completed_at = datetime.now()

            # Use _cleanup_job to properly terminate process and release resources
            self._cleanup_job(job_id)
        finally:
            # Clean up thread reference
            if job_id in self._startup_threads:
                del self._startup_threads[job_id]

    def _launch_vllm_server(self, job: JobInfo) -> subprocess.Popen:
        """
        Launch a vLLM server subprocess for inference.

        Args:
            job: JobInfo object containing inference job configuration

        Returns:
            Popen process object
        """
        env = os.environ.copy()
        # Set CUDA_VISIBLE_DEVICES to comma-separated list of GPU IDs
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, job.gpu_ids))

        config = job.config

        # Build vllm serve command
        # Note: Always bind to 0.0.0.0 for cross-node access
        # (job.vllm_server_url uses the actual IP for client connections)
        cmd = [
            "vllm",
            "serve",
            config["model_path"],
            "--host",
            "0.0.0.0",  # Bind to all interfaces for cross-node access
            "--port",
            str(job.port),
            "--tensor-parallel-size",
            str(config.get("tensor_parallel_size", 1)),
            "--gpu-memory-utilization",
            str(config.get("gpu_memory_utilization", 0.9)),
        ]

        # Add optional arguments
        if (
            config.get("tokenizer_path")
            and config["tokenizer_path"] != config["model_path"]
        ):
            cmd.extend(["--tokenizer", config["tokenizer_path"]])

        if config.get("max_model_len"):
            cmd.extend(["--max-model-len", str(config["max_model_len"])])

        if config.get("trust_remote_code", True):
            cmd.append("--trust-remote-code")

        logger.info(
            f"Inference Job {job.job_id}: Launching vLLM server with command: {' '.join(cmd)}"
        )

        # Create log files with human-readable timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        stdout_log = (
            self.logs_dir / f"{timestamp}_inference_job_{job.job_id}_stdout.log"
        )
        stderr_log = (
            self.logs_dir / f"{timestamp}_inference_job_{job.job_id}_stderr.log"
        )

        logger.info(f"Inference Job {job.job_id}: Logging stdout to {stdout_log}")
        logger.info(f"Inference Job {job.job_id}: Logging stderr to {stderr_log}")

        stdout_file = open(stdout_log, "w")
        stderr_file = open(stderr_log, "w")

        try:
            process = subprocess.Popen(
                cmd,
                cwd=self.base_dir,
                env=env,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
            )

            # Close file handles in parent process
            stdout_file.close()
            stderr_file.close()

            # Store log paths for later reference
            process.stdout_log_path = str(stdout_log)
            process.stderr_log_path = str(stderr_log)

            # Give server a brief moment to start
            time.sleep(0.5)

            # Check if process is still running
            if process.poll() is not None:
                with open(stderr_log, "r") as f:
                    stderr_content = f.read()
                raise RuntimeError(
                    f"vLLM server process exited immediately. Stderr: {stderr_content}"
                )

            return process

        except Exception:
            try:
                if not stdout_file.closed:
                    stdout_file.close()
            except:
                pass
            try:
                if not stderr_file.closed:
                    stderr_file.close()
            except:
                pass
            raise

    def _wait_for_vllm_server_ready(
        self, job: JobInfo, timeout: float = 300.0, interval: float = 3.0
    ) -> bool:
        """
        Wait for vLLM server to be ready by polling health endpoint.

        Args:
            job: JobInfo object
            timeout: Maximum time to wait in seconds (default 300s for large models)
            interval: Polling interval in seconds

        Returns:
            True if server is ready, False if timeout or process crashed
        """
        import requests

        start_time = time.time()
        # Use 127.0.0.1 for health check since it's local to the scheduler
        # (job.vllm_server_url uses the external IP for client connections)
        health_url = f"http://127.0.0.1:{job.port}/health"
        last_log_time = start_time
        log_interval = 30.0  # Log progress every 30 seconds

        logger.info(
            f"Inference Job {job.job_id}: Waiting for vLLM server to be ready at {health_url} (timeout: {timeout}s)"
        )

        while time.time() - start_time < timeout:
            elapsed = time.time() - start_time

            # Log progress periodically
            if time.time() - last_log_time >= log_interval:
                logger.info(
                    f"Inference Job {job.job_id}: Still waiting for vLLM server... ({elapsed:.0f}s elapsed, {timeout - elapsed:.0f}s remaining)"
                )
                last_log_time = time.time()

            # Check if the process is still running
            if job.process and job.process.poll() is not None:
                exit_code = job.process.poll()
                logger.error(
                    f"Inference Job {job.job_id}: ❌ vLLM process crashed during startup (exit code {exit_code})"
                )

                if hasattr(job.process, "stderr_log_path"):
                    try:
                        with open(job.process.stderr_log_path, "r") as f:
                            stderr_content = f.read()
                        if stderr_content.strip():
                            logger.error(
                                f"Inference Job {job.job_id}: vLLM stderr output:\n{stderr_content[-2000:]}"
                            )
                    except Exception as e:
                        logger.warning(
                            f"Inference Job {job.job_id}: Could not read stderr log: {e}"
                        )

                return False

            # Try health check
            try:
                response = requests.get(health_url, timeout=5.0)
                if response.status_code == 200:
                    logger.info(
                        f"Inference Job {job.job_id}: ✅ vLLM server ready after {elapsed:.1f}s"
                    )
                    return True
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                pass
            except Exception as e:
                logger.warning(
                    f"Inference Job {job.job_id}: Unexpected error during health check: {e}"
                )

            time.sleep(interval)

        # Timeout - log stderr for diagnosis
        logger.error(
            f"Inference Job {job.job_id}: ❌ vLLM server health check timed out after {timeout}s"
        )
        if job.process and hasattr(job.process, "stderr_log_path"):
            try:
                with open(job.process.stderr_log_path, "r") as f:
                    stderr_content = f.read()
                if stderr_content.strip():
                    logger.error(
                        f"Inference Job {job.job_id}: vLLM stderr output at timeout:\n{stderr_content[-2000:]}"
                    )
            except Exception as e:
                logger.warning(
                    f"Inference Job {job.job_id}: Could not read stderr log: {e}"
                )

        return False


# ==================== FastAPI Application ====================


def create_app(
    scheduler_actor, user_manager: UserManager, enable_auth: bool = True
) -> FastAPI:
    """
    Create FastAPI application with scheduler endpoints.

    Args:
        scheduler_actor: Ray actor handle for the job scheduler
        user_manager: UserManager instance for authentication
        enable_auth: Whether to enable authentication (default: True)

    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="OpenTinker Job Scheduler",
        description="Scheduler for managing distributed training jobs with user authentication",
        version="2.0.0",
    )

    # Add CORS middleware to allow Web Dashboard requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins in development; restrict in production
        allow_credentials=True,
        allow_methods=["*"],  # Allow all methods (GET, POST, DELETE, OPTIONS, etc.)
        allow_headers=["*"],  # Allow all headers including Authorization
    )

    # Authentication dependency
    async def get_current_user(
        authorization: Optional[str] = Header(None),
    ) -> Optional[User]:
        """Extract and validate user from Authorization header"""
        if not enable_auth:
            return None  # No auth required

        if not authorization:
            raise HTTPException(status_code=401, detail="Authorization header required")

        # Expect format: "Bearer <api_key>"
        parts = authorization.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise HTTPException(
                status_code=401, detail="Invalid authorization header format"
            )

        api_key = parts[1]
        user = user_manager.authenticate(api_key)

        if not user:
            raise HTTPException(status_code=401, detail="Invalid API key")

        return user

    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "service": "OpenTinker Job Scheduler",
            "version": "2.0.0",
            "status": "running",
            "authentication": "enabled" if enable_auth else "disabled",
            "endpoints": [
                "/register (POST)",
                "/submit_job (POST)",
                "/submit_inference_job (POST)",
                "/job_status/{job_id} (GET)",
                "/list_jobs (GET)",
                "/cancel_job/{job_id} (DELETE)",
                "/complete_job/{job_id} (POST)",
            ],
        }

    @app.post("/register")
    async def register(username: str):
        """
        Register a new user (self-service registration).

        Returns API key for the new user.
        """
        try:
            user = user_manager.register_user(username, is_admin=False)
            return {
                "user_id": user.user_id,
                "username": user.username,
                "api_key": user.api_key,
                "message": "User registered successfully. Save your API key - it cannot be retrieved later!",
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error registering user: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/submit_job", response_model=SubmitJobResponse)
    async def submit_job(
        request: SubmitJobRequest, user: Optional[User] = Depends(get_current_user)
    ):
        """Submit a new training job (requires authentication)"""
        try:
            result = ray.get(
                scheduler_actor.submit_job.remote(
                    config=request.config,
                    user_id=user.user_id if user else None,
                    username=user.username if user else None,
                    enable_agent_loop=request.enable_agent_loop,
                    wandb_key=request.wandb_key,
                    num_gpus=request.num_gpus,
                )
            )
            return SubmitJobResponse(**result)
        except Exception as e:
            logger.error(f"Error submitting job: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/submit_inference_job", response_model=SubmitInferenceJobResponse)
    async def submit_inference_job(
        request: SubmitInferenceJobRequest,
        user: Optional[User] = Depends(get_current_user),
    ):
        """Submit a new inference job (launches vLLM server)"""
        try:
            result = ray.get(
                scheduler_actor.submit_inference_job.remote(
                    model_path=request.model_path,
                    tokenizer_path=request.tokenizer_path,
                    tensor_parallel_size=request.tensor_parallel_size,
                    num_gpus=request.num_gpus,
                    gpu_memory_utilization=request.gpu_memory_utilization,
                    max_model_len=request.max_model_len,
                    trust_remote_code=request.trust_remote_code,
                    user_id=user.user_id if user else None,
                    username=user.username if user else None,
                )
            )
            return SubmitInferenceJobResponse(**result)
        except Exception as e:
            logger.error(f"Error submitting inference job: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/job_status/{job_id}", response_model=JobStatusResponse)
    async def get_job_status(
        job_id: str, user: Optional[User] = Depends(get_current_user)
    ):
        """Get status of a specific job"""
        try:
            result = ray.get(scheduler_actor.get_job_status.remote(job_id))
            if result is None:
                raise HTTPException(status_code=404, detail="Job not found")

            # Check authorization if auth is enabled
            if user and result.get("user_id") and result["user_id"] != user.user_id:
                raise HTTPException(
                    status_code=403, detail="Not authorized to view this job"
                )

            return JobStatusResponse(**result)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting job status: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/list_jobs", response_model=ListJobsResponse)
    async def list_jobs(user: Optional[User] = Depends(get_current_user)):
        """List all jobs (filtered by user if authentication enabled)"""
        try:
            user_id = user.user_id if user else None
            jobs = ray.get(scheduler_actor.list_jobs.remote(user_id=user_id))
            return ListJobsResponse(jobs=[JobStatusResponse(**job) for job in jobs])
        except Exception as e:
            logger.error(f"Error listing jobs: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/cancel_job/{job_id}", response_model=CancelJobResponse)
    async def cancel_job(job_id: str, user: Optional[User] = Depends(get_current_user)):
        """Cancel a job (requires ownership)"""
        try:
            user_id = user.user_id if user else None
            result = ray.get(scheduler_actor.cancel_job.remote(job_id, user_id=user_id))

            if result.get("status") == "FORBIDDEN":
                raise HTTPException(status_code=403, detail=result.get("message"))

            return CancelJobResponse(**result)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error cancelling job: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/complete_job/{job_id}", response_model=CancelJobResponse)
    async def complete_job(
        job_id: str, user: Optional[User] = Depends(get_current_user)
    ):
        """Mark a job as completed (requires ownership)"""
        try:
            user_id = user.user_id if user else None
            result = ray.get(
                scheduler_actor.complete_job.remote(job_id, user_id=user_id)
            )

            if result.get("status") == "FORBIDDEN":
                raise HTTPException(status_code=403, detail=result.get("message"))

            return CancelJobResponse(**result)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error completing job: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Admin endpoints (if needed in the future)
    @app.get("/admin/users")
    async def list_users(user: Optional[User] = Depends(get_current_user)):
        """List all users (admin only)"""
        if enable_auth and (not user or not user.is_admin):
            raise HTTPException(status_code=403, detail="Admin access required")

        users = user_manager.list_users()
        return {
            "users": [
                {
                    "user_id": u.user_id,
                    "username": u.username,
                    "created_at": u.created_at,
                    "is_admin": u.is_admin,
                }
                for u in users
            ]
        }

    return app
