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
# See the License for the specific language governing permissions and
# limitations under the License.
"""
HTTP Training Server for PPO Training

This module provides a FastAPI-based HTTP server that wraps PPO training logic,
allowing clients to send training batches via HTTP requests instead of Ray RPC.

Key Features:
- REST API endpoints for training operations
- DataProto serialization/deserialization for HTTP transmission
- Custom dataloader support on client side
- Generation config customization from client
"""

import asyncio
import base64
import logging
import signal
import sys
import traceback
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from verl import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator
# from verl.trainer.ppo.ray_trainer import (
#     RayPPOTrainer,
#     ResourcePoolManager,
#     apply_kl_penalty,
#     compute_advantage,
#     compute_response_mask,
# )

from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_timing_metrics,
    compute_throughout_metrics,
)
from verl.trainer.ppo.mismatch_helper import compute_rollout_importance_weights
import os
from transformers import AutoTokenizer
import ray
from omegaconf import OmegaConf, open_dict
import json


from verl.trainer.ppo.utils import Role

from opentinker.backend_patch.verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    ResourcePoolManager,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)

from opentinker.backend_patch.verl.trainer.ppo.reward import (
    compute_reward,
    compute_reward_async,
    load_reward_manager,
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global server instance
_training_server: Optional["PPOTrainingServerBackend"] = None
_server_cfg = None
_generation_config = None
_config_ready_event = threading.Event()


# ==================== Pydantic Models for API ====================


class InitWorkersRequest(BaseModel):
    """Request model for initializing workers"""

    total_steps: Optional[int] = None  # Empty for now, could add config overrides later


class GenerationConfigRequest(BaseModel):
    """Request model for setting generation configuration"""

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_new_tokens: Optional[int] = None
    repetition_penalty: Optional[float] = None
    # Add more generation parameters as needed


class TrainStepRequest(BaseModel):
    """Request model for training step"""

    batch_data: Dict[str, Any]  # Serialized DataProto


class TrainStepResponse(BaseModel):
    """Response model for training step"""

    status: str
    metrics: Dict[str, float]
    global_steps: int
    error: Optional[str] = None
    traceback: Optional[str] = None


class StatusResponse(BaseModel):
    """Response model for server status"""

    status: str
    is_initialized: bool
    global_steps: int
    generation_config: Dict[str, Any]


class ValidateRequest(BaseModel):
    """Request model for validation"""

    batch_data: Dict[str, Any]  # Serialized DataProto


class ValidateResponse(BaseModel):
    """Response model for validation"""

    status: str
    metrics: Dict[str, float]
    samples: list = []
    error: Optional[str] = None
    traceback: Optional[str] = None


class SaveCheckpointRequest(BaseModel):
    """Request model for saving checkpoint"""

    pass  # Empty for now


class ConfigOverrideRequest(BaseModel):
    """Request model for overriding server configuration"""

    config_overrides: Dict[str, Any]  # Configuration overrides in dictionary format


class UploadRewardFunctionRequest(BaseModel):
    """Request model for uploading custom reward function code"""

    function_name: str
    source_code: str


# ==================== DataProto Serialization ====================


def serialize_dataproto(data: DataProto) -> Dict[str, Any]:
    """
    Serialize DataProto to JSON-compatible dict for HTTP transmission.

    Args:
        data: DataProto to serialize

    Returns:
        JSON-compatible dict
    """

    def serialize_tensor(t):
        """Serialize a single tensor to base64-encoded dict"""
        if isinstance(t, torch.Tensor):
            return {
                "__type__": "torch.Tensor",
                "__dtype__": str(t.dtype),
                "__shape__": list(t.shape),
                "__device__": str(t.device),
                "__data__": base64.b64encode(t.cpu().numpy().tobytes()).decode("utf-8"),
            }
        elif isinstance(t, np.ndarray):
            return {
                "__type__": "numpy.ndarray",
                "__dtype__": str(t.dtype),
                "__shape__": list(t.shape),
                "__data__": base64.b64encode(t.tobytes()).decode("utf-8"),
            }
        # Handle PIL Images for VL models
        elif hasattr(t, "save") and hasattr(t, "mode"):
            # This is a PIL Image
            import io

            buffer = io.BytesIO()
            # Convert to RGB if necessary (some formats like RGBA need conversion)
            if hasattr(t, "mode") and t.mode in ("RGBA", "P", "LA"):
                t = t.convert("RGB")
            t.save(buffer, format="PNG")
            return {
                "__type__": "PIL.Image",
                "__mode__": t.mode,
                "__size__": list(t.size),
                "__data__": base64.b64encode(buffer.getvalue()).decode("utf-8"),
            }
        return t

    def deep_serialize(obj):
        """Recursively serialize nested structures (dicts, lists) containing tensors/images."""
        if isinstance(obj, dict):
            return {k: deep_serialize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [deep_serialize(item) for item in obj]
        elif isinstance(obj, tuple):
            return [deep_serialize(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return serialize_tensor(obj)
        elif isinstance(obj, np.ndarray) and obj.dtype != object:
            return serialize_tensor(obj)
        elif hasattr(obj, "save") and hasattr(obj, "mode"):
            # PIL Image
            return serialize_tensor(obj)
        else:
            return obj

    # Serialize batch (TensorDict)
    serialized_batch = {}
    if data.batch is not None:
        for k, v in data.batch.items():
            serialized_batch[k] = serialize_tensor(v)

    # Serialize non_tensor_batch
    serialized_non_tensor = {}
    for k, v in data.non_tensor_batch.items():
        if isinstance(v, np.ndarray):
            # For object dtype arrays, convert to list (preserving structure)
            if v.dtype == object:
                # Recursively serialize any nested objects (dicts, lists, tensors, PIL Images)
                data_list = [deep_serialize(item) for item in v.flatten()]
                serialized_non_tensor[k] = {
                    "__type__": "numpy.ndarray",
                    "__dtype__": "object",
                    "__shape__": list(v.shape),
                    "__data__": data_list,
                }
            else:
                serialized_non_tensor[k] = serialize_tensor(v)
        else:
            serialized_non_tensor[k] = deep_serialize(v)

    return {
        "batch": serialized_batch,
        "non_tensor_batch": serialized_non_tensor,
        "meta_info": data.meta_info,
    }


def deserialize_dataproto(data_dict: Dict[str, Any]) -> DataProto:
    """
    Deserialize DataProto from JSON dict.

    Args:
        data_dict: Serialized DataProto dict

    Returns:
        DataProto instance
    """

    def deserialize_tensor(obj):
        """Deserialize a single tensor from base64-encoded dict"""
        if not isinstance(obj, dict) or "__type__" not in obj:
            return obj

        if obj["__type__"] == "torch.Tensor":
            dtype_str = obj["__dtype__"].replace("torch.", "")
            dtype = getattr(torch, dtype_str)
            shape = tuple(obj["__shape__"])
            data_bytes = base64.b64decode(obj["__data__"])
            array = np.frombuffer(
                data_bytes, dtype=np.dtype(str(dtype).replace("torch.", ""))
            ).copy()
            tensor = torch.from_numpy(array).reshape(shape)
            return tensor.to(dtype)

        elif obj["__type__"] == "numpy.ndarray":
            if obj["__dtype__"] == "object":
                # Reconstruct object array from list
                # Use np.empty + fill to avoid np.array creating multi-dimensional arrays
                # when elements are lists of equal length
                data_list = obj["__data__"]
                shape = tuple(obj["__shape__"])
                # Create empty object array and fill it element by element
                array = np.empty(len(data_list), dtype=object)
                for i, item in enumerate(data_list):
                    array[i] = item
                array = array.reshape(shape)
                return array
            else:
                dtype = np.dtype(obj["__dtype__"])
                shape = tuple(obj["__shape__"])
                data_bytes = base64.b64decode(obj["__data__"])
                array = np.frombuffer(data_bytes, dtype=dtype).reshape(shape)
                return array

        # Handle PIL Images for VL models
        elif obj["__type__"] == "PIL.Image":
            from PIL import Image
            import io

            data_bytes = base64.b64decode(obj["__data__"])
            buffer = io.BytesIO(data_bytes)
            return Image.open(buffer).copy()  # .copy() to detach from buffer

        return obj

    def deep_deserialize(obj):
        """Recursively deserialize nested structures (dicts, lists) containing serialized tensors/images."""
        if isinstance(obj, dict):
            if "__type__" in obj:
                # This is a serialized object (Tensor, ndarray, or PIL.Image)
                return deserialize_tensor(obj)
            else:
                # Regular dict, recursively deserialize values
                return {k: deep_deserialize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [deep_deserialize(item) for item in obj]
        else:
            return obj

    # Deserialize batch
    tensors = {}
    if "batch" in data_dict and data_dict["batch"]:
        for k, v in data_dict["batch"].items():
            tensors[k] = deserialize_tensor(v)

    # Deserialize non_tensor_batch
    non_tensors = {}
    if "non_tensor_batch" in data_dict:
        for k, v in data_dict["non_tensor_batch"].items():
            non_tensors[k] = deep_deserialize(v)

    # Get meta_info
    meta_info = data_dict.get("meta_info", {})

    return DataProto.from_dict(
        tensors=tensors, non_tensors=non_tensors, meta_info=meta_info
    )


# ==================== helper function for worker initialization ====================


def build_resource_pool_manager(config):
    """Build resource pool manager from config"""
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager

    # resource_pool_spec = {}
    # for pool_name, process_on_nodes in config.trainer.resource_pool_spec.items():
    #     resource_pool_spec[pool_name] = process_on_nodes

    # NOTE: initialze two resource pool
    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    # TODO Here you can use the new registration method to support dynamic registration of roles
    if config.reward_model.enable_resource_pool:
        if config.reward_model.n_gpus_per_node <= 0:
            raise ValueError(
                "config.reward_model.n_gpus_per_node must be greater than 0"
            )
        if config.reward_model.nnodes <= 0:
            raise ValueError("config.reward_model.nnodes must be greater than 0")

        reward_pool = [config.reward_model.n_gpus_per_node] * config.reward_model.nnodes
        resource_pool_spec["reward_pool"] = reward_pool

    print(f"resource_pool_spec: {resource_pool_spec}")
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
        # Role.RewardModel: global_pool_id,
    }

    # mapping = {}
    # if config.actor_rollout_ref.hybrid_engine:
    #     mapping[Role.ActorRollout] = "ppo_pool"
    # if config.critic.enable:
    #     mapping[Role.Critic] = "ppo_pool"
    # if hasattr(config, "ref_policy") and config.ref_policy.enable:
    #     mapping[Role.RefPolicy] = "ppo_pool"
    # if config.reward_model.enable:
    #     mapping[Role.RewardModel] = "ppo_pool"

    return ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)


def build_role_worker_mapping(config):
    """Build role to worker class mapping"""
    from verl.trainer.ppo.ray_trainer import Role
    from verl.trainer.ppo.utils import need_critic
    import ray

    role_worker_mapping = {}

    # Actor rollout - based on strategy
    if config.actor_rollout_ref.hybrid_engine:
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import (
                ActorRolloutRefWorker,
                AsyncActorRolloutRefWorker,
            )

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import (
                ActorRolloutRefWorker,
                AsyncActorRolloutRefWorker,
            )

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
        else:
            raise NotImplementedError(
                f"Unsupported actor strategy: {config.actor_rollout_ref.actor.strategy}"
            )

        role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)

    # Critic - use need_critic() to determine if we need it
    if need_critic(config):
        if config.critic.strategy in {"fsdp", "fsdp2"}:
            use_legacy_worker_impl = config.trainer.get(
                "use_legacy_worker_impl", "auto"
            )
            if use_legacy_worker_impl in ["auto", "enable"]:
                from verl.workers.fsdp_workers import CriticWorker
            elif use_legacy_worker_impl == "disable":
                from verl.workers.roles import CriticWorker
            else:
                raise ValueError(
                    f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}"
                )
        elif config.critic.strategy == "megatron":
            from verl.workers.megatron_workers import CriticWorker
        else:
            raise NotImplementedError(
                f"Unsupported critic strategy: {config.critic.strategy}"
            )

        role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)
        print("✓ Critic worker added to role_worker_mapping")
    else:
        print(f"✗ Critic not needed (adv_estimator={config.algorithm.adv_estimator})")

    # Reference policy (if KL loss or KL reward is used)
    if hasattr(config, "algorithm") and (
        config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss
    ):
        # Use the same actor rollout class for reference policy
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import ActorRolloutRefWorker
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import ActorRolloutRefWorker
        else:
            raise NotImplementedError(
                f"Unsupported ref policy strategy: {config.actor_rollout_ref.actor.strategy}"
            )

        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
        print("✓ RefPolicy worker added to role_worker_mapping")

    # Reward model
    if config.reward_model.enable:
        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        if use_legacy_worker_impl in ["auto", "enable"]:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError(
                    f"Unsupported reward model strategy: {config.reward_model.strategy}"
                )
        elif use_legacy_worker_impl == "disable":
            from verl.workers.roles import RewardModelWorker
        else:
            raise ValueError(
                f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}"
            )

        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        print("✓ RewardModel worker added to role_worker_mapping")

    print(f"Final role_worker_mapping keys: {list(role_worker_mapping.keys())}")
    return role_worker_mapping


def build_reward_fn(config, tokenizer):
    """Build reward function.

    For Gomoku environment, wraps the standard reward function with GomokuRewardManager
    to track game outcome metrics (win rate, loss rate, etc.) which are logged to wandb.
    """

    reward_fn = load_reward_manager(
        config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
    )
    val_reward_fn = load_reward_manager(
        config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
    )

    # Check if this is a Gomoku environment by looking at interaction config
    is_gomoku = False
    try:
        interaction_config_path = config.actor_rollout_ref.rollout.multi_turn.get(
            "interaction_config_path"
        )
        if interaction_config_path:
            import yaml
            import os

            if os.path.exists(interaction_config_path):
                with open(interaction_config_path, "r") as f:
                    interaction_config = yaml.safe_load(f)
                    interactions = interaction_config.get("interaction", [])
                    for interaction in interactions:
                        if interaction.get("name") == "gomoku":
                            is_gomoku = True
                            break
    except Exception as e:
        logger.debug(f"Failed to detect Gomoku environment: {e}")

    # Wrap with GomokuRewardManager for metrics tracking
    if is_gomoku:
        try:
            from opentinker.environment.gomoku.gomoku_metrics import GomokuRewardManager

            print(
                "[build_reward_fn] Detected Gomoku environment, wrapping with GomokuRewardManager for metrics"
            )
            reward_fn = GomokuRewardManager(
                tokenizer=tokenizer, base_reward_fn=reward_fn, num_examine=0
            )
            val_reward_fn = GomokuRewardManager(
                tokenizer=tokenizer, base_reward_fn=val_reward_fn, num_examine=1
            )
        except ImportError as e:
            logger.warning(
                f"Failed to import GomokuRewardManager: {e}. Using standard reward function."
            )

    return reward_fn, val_reward_fn


# ==================== PPO Training Server Backend ====================


class PPOTrainingServerBackend:
    """
    Backend for PPO training server.

    This class wraps the core PPO training logic and provides methods
    that can be called via HTTP API endpoints.
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping,
        resource_pool_manager: ResourcePoolManager,
        reward_fn=None,
        val_reward_fn=None,
        ray_worker_group_cls=None,
    ):
        """
        Initialize PPO training server backend.

        Args:
            config: Training configuration
            tokenizer: Tokenizer for text processing
            role_worker_mapping: Mapping from roles to worker classes
            resource_pool_manager: Manager for Ray resource pools
            reward_fn: Reward function for training
            val_reward_fn: Reward function for validation
            ray_worker_group_cls: Ray worker group class
        """
        self.config = config
        self.tokenizer = tokenizer
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        # Initialize trainer (without dataloader - client provides custom dataloader)
        from verl.single_controller.ray import RayWorkerGroup

        self.trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls or RayWorkerGroup,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=None,  # No dataset on server side
            val_dataset=None,
            skip_dataloader_init=True,  # Skip dataloader creation - client provides batches
        )

        # Server state
        self.is_initialized = False
        self.global_steps = 0

        # Generation config (can be overridden by client)
        self.generation_config = {
            "do_sample": True,  # CRITICAL: Enable sampling by default for PPO training
            "temperature": config.actor_rollout_ref.rollout.get("temperature", 1.0),
            "top_p": config.actor_rollout_ref.rollout.get("top_p", 1.0),
            "max_new_tokens": config.actor_rollout_ref.rollout.get(
                "max_new_tokens", 4096
            ),
        }

        # Extract necessary components from trainer
        self.use_critic = self.trainer.use_critic
        self.use_reference_policy = self.trainer.use_reference_policy
        self.use_rm = self.trainer.use_rm
        self.ref_in_actor = self.trainer.ref_in_actor
        self.async_rollout_mode = False  # Will be set after init_workers

        # KL control
        if self.config.algorithm.use_kl_in_reward:
            from verl.trainer.ppo import core_algos

            self.kl_ctrl_in_reward = core_algos.get_kl_controller(
                self.config.algorithm.kl_ctrl
            )
        else:
            self.kl_ctrl_in_reward = None

    def init_workers(self, total_steps: int) -> Dict[str, Any]:
        """
        Initialize Ray workers for training.

        Returns:
            Status dict
        """
        try:
            # optimizer needs parameter: total_steps
            self.trainer.post_init(total_steps)
            logger.info("Initializing workers...")

            # Check async rollout mode
            self.async_rollout_mode = (
                self.config.actor_rollout_ref.rollout.mode == "async"
            )

            if self.async_rollout_mode:
                # Run init_workers in a separate thread to avoid blocking the FastAPI event loop
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self.trainer.init_workers)
                    future.result(timeout=600)  # 10 minute timeout
            else:
                self.trainer.init_workers()

            # Get worker groups
            self.actor_rollout_wg = self.trainer.actor_rollout_wg
            self.critic_wg = self.trainer.critic_wg if self.use_critic else None
            self.ref_policy_wg = (
                self.trainer.ref_policy_wg
                if self.use_reference_policy and not self.ref_in_actor
                else None
            )
            self.rm_wg = self.trainer.rm_wg if self.use_rm else None

            if self.async_rollout_mode:
                self.async_rollout_manager = self.trainer.async_rollout_manager

            self.is_initialized = True
            logger.info("Workers initialized successfully")
            return {"status": "success", "message": "Workers initialized"}

        except Exception as e:
            logger.error(f"Failed to initialize workers: {e}")
            raise

    def set_generation_config(self, config: Dict[str, Any]):
        """
        Set generation configuration.

        Args:
            config: Generation config dict

        Returns:
            Updated config
        """
        self.generation_config.update(config)
        logger.info(f"Generation config updated: {self.generation_config}")
        return self.generation_config

    def compute_rollout_importance_weights_and_add_to_batch(
        self, batch: DataProto
    ) -> tuple[DataProto, dict]:
        """Compute IS weights and apply rejection sampling for rollout-training mismatch.

        Computes importance sampling weights to correct for distribution mismatch between
        rollout and training policies. Applies rejection sampling (mask mode/veto) by
        modifying response_mask. Always updates response_mask; conditionally adds IS weights.

        Key behavior:
        - response_mask: ALWAYS updated with rejection (mask mode + veto excluded from training)
        - rollout_is_weights: Added to batch ONLY if config.algorithm.rollout_is=True

        This separation ensures:
        - Rejection works even when IS weights are disabled (rollout_is=False)
        - Metrics can be monitored before enabling IS weight application

        Args:
            batch: DataProto with old_log_probs, rollout_log_probs, response_mask

        Returns:
            Tuple of (updated_batch, metrics):
                updated_batch: Batch with modified response_mask (always) and rollout_is_weights (if rollout_is=True)
                metrics: Dict of IS and mismatch metrics, all with "mismatch/" prefix
        """
        # Compute rollout IS weights if enabled and data is available
        # rollout_is_threshold is the main on/off switch (None = disabled, float = enabled)
        rollout_is_threshold = self.config.algorithm.get("rollout_is_threshold", None)
        if (
            rollout_is_threshold is not None
            and rollout_is_threshold > 0
            and "rollout_log_probs" in batch.batch
        ):
            logger.info(
                f"DEBUG: Computing rollout IS weights. Batch keys: {batch.batch.keys()}"
            )
            # Compute IS weights and get modified response_mask
            rollout_is_weights, modified_response_mask, rollout_is_metrics = (
                compute_rollout_importance_weights(
                    old_log_prob=batch.batch["old_log_probs"],
                    rollout_log_prob=batch.batch["rollout_log_probs"],
                    response_mask=batch.batch["response_mask"],
                    rollout_is_level=self.config.algorithm.rollout_is_level,
                    rollout_is_mode=self.config.algorithm.rollout_is_mode,
                    rollout_is_threshold=self.config.algorithm.rollout_is_threshold,
                    rollout_is_threshold_lower=self.config.algorithm.get(
                        "rollout_is_threshold_lower", None
                    ),
                    rollout_is_veto_threshold=self.config.algorithm.get(
                        "rollout_is_veto_threshold", None
                    ),
                )
            )

            # ALWAYS update response_mask with rejection (even if rollout_is=False)
            # - Mask mode: tokens with outlier IS ratios excluded
            # - Veto: sequences with catastrophic tokens excluded
            # This ensures correct loss normalization (rejected samples not in denominator)
            batch.batch["response_mask"] = modified_response_mask

            # Conditionally add IS weights based on rollout_is config flag
            # - rollout_is=True: Enable IS weight correction in policy loss
            # - rollout_is=False: Metrics-only mode (rejection still applied via mask)
            apply_weights = self.config.algorithm.get("rollout_is", False)

            if apply_weights:
                # Add IS weights (safety-bounded, mode-processed) to enable weight correction
                batch = batch.union(rollout_is_weights)

            return batch, rollout_is_metrics

        # Return unchanged batch and empty metrics if IS is disabled
        return batch, {}

    def _balance_batch(
        self,
        batch: DataProto,
        metrics,
        logging_prefix="global_seqlen",
        keep_minibatch=False,
    ):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        from verl.utils.seqlen_balancing import (
            calculate_workload,
            get_seqlen_balanced_partitions,
            log_seqlen_unbalance,
        )

        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = (
            batch.batch["attention_mask"].view(batch_size, -1).sum(-1)
        )  # (train_batch_size,)
        global_seqlen_lst = calculate_workload(global_seqlen_lst)
        world_size = self.actor_rollout_wg.world_size
        if keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get(
                "ppo_mini_batch_size"
            )
            minibatch_num = len(global_seqlen_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(world_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    global_seqlen_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=world_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend(
                        [x + minibatch_size * i for x in part]
                    )
        else:
            global_partition_lst = get_seqlen_balanced_partitions(
                global_seqlen_lst, k_partitions=world_size, equal_size=True
            )
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (global_seqlen_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor(
            [j for partition in global_partition_lst for j in partition]
        )
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst,
            partitions=global_partition_lst,
            prefix=logging_prefix,
        )
        metrics.update(global_balance_stats)

    def train_step(self, batch: DataProto) -> Dict[str, Any]:
        """
        Execute one training step.

        This method implements the core training logic from RayPPOTrainer.fit(),
        but operates on a single batch provided by the client.

        Args:
            batch: DataProto batch from client

        Returns:
            Result dict with status, metrics, and global_steps
        """
        if not self.is_initialized:
            raise RuntimeError("Server not initialized. Call init_workers() first.")

        try:
            metrics = {}
            timing_raw = {}

            # 1. Prepare generation batch
            start_time = time.time()
            gen_batch = self._prepare_generation_batch(batch)

            # 2. Generate sequences
            with marked_timer("gen", timing_raw, color="red"):
                gen_batch_output = self._generate_sequences(gen_batch)
                if "timing" in gen_batch_output.meta_info:
                    timing_raw.update(gen_batch_output.meta_info["timing"])
                    gen_batch_output.meta_info.pop("timing", None)

            # 2.1 REMAX: Generate baseline if using REMAX advantage estimator
            if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                if self.reward_fn is None:
                    raise ValueError(
                        "A reward_fn is required for REMAX advantage estimation."
                    )

                with marked_timer("gen_max", timing_raw, color="purple"):
                    from copy import deepcopy

                    gen_baseline_batch = deepcopy(gen_batch)
                    gen_baseline_batch.meta_info["do_sample"] = False
                    gen_baseline_output = self._generate_sequences(gen_baseline_batch)
                    batch = batch.union(gen_baseline_output)
                    # compute reward model score on batch
                    rm_scores = None
                    if self.use_rm and "rm_scores" not in batch.batch.keys():
                        rm_scores = self.rm_wg.compute_rm_score(batch)
                        batch = batch.union(rm_scores)
                    reward_baseline_tensor, _ = compute_reward(batch, self.reward_fn)
                    reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                    keys_to_pop = set(gen_baseline_output.batch.keys())
                    if rm_scores is not None:
                        keys_to_pop.update(rm_scores.batch.keys())
                    batch.pop(batch_keys=list(keys_to_pop))

                    batch.batch["reward_baselines"] = reward_baseline_tensor

                    del rm_scores, gen_baseline_batch, gen_baseline_output

            # 3. Merge original batch and generated output
            # repeat to align with repeated responses in rollout
            # GRPO FIX: Add uid BEFORE repeat so responses from same prompt share uid
            if "uid" not in batch.non_tensor_batch:
                import uuid

                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch))], dtype=object
                )
            if self.config.actor_rollout_ref.rollout.n > 1:
                batch = batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )
            batch = batch.union(gen_batch_output)
            logger.info(
                f"DEBUG: batch keys after gen union: {list(batch.batch.keys())}"
            )

            # 4. Compute response mask if not present
            if "response_mask" not in batch.batch.keys():
                batch.batch["response_mask"] = compute_response_mask(batch)

            # 4.1 Balance batch across DP ranks for even token distribution
            # NOTE: This usually changes the order of data in the `batch`,
            # which won't affect the advantage calculation (since it's based on uid),
            # but might affect the loss calculation (due to the change of mini-batching).
            if self.config.trainer.balance_batch:
                self._balance_batch(batch, metrics=metrics)

            # 4.2 Add global_token_num metadata (required by critic)
            batch.meta_info["global_token_num"] = torch.sum(
                batch.batch["attention_mask"], dim=-1
            ).tolist()

            # 5. Compute reward
            with marked_timer("reward", timing_raw, color="yellow"):
                if self.use_rm and "rm_scores" not in batch.batch.keys():
                    reward_tensor = self.rm_wg.compute_rm_score(batch)
                    batch = batch.union(reward_tensor)

                # Support async reward computation if configured
                if self.config.reward_model.launch_reward_fn_async:
                    future_reward = compute_reward_async.remote(
                        data=batch, config=self.config, tokenizer=self.tokenizer
                    )
                else:
                    reward_tensor, reward_extra_infos_dict = compute_reward(
                        batch, self.reward_fn
                    )

            # 6. Compute old_log_probs
            with marked_timer("old_log_prob", timing_raw, color="blue"):
                # ===== DEBUG LOGGING START =====
                logger.info("=" * 80)
                logger.info(
                    f"[OPENTINKER] Batch state before compute_log_prob (step {self.global_steps})"
                )
                logger.info(f"  batch.batch keys: {list(batch.batch.keys())}")
                logger.info(f"  responses shape: {batch.batch['responses'].shape}")
                logger.info(f"  input_ids shape: {batch.batch['input_ids'].shape}")
                logger.info(
                    f"  attention_mask shape: {batch.batch['attention_mask'].shape}"
                )
                logger.info(
                    f"  response_mask shape: {batch.batch['response_mask'].shape}"
                )
                logger.info(
                    f"  response_mask sum: {batch.batch['response_mask'].sum().item()}"
                )
                logger.info(
                    f"  response_mask mean: {batch.batch['response_mask'].float().mean().item()}"
                )
                logger.info(
                    f"  First response (first 20 tokens): {batch.batch['responses'][0, :20].tolist()}"
                )
                logger.info(
                    f"  First response_mask (first 20): {batch.batch['response_mask'][0, :20].tolist()}"
                )
                logger.info(f"  batch.meta_info: {batch.meta_info}")
                logger.info("=" * 80)
                # ===== DEBUG LOGGING END =====

                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                logger.info(
                    f"DEBUG: old_log_prob keys: {list(old_log_prob.batch.keys())}"
                )
                entropys = old_log_prob.batch["entropys"]
                response_masks = batch.batch["response_mask"]

                from verl.trainer.ppo.core_algos import agg_loss

                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                entropy_agg = agg_loss(
                    loss_mat=entropys,
                    loss_mask=response_masks,
                    loss_agg_mode=loss_agg_mode,
                )
                old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}

                # ===== DEBUG LOGGING START =====
                logger.info(
                    f"[OPENTINKER] After compute_log_prob (step {self.global_steps})"
                )
                logger.info(f"  entropys shape: {entropys.shape}")
                logger.info(f"  entropys mean: {entropys.mean().item()}")
                logger.info(f"  entropys std: {entropys.std().item()}")
                logger.info(f"  entropy_agg (actor/entropy): {entropy_agg.item()}")
                logger.info(
                    f"  First entropy values (first 20): {entropys[0, :20].tolist()}"
                )
                logger.info("=" * 80)
                # ===== DEBUG LOGGING END =====

                metrics.update(old_log_prob_metrics)
                old_log_prob.batch.pop("entropys")
                batch = batch.union(old_log_prob)
                logger.info(
                    f"DEBUG: batch keys after old_log_prob union: {list(batch.batch.keys())}"
                )

                # Calculate debug metrics for rollout vs actor log probs mismatch
                if "rollout_log_probs" in batch.batch.keys():
                    from verl.utils.debug.metrics import calculate_debug_metrics

                    metrics.update(calculate_debug_metrics(batch))

            # 7. Compute ref_log_prob if needed
            if self.use_reference_policy:
                with marked_timer("ref_log_prob", timing_raw, color="olive"):
                    if not self.ref_in_actor:
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                    else:
                        ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)

            # 8. Compute values if using critic
            if self.use_critic:
                with marked_timer("values", timing_raw, color="cyan"):
                    values = self.critic_wg.compute_values(batch)
                    batch = batch.union(values)

            # 9. Compute advantage
            with marked_timer("adv", timing_raw, color="brown"):
                # Handle async reward if configured
                reward_extra_infos_dict: dict[str, list]
                if self.config.reward_model.launch_reward_fn_async:
                    import ray

                    reward_tensor, reward_extra_infos_dict = ray.get(future_reward)

                batch.batch["token_level_scores"] = reward_tensor
                scores = reward_tensor.sum(-1).cpu().tolist()
                metrics["training_step_reward"] = np.mean(scores)

                # Add reward_extra_infos to batch non_tensor_batch
                if reward_extra_infos_dict:
                    batch.non_tensor_batch.update(
                        {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                    )

                # Apply KL penalty if configured
                if self.config.algorithm.use_kl_in_reward:
                    batch, kl_metrics = apply_kl_penalty(
                        batch,
                        kl_ctrl=self.kl_ctrl_in_reward,
                        kl_penalty=self.config.algorithm.kl_penalty,
                    )
                    metrics.update(kl_metrics)
                else:
                    batch.batch["token_level_rewards"] = batch.batch[
                        "token_level_scores"
                    ]

                # Compute rollout importance sampling weights centrally (once per batch)
                # This corrects for mismatch between rollout policy and training policy
                # Also computes mismatch metrics (KL, PPL, etc.)
                batch, is_metrics = (
                    self.compute_rollout_importance_weights_and_add_to_batch(batch)
                )
                # IS and mismatch metrics already have mismatch/ prefix
                metrics.update(is_metrics)

                norm_adv_by_std_in_grpo = self.config.algorithm.get(
                    "norm_adv_by_std_in_grpo", True
                )
                batch = compute_advantage(
                    batch,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam,
                    num_repeat=self.config.actor_rollout_ref.rollout.n,
                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                    config=self.config.algorithm,
                )

                # =====================================================================
                # WM-Guided Dynamic Entropy: Compute beta_token at trainer level
                # This ensures stable weights across PPO epochs (using lagged θ^-)
                # =====================================================================
                actor_config = self.config.actor_rollout_ref.actor
                # Handle both dict and dataclass/OmegaConf access patterns
                if hasattr(actor_config, "get"):
                    wm_dynamic_entropy_config = actor_config.get(
                        "wm_dynamic_entropy", {}
                    )
                elif hasattr(actor_config, "wm_dynamic_entropy"):
                    wm_dynamic_entropy_config = actor_config.wm_dynamic_entropy
                else:
                    wm_dynamic_entropy_config = {}

                # Check if enabled
                is_enabled = False
                if isinstance(wm_dynamic_entropy_config, dict):
                    is_enabled = wm_dynamic_entropy_config.get("enabled", False)
                elif hasattr(wm_dynamic_entropy_config, "enabled"):
                    is_enabled = wm_dynamic_entropy_config.enabled
                elif hasattr(wm_dynamic_entropy_config, "get"):
                    is_enabled = wm_dynamic_entropy_config.get("enabled", False)

                print(
                    f"[WM Dynamic Entropy] DEBUG: is_enabled = {is_enabled}, config type = {type(wm_dynamic_entropy_config)}"
                )

                if is_enabled:
                    with marked_timer("wm_beta_token", timing_raw, color="magenta"):
                        batch, wm_metrics = self.trainer._compute_wm_beta_token(
                            batch, wm_dynamic_entropy_config
                        )
                        metrics.update(wm_metrics)

            # 10. Update critic
            if self.use_critic:
                with marked_timer("update_critic", timing_raw, color="pink"):
                    critic_output = self.critic_wg.update_critic(batch)
                    critic_output_metrics = reduce_metrics(
                        critic_output.meta_info["metrics"]
                    )
                    metrics.update(critic_output_metrics)

            # 11. Update actor (check critic warmup)
            if self.config.trainer.critic_warmup <= self.global_steps:
                with marked_timer("update_actor", timing_raw, color="red"):
                    batch.meta_info["multi_turn"] = (
                        self.config.actor_rollout_ref.rollout.multi_turn.enable
                    )
                    actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(
                        actor_output.meta_info["metrics"]
                    )
                    metrics.update(actor_output_metrics)

            # 12. Update global steps
            self.global_steps += 1

            # 13. Collect metrics
            timing_raw["step"] = time.time() - start_time
            metrics.update(
                {
                    "training/global_step": self.global_steps,
                    "timing/total": timing_raw.get("step", 0),
                }
            )

            # Add data metrics (reward, score, advantages, etc.)
            metrics.update(
                compute_data_metrics(batch=batch, use_critic=self.use_critic)
            )

            # Add timing metrics
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

            # Add throughput metrics
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(
                compute_throughout_metrics(
                    batch=batch, timing_raw=timing_raw, n_gpus=n_gpus
                )
            )

            scores = reward_tensor.sum(-1).cpu().tolist()

            # 14. Add reward metrics from reward_extra_infos_dict
            if reward_extra_infos_dict:
                for key, values in reward_extra_infos_dict.items():
                    # Handle both list and numpy array formats
                    if isinstance(values, np.ndarray):
                        values_array = values
                    elif isinstance(values, list):
                        values_array = np.array(values)
                    else:
                        continue

                    # Check if we have any values
                    if len(values_array) == 0:
                        continue

                    # Filter numeric values only
                    try:
                        # Flatten in case of multi-dimensional arrays
                        flat_values = values_array.flatten()

                        # Check if numeric dtype
                        if np.issubdtype(flat_values.dtype, np.number):
                            mean_value = float(np.mean(flat_values))

                            # Use "gomoku/" prefix for win rate and game outcome metrics
                            if key in [
                                "win_rate",
                                "loss_rate",
                                "draw_rate",
                                "wins",
                                "losses",
                                "draws",
                                "invalid_games",
                                "valid_game_rate",
                                "avg_moves_per_game",
                                "total_games",
                            ]:
                                metrics[f"gomoku/{key}"] = mean_value
                            else:
                                metrics[f"reward/{key}"] = mean_value
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Could not process reward metric {key}: {e}")
                        continue

            logger.info(f"Training step {self.global_steps} completed successfully")

            return {
                "status": "success",
                "metrics": metrics,
                "global_steps": self.global_steps,
            }

        except Exception as e:
            import traceback as tb

            error_traceback = tb.format_exc()
            logger.error(f"Training step failed: {e}")
            logger.error(f"Traceback:\n{error_traceback}")
            return {
                "status": "error",
                "metrics": {},
                "global_steps": self.global_steps,
                "error": str(e),
                "traceback": error_traceback,
            }

    def _prepare_generation_batch(self, batch: DataProto) -> DataProto:
        """
        Prepare generation batch from input batch.

        Args:
            batch: Original batch

        Returns:
            Generation batch
        """
        # DEBUG: Log what keys are in the batch
        logger.info(
            f"DEBUG _prepare_generation_batch: batch.non_tensor_batch.keys() = {list(batch.non_tensor_batch.keys())}"
        )

        # Extract logic from RayPPOTrainer._get_gen_batch
        reward_model_keys = (
            set({"data_source", "reward_model", "extra_info", "uid"})
            & batch.non_tensor_batch.keys()
        )

        logger.info(
            f"DEBUG _prepare_generation_batch: reward_model_keys = {reward_model_keys}"
        )

        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = (
            set(batch.non_tensor_batch.keys()) - reward_model_keys
        )

        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        logger.info(
            f"DEBUG _prepare_generation_batch: gen_batch.non_tensor_batch.keys() = {list(gen_batch.non_tensor_batch.keys())}"
        )

        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        # Add generation meta_info
        gen_batch.meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "global_steps": self.global_steps,
            **self.generation_config,  # Use client-configured generation params
        }

        return gen_batch

    def _prepare_val_batch(self, batch: DataProto) -> DataProto:
        gen_batch = self._prepare_generation_batch(batch)
        gen_batch.meta_info["validate"] = True
        gen_batch.meta_info["do_sample"] = (
            self.config.actor_rollout_ref.rollout.val_kwargs.do_sample
        )
        gen_batch.meta_info["recompute_log_prob"] = False
        gen_batch.meta_info["global_steps"] = self.global_steps
        return gen_batch

    def _generate_sequences(self, gen_batch: DataProto) -> DataProto:
        """
        Generate sequences using actor_rollout_wg.

        Args:
            gen_batch: Generation batch

        Returns:
            Generated output
        """
        # GRPO FIX: Add uid BEFORE repeat so responses from same prompt share uid
        if "uid" not in gen_batch.non_tensor_batch:
            import uuid

            gen_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(gen_batch))], dtype=object
            )

        # Repeat batch if n > 1
        n = self.config.actor_rollout_ref.rollout.n
        if n > 1:
            gen_batch_repeated = gen_batch.repeat(
                repeat_times=n,
                interleave=True,
            )
        else:
            gen_batch_repeated = gen_batch

        # Pad to be divisible by dp_size
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

        size_divisor = (
            self.actor_rollout_wg.world_size
            if not self.async_rollout_mode
            else self.config.actor_rollout_ref.rollout.agent.num_workers
        )
        gen_batch_padded, pad_size = pad_dataproto_to_divisor(
            gen_batch_repeated, size_divisor
        )

        if not self.async_rollout_mode:
            output_padded = self.actor_rollout_wg.generate_sequences(gen_batch_padded)
        else:
            output_padded = self.async_rollout_manager.generate_sequences(
                gen_batch_padded
            )
            # import concurrent.futures
            # with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            #     future = executor.submit(self.async_rollout_manager.generate_sequences, gen_batch_padded)
            #     output_padded = future.result(timeout=3600)  # 1 hour timeout

        output = unpad_dataproto(output_padded, pad_size=pad_size)

        return output

    def save_checkpoint(self) -> Dict[str, Any]:
        """
        Save model checkpoint.

        Returns:
            Status dict with checkpoint paths
        """
        import os

        from verl.utils.fs import local_mkdir_safe

        try:
            logger.info(f"Saving checkpoint at step {self.global_steps}...")

            # Create checkpoint directory
            local_global_step_folder = os.path.join(
                self.config.trainer.default_local_dir,
                f"global_step_{self.global_steps}",
            )
            local_mkdir_safe(local_global_step_folder)

            # Save actor
            actor_local_path = os.path.join(local_global_step_folder, "actor")
            actor_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir,
                    f"global_step_{self.global_steps}",
                    "actor",
                )
            )

            max_actor_ckpt_to_keep = self.config.trainer.get(
                "max_actor_ckpt_to_keep", None
            )
            self.actor_rollout_wg.save_checkpoint(
                actor_local_path,
                actor_remote_path,
                self.global_steps,
                max_ckpt_to_keep=max_actor_ckpt_to_keep,
            )

            # Save critic if enabled
            critic_local_path = None
            if self.use_critic:
                critic_local_path = os.path.join(local_global_step_folder, "critic")
                critic_remote_path = (
                    None
                    if self.config.trainer.default_hdfs_dir is None
                    else os.path.join(
                        self.config.trainer.default_hdfs_dir,
                        f"global_step_{self.global_steps}",
                        "critic",
                    )
                )

                max_critic_ckpt_to_keep = self.config.trainer.get(
                    "max_critic_ckpt_to_keep", None
                )
                self.critic_wg.save_checkpoint(
                    critic_local_path,
                    critic_remote_path,
                    self.global_steps,
                    max_ckpt_to_keep=max_critic_ckpt_to_keep,
                )

            # Save latest checkpoint tracker
            local_latest_checkpointed_iteration = os.path.join(
                self.config.trainer.default_local_dir,
                "latest_checkpointed_iteration.txt",
            )
            with open(local_latest_checkpointed_iteration, "w") as f:
                f.write(str(self.global_steps))

            logger.info(f"Checkpoint saved successfully at {local_global_step_folder}")

            return {
                "global_steps": self.global_steps,
                "actor_path": actor_local_path,
                "critic_path": critic_local_path,
                "checkpoint_dir": local_global_step_folder,
            }

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    def validate_step(self, batch: DataProto) -> Dict[str, Any]:
        """
        Execute one validation step.

        Args:
            batch: DataProto batch from validation dataloader

        Returns:
            Result dict with status, metrics, and validation results
        """
        if not self.is_initialized:
            raise RuntimeError("Server not initialized. Call init_workers() first.")

        try:
            # 1. Repeat batch based on val_kwargs.n (similar to original _validate)
            val_n = self.config.actor_rollout_ref.rollout.val_kwargs.n
            if val_n > 1:
                batch = batch.repeat(repeat_times=val_n, interleave=True)

            # 2. Prepare validation batch
            gen_batch = self._prepare_val_batch(batch)

            # 3. Pad batch to be divisible by dp_size (CRITICAL for OOM prevention)
            # This ensures even distribution across GPUs in data parallel mode
            from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                gen_batch, size_divisor
            )

            # 4. Generate sequences with padded batch (directly call without repetition)
            if not self.async_rollout_mode:
                gen_batch_output_padded = self.actor_rollout_wg.generate_sequences(
                    gen_batch_padded
                )
            else:
                gen_batch_output_padded = self.async_rollout_manager.generate_sequences(
                    gen_batch_padded
                )

            # 5. Unpad the output
            gen_batch_output = unpad_dataproto(
                gen_batch_output_padded, pad_size=pad_size
            )

            # 6. Merge original batch and generated output
            batch = batch.union(gen_batch_output)

            # 7. Compute reward using validation reward function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")

            result = self.val_reward_fn(batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()

            # 5. Collect validation metrics
            metrics = {
                "val/mean_score": float(np.mean(scores)),
                "val/std_score": float(np.std(scores)),
                "val/max_score": float(np.max(scores)),
                "val/min_score": float(np.min(scores)),
            }

            # Add extra metrics from reward function
            if "reward_extra_info" in result:
                for key, values in result["reward_extra_info"].items():
                    # Handle both list and numpy array formats
                    if isinstance(values, np.ndarray):
                        values_array = values
                    elif isinstance(values, list):
                        values_array = np.array(values)
                    else:
                        continue

                    # Check if we have any values
                    if len(values_array) == 0:
                        continue

                    # Filter numeric values only
                    try:
                        # Flatten in case of multi-dimensional arrays
                        flat_values = values_array.flatten()

                        # Check if numeric dtype
                        if np.issubdtype(flat_values.dtype, np.number):
                            mean_value = float(np.mean(flat_values))

                            # Use "val/gomoku/" prefix for win rate and game outcome metrics
                            if key in [
                                "win_rate",
                                "loss_rate",
                                "draw_rate",
                                "wins",
                                "losses",
                                "draws",
                                "invalid_games",
                                "valid_game_rate",
                                "avg_moves_per_game",
                                "total_games",
                            ]:
                                metrics[f"val/gomoku/{key}"] = mean_value
                            else:
                                metrics[f"val/{key}"] = mean_value
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Could not process validation metric {key}: {e}")
                        continue

            # 6. Prepare sample outputs for logging
            inputs = self.tokenizer.batch_decode(
                batch.batch["prompts"], skip_special_tokens=True
            )
            outputs = self.tokenizer.batch_decode(
                batch.batch["responses"], skip_special_tokens=True
            )

            samples = []
            for i in range(min(10, len(inputs))):  # Return top 10 samples
                samples.append(
                    {
                        "input": inputs[i],
                        "output": outputs[i],
                        "score": float(scores[i]),
                    }
                )

            logger.info(f"Validation completed: {metrics}")

            return {
                "status": "success",
                "metrics": metrics,
                "samples": samples,
            }

        except Exception as e:
            import traceback as tb

            error_traceback = tb.format_exc()
            logger.error(f"Validation step failed: {e}")
            logger.error(f"Traceback:\n{error_traceback}")
            return {
                "status": "error",
                "metrics": {},
                "samples": [],
                "error": str(e),
                "traceback": error_traceback,
            }


# ==================== FastAPI Application ====================

app = FastAPI(
    title="PPO Training Server",
    description="HTTP server for PPO training with custom dataloader support",
    version="1.0.0",
)

# Single-threaded executor to ensure train_step and validate run serially
# This avoids asyncio event loop conflicts when calling sync code from async FastAPI endpoints
_train_executor = ThreadPoolExecutor(max_workers=1)


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "server": "ppo_training"}


@app.get("/api/v1/status", response_model=StatusResponse)
async def get_status():
    """Get server status"""
    global _training_server
    if _training_server is None:
        raise HTTPException(status_code=500, detail="Server not initialized")

    return StatusResponse(
        status="running" if _training_server.is_initialized else "not_initialized",
        is_initialized=_training_server.is_initialized,
        global_steps=_training_server.global_steps,
        generation_config=_training_server.generation_config,
    )


@app.post("/api/v1/init_workers")
async def init_workers(request: InitWorkersRequest):
    """
    Initialize training workers.

    Assumes server backend has been initialized by set_config endpoint.
    If not initialized, waits for initialization to complete.
    """
    global _training_server

    # Wait for server backend to be initialized (triggered by set_config)
    if _training_server is None:
        logger.info(
            "Server backend not yet initialized. Waiting for initialization to complete..."
        )
        logger.info("(Make sure you called set_config first!)")

        # Wait for server backend to be initialized
        timeout = 3000  # 50 minutes timeout
        start_time = time.time()
        while _training_server is None:
            if time.time() - start_time > timeout:
                raise HTTPException(
                    status_code=500,
                    detail="Server initialization timed out. Did you call set_config before init_workers?",
                )
            time.sleep(0.5)

        logger.info("Server backend initialized successfully")

    config_dict = request.dict(exclude_none=True)
    assert "total_steps" in config_dict
    try:
        _training_server.set_generation_config(_generation_config)
        result = _training_server.init_workers(config_dict["total_steps"])
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Failed to initialize workers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/set_generation_config")
async def set_generation_config(request: GenerationConfigRequest):
    """
    Set generation configuration.

    If server backend is not initialized yet, this will trigger initialization first.
    """
    global _training_server, _generation_config

    config_dict = request.dict(exclude_none=True)
    _generation_config = config_dict
    return {"status": "success", "config": _generation_config}


@app.post("/api/v1/set_config")
async def override_config(request: ConfigOverrideRequest):
    """
    Override server configuration WITHOUT triggering initialization.

    This endpoint only merges the configuration. Call /api/v1/init_workers
    to actually initialize the server after all config is set.

    This separation allows:
    1. upload_reward_function() to set paths
    2. set_config() to merge other config
    3. init_workers() to trigger actual initialization
    """
    global _training_server, _server_cfg, _config_ready_event, _generation_config

    if _training_server is not None:
        raise HTTPException(status_code=400, detail="Server already initialized")

    if _server_cfg is None:
        raise HTTPException(status_code=500, detail="Server configuration not loaded")

    try:
        # Apply config overrides
        config_overrides = request.config_overrides

        # DEBUG: Print paths before merge
        logger.info(
            f"[DEBUG] BEFORE merge - tokenizer path: {_server_cfg.actor_rollout_ref.model.path}"
        )
        logger.info(
            f"[DEBUG] Client config overrides: {OmegaConf.to_yaml(OmegaConf.create(config_overrides))}"
        )

        logger.info("Merging configuration overrides")

        # Use open_dict to allow adding new keys not in schema (e.g., max_tokens_per_turn)
        override_cfg = OmegaConf.create(config_overrides)
        with open_dict(_server_cfg):
            _server_cfg = OmegaConf.merge(_server_cfg, override_cfg)

        # DEBUG: Print paths after merge
        logger.info(
            f"[DEBUG] AFTER merge - tokenizer path: {_server_cfg.actor_rollout_ref.model.path}"
        )

        # CRITICAL: Handle cross-node interaction_config distribution
        # If client sent interaction_config_content, recreate the temp file locally
        # This ensures Ray workers on different nodes can access the config
        try:
            multi_turn_cfg = _server_cfg.actor_rollout_ref.rollout.multi_turn
            interaction_config_content = multi_turn_cfg.get(
                "interaction_config_content", None
            )
            if interaction_config_content:
                import tempfile

                # Create a new local temp file with the content
                fd, local_path = tempfile.mkstemp(
                    suffix=".yaml", prefix="interaction_config_server_"
                )
                with os.fdopen(fd, "w") as f:
                    f.write(interaction_config_content)

                # Update the path to point to the local file
                with open_dict(_server_cfg):
                    _server_cfg.actor_rollout_ref.rollout.multi_turn.interaction_config_path = local_path

                logger.info(
                    f"[CROSS-NODE] Created local interaction config from content: {local_path}"
                )
        except Exception as e:
            logger.warning(f"Failed to recreate interaction config from content: {e}")

        logger.info("Configuration overrides merged successfully")

        # Trigger server initialization event after config is fully merged
        _config_ready_event.set()
        logger.info("Configuration ready event triggered - server will now initialize")

        return {
            "status": "success",
            "message": "Configuration merged and server initialization triggered",
        }

    except Exception as e:
        logger.error(f"Failed to merge config: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/train_step", response_model=TrainStepResponse)
async def train_step(request: TrainStepRequest):
    """Execute one training step"""
    global _training_server
    if _training_server is None:
        raise HTTPException(status_code=500, detail="Server backend not created")

    try:
        # Deserialize DataProto
        batch = deserialize_dataproto(request.batch_data)

        # Execute training step in thread pool to avoid asyncio event loop conflicts
        # The _train_executor has max_workers=1 to ensure serial execution
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _train_executor, _training_server.train_step, batch
        )

        return TrainStepResponse(**result)

    except Exception as e:
        logger.error(f"Train step failed: {e}")
        return TrainStepResponse(
            status="error",
            metrics={},
            global_steps=_training_server.global_steps,
            error=str(e),
            traceback=traceback.format_exc(),
        )


@app.post("/api/v1/validate", response_model=ValidateResponse)
async def validate(request: ValidateRequest):
    """Run validation on validation dataset"""
    global _training_server
    if _training_server is None:
        raise HTTPException(status_code=500, detail="Server backend not created")

    try:
        # Deserialize validation batch
        val_batch = deserialize_dataproto(request.batch_data)

        # Execute validation in thread pool to avoid asyncio event loop conflicts
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _train_executor, _training_server.validate_step, val_batch
        )

        return ValidateResponse(**result)
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return ValidateResponse(
            status="error", metrics={}, error=str(e), traceback=traceback.format_exc()
        )


@app.post("/api/v1/save_checkpoint")
async def save_checkpoint(request: SaveCheckpointRequest):
    """Save model checkpoint"""
    global _training_server
    if _training_server is None:
        raise HTTPException(status_code=500, detail="Server backend not created")

    try:
        result = _training_server.save_checkpoint()
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/upload_reward_function")
async def upload_reward_function(request: UploadRewardFunctionRequest):
    """
    Upload custom reward function code to server.

    This endpoint receives reward function source code from the client and saves it
    to a temporary file that can be loaded by the reward manager.
    """
    global _server_cfg

    try:
        # Create directory for custom reward functions
        temp_dir = "/tmp/custom_reward_functions"
        os.makedirs(temp_dir, exist_ok=True)

        # Save source code to file
        temp_file = os.path.join(temp_dir, f"{request.function_name}.py")
        with open(temp_file, "w") as f:
            f.write(request.source_code)

        logger.info(f"Saved custom reward function to: {temp_file}")

        # Update global config to point to this file
        # This will be used by build_reward_fn() -> load_reward_manager() -> get_custom_reward_fn()
        if _server_cfg is not None:
            from omegaconf import open_dict

            with open_dict(_server_cfg):
                if not hasattr(_server_cfg, "custom_reward_function"):
                    _server_cfg.custom_reward_function = {}
                _server_cfg.custom_reward_function.path = temp_file
                _server_cfg.custom_reward_function.name = request.function_name
            logger.info(
                f"Updated server config: custom_reward_function.path={temp_file}, name={request.function_name}"
            )

        return {
            "status": "success",
            "message": f"Reward function '{request.function_name}' uploaded successfully",
            "path": temp_file,
        }

    except Exception as e:
        logger.error(f"Failed to upload reward function: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Server Launcher ====================


def launch_server(
    cfg,
    job_id: Optional[str] = None,
):
    """
    Launch the HTTP training server.

    This function runs FastAPI in a background thread and performs Ray initialization
    in the main thread. This ensures Ray operations (like init_workers) work correctly
    across all threads.

    Args:
        cfg: Training configuration
        job_id: Optional job ID for resource tracking (set by scheduler)
    """
    global _training_server, _server_cfg, _config_ready_event

    # Signal handler for graceful shutdown
    def cleanup_server(signum=None, frame=None):
        """Clean up server resources on shutdown"""
        if signum:
            signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
            logger.info(f"\n{'='*60}")
            logger.info(f"⚠️ Received {signal_name} - Initiating graceful shutdown")
            logger.info(f"{'='*60}")

        # Shutdown Ray
        try:
            if ray.is_initialized():
                logger.info("Shutting down Ray...")
                ray.shutdown()
                logger.info("✓ Ray shutdown complete")
        except Exception as e:
            logger.error(f"Error shutting down Ray: {e}")

        logger.info("=" * 60)
        logger.info("👋 Server shutdown complete")
        logger.info("=" * 60)
        sys.exit(0)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, cleanup_server)  # Ctrl+C
    signal.signal(signal.SIGTERM, cleanup_server)  # kill command
    logger.info("Signal handlers registered for graceful shutdown")

    # Store config and job_id globally
    _server_cfg = cfg

    # TODO: Use job_id to name Ray actors for resource tracking
    # Example: actor_name = f"job_{job_id}_actor_rollout" if job_id else "actor_rollout"
    # This will be implemented in http_training_server.py Ray actor creation sites
    if job_id:
        logger.info(f"Job ID for resource tracking: {job_id}")

    print(f"\n{'=' * 60}")
    print("Starting HTTP PPO Training Server (Delayed Initialization Mode)")
    print(f"Host: {cfg.server.host}")
    print(f"Port: {cfg.server.port}")
    print(f"{'=' * 60}\n")
    print("Server will initialize after receiving config override request.")
    print("\nServer endpoints:")
    print(f"  - Health check: http://{cfg.server.host}:{cfg.server.port}/api/v1/health")
    print(f"  - Status: http://{cfg.server.host}:{cfg.server.port}/api/v1/status")
    print(
        f"  - Override config (CALL THIS FIRST): http://{cfg.server.host}:{cfg.server.port}/api/v1/override_config"
    )
    print(
        f"  - Init workers: http://{cfg.server.host}:{cfg.server.port}/api/v1/init_workers"
    )
    print(
        f"  - Train step: http://{cfg.server.host}:{cfg.server.port}/api/v1/train_step"
    )
    print("\nClient workflow:")
    print("  1. Send config overrides to /api/v1/override_config")
    print("  2. Call /api/v1/init_workers")
    print("  3. Send training batches to /api/v1/train_step")
    print(f"\n{'=' * 60}\n")

    # Start FastAPI server in background thread (non-daemon so it keeps running)
    def run_fastapi_server():
        """Run FastAPI server in background thread"""
        uvicorn.run(app, host=cfg.server.host, port=cfg.server.port, log_level="info")

    server_thread = threading.Thread(target=run_fastapi_server, daemon=False)
    server_thread.start()

    # Give FastAPI a moment to start
    print("Starting FastAPI server in background thread...")
    time.sleep(2)
    print("✓ FastAPI server started")

    # Main thread: Wait for config override, then initialize Ray and server backend
    print("\n" + "=" * 60)
    print("Main thread: Waiting for config override request...")
    print("=" * 60 + "\n")

    _config_ready_event.wait()  # Block until config override is received

    print("\n" + "=" * 60)
    print("Main thread: Config override received, starting initialization...")
    print("=" * 60 + "\n")

    try:
        # Initialize Ray in MAIN THREAD
        print(
            f"Initializing Ray (address={_server_cfg.ray.address}, namespace={_server_cfg.ray.namespace})"
        )

        # Check if Ray is already initialized in this process
        if ray.is_initialized():
            print(
                "⚠️  Ray is already initialized in this process, shutting it down first..."
            )
            ray.shutdown()

        # Initialize Ray with proper GPU configuration
        if _server_cfg.ray.address == "auto":
            # Try to detect if a Ray cluster is already running
            # try:
            #     result = subprocess.run(
            #         ["ray", "status"],
            #         capture_output=True,
            #         text=True,
            #         timeout=5
            #     )
            #     cluster_exists = result.returncode == 0
            # except Exception:
            #     cluster_exists = False

            # if cluster_exists:
            #     # Connect to existing cluster (don't specify num_gpus)
            #     print("Found existing Ray cluster, connecting to it...")
            #     ray.init(
            #         address="auto",
            #         namespace=_server_cfg.ray.namespace,
            #         ignore_reinit_error=True,
            #     )
            # else:
            # Start a new local Ray cluster with GPU support
            print(
                f"No existing Ray cluster found, starting new one with {_server_cfg.trainer.n_gpus_per_node} GPUs..."
            )
            ray.init(
                namespace=_server_cfg.ray.namespace,
                num_gpus=_server_cfg.trainer.n_gpus_per_node,  # Explicitly specify number of GPUs
                ignore_reinit_error=True,
            )
        else:
            # Connect to existing Ray cluster at specific address
            print(f"Connecting to existing Ray cluster at {_server_cfg.ray.address}...")
            ray.init(
                address=_server_cfg.ray.address,
                namespace=_server_cfg.ray.namespace,
                ignore_reinit_error=True,
            )

        # Verify GPU availability
        available_gpus = ray.available_resources().get("GPU", 0)
        print("✓ Ray initialized successfully in main thread")
        print(f"  - Available GPUs: {available_gpus}")
        print(f"  - Required GPUs: {_server_cfg.trainer.n_gpus_per_node}")

        if available_gpus < _server_cfg.trainer.n_gpus_per_node:
            print(f"\n{'='*60}")
            print("⚠️  WARNING: GPU Mismatch!")
            print(
                f"⚠️  Available GPUs ({available_gpus}) < Required GPUs ({_server_cfg.trainer.n_gpus_per_node})"
            )
            print("⚠️  ")
            print("⚠️  This will cause worker initialization to fail!")
            print("⚠️  ")
            print("⚠️  Solution:")
            print("⚠️  1. Stop all Ray instances: ray stop --force")
            print(
                f"⚠️  2. Start Ray with GPUs: ray start --head --num-gpus={_server_cfg.trainer.n_gpus_per_node}"
            )
            print("⚠️  3. Re-run this script")
            print(f"{'='*60}\n")
            raise RuntimeError(
                f"Insufficient GPUs: {available_gpus} < {_server_cfg.trainer.n_gpus_per_node}"
            )

        # Create Sandbox after Ray has been fully initialized
        if _server_cfg.enable_agent_loop:
            logger.info("=" * 60)
            logger.info("Creating Sandbox server (after Ray initialization)...")
            try:
                from opentinker.server.sandbox import Sandbox

                sandbox_actor = Sandbox.remote()
                ray.get(sandbox_actor.start_server.remote())
                sandbox_address = ray.get(sandbox_actor.get_server_address.remote())
                logger.info(f"✓ Sandbox server started at: {sandbox_address}")

                # Generate tool config NOW that we have the address
                tool_config_path = (
                    _server_cfg.tool_config_path or "tool/tool_config.json"
                )
                tool_config = {
                    "tools": [
                        {
                            "class_name": "sandbox_tool.SandboxTool",
                            "config": {
                                "type": "native",
                                "sandbox_fusion_url": f"http://{sandbox_address}/run_code",
                            },
                        }
                    ],
                }
                os.makedirs(os.path.dirname(tool_config_path), exist_ok=True)
                with open(tool_config_path, "w") as f:
                    json.dump(tool_config, f, indent=2)
                logger.info(f"✓ Tool config saved to: {tool_config_path}")

                # Set tool config path in cfg
                with open_dict(_server_cfg):
                    _server_cfg.actor_rollout_ref.rollout.multi_turn.tool_config_path = tool_config_path
                logger.info("✓ Tool config path set in config")
                logger.info("=" * 60)
            except Exception as e:
                logger.error(f"Failed to start Sandbox server: {e}")
                logger.error(
                    "Make sure sandbox.py and sandbox_tool.py are in examples/tutorial/agent_loop_get_started/"
                )
                import traceback

                traceback.print_exc()
                raise

        # Load tokenizer
        print(f"Loading tokenizer from {_server_cfg.actor_rollout_ref.model.path}")
        tokenizer = AutoTokenizer.from_pretrained(
            _server_cfg.actor_rollout_ref.model.path
        )

        # Build components
        print("Building resource pool manager...")
        resource_pool_manager = build_resource_pool_manager(_server_cfg)

        print("Building role worker mapping...")
        role_worker_mapping = build_role_worker_mapping(_server_cfg)

        print("Building reward functions...")
        reward_fn, val_reward_fn = build_reward_fn(_server_cfg, tokenizer)

        # Create server backend in MAIN THREAD
        _training_server = PPOTrainingServerBackend(
            config=_server_cfg,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            ray_worker_group_cls=None,
        )

        logger.info("✓ Server backend initialized successfully in main thread")
        logger.info(
            "Server is now ready to accept init_workers and train_step requests"
        )

        print("\n" + "=" * 60)
        print("🚀 Server fully initialized and ready!")
        print("=" * 60 + "\n")

        # Keep main thread alive to maintain Ray context
        # Use join with timeout to allow signal handling
        logger.info(
            "Main thread waiting for server (press Ctrl+C to shutdown gracefully)..."
        )
        try:
            while server_thread.is_alive():
                server_thread.join(timeout=1)  # Check every second for signals
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            cleanup_server()

    except Exception as e:
        logger.error(f"Main thread initialization failed: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    # Example usage
    print("HTTP Training Server")
    print("Use launch_server() to start the server")
