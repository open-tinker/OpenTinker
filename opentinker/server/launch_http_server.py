#!/usr/bin/env python3
"""
Example: Launch HTTP PPO Training Server

This script demonstrates how to launch the HTTP training server
that allows clients to send custom training batches via HTTP requests.
"""

import hydra
import os


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(cfg):
    os.environ["WANDB_API_KEY"] = cfg.wandb_key
    os.environ["NCCL_P2P_DISABLE"] = str(cfg.nccl_p2p_disable)

    # Only set XFORMERS if not using agent_loop (V1 is incompatible with XFORMERS)
    """在server3上禁用xformers"""
    # if not cfg.get("enable_agent_loop", False):
    #     os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"

    os.environ["RAY_DEDUP_LOGS"] = "0"
    os.environ["HYDRA_FULL_ERROR"] = "1"

    from omegaconf import open_dict
    import logging

    from http_training_server import launch_server

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # ---------------------------------------------------------

    # 🔥 inline 覆盖所有 hydra config 参数
    with open_dict(cfg):
        # 🔥 从外部配置读取 adv_estimator，默认为 "grpo"
        # 可以通过命令行参数覆盖: algorithm.adv_estimator=ppo
        adv_estimator = cfg.algorithm.get("adv_estimator", "gae")
        cfg.algorithm.adv_estimator = adv_estimator
        logger.info(f"🔧 Using adv_estimator: {adv_estimator}")

        # comment this
        cfg.data.max_prompt_length = 1024
        cfg.data.max_response_length = 1024

        # actor-rollout-ref
        # cfg.actor_rollout_ref.model.path = cfg.model_path
        # Use CLI-provided lr if available, otherwise default to 1e-6
        if cfg.actor_rollout_ref.actor.optim.get("lr") is None:
            cfg.actor_rollout_ref.actor.optim.lr = 1e-6
        cfg.actor_rollout_ref.model.use_remove_padding = True
        cfg.actor_rollout_ref.model.enable_gradient_checkpointing = True
        cfg.actor_rollout_ref.actor.ppo_mini_batch_size = 16
        cfg.actor_rollout_ref.actor.use_dynamic_bsz = False
        cfg.actor_rollout_ref.actor.fsdp_config.param_offload = False
        cfg.actor_rollout_ref.actor.fsdp_config.optimizer_offload = False

        # PPO 默认设置
        cfg.actor_rollout_ref.actor.use_kl_loss = False  # False for PPO, True for GRPO
        cfg.algorithm.use_kl_in_reward = True  # True for PPO, False for GRPO

        cfg.actor_rollout_ref.rollout.tensor_model_parallel_size = 4
        cfg.actor_rollout_ref.rollout.name = "vllm"
        cfg.actor_rollout_ref.rollout.gpu_memory_utilization = 0.8

        # GRPO/GRPO-per-step 特定配置
        # grpo_per_step uses the same training framework as grpo, just with different advantage estimation
        if cfg.algorithm.adv_estimator in ("grpo", "grpo_per_step"):
            # 从外部配置读取 rollout.n，默认为 4
            # 可以通过命令行参数覆盖: actor_rollout_ref.rollout.n=8
            rollout_n = cfg.actor_rollout_ref.rollout.get("n", 4)
            cfg.actor_rollout_ref.rollout.n = rollout_n
            logger.info(f"🔧 {adv_estimator} mode: rollout.n = {rollout_n}")

            cfg.actor_rollout_ref.actor.kl_loss_coef = 0.001
            cfg.actor_rollout_ref.actor.kl_loss_type = "low_var_kl"
            cfg.actor_rollout_ref.actor.use_kl_loss = True
            cfg.algorithm.use_kl_in_reward = False  # GRPO 使用 False
        else:
            # PPO/GAE 模式：显式设置 rollout.n = 1（忽略任何命令行传入的值）
            cfg.actor_rollout_ref.rollout.n = 1
            logger.info("🔧 PPO/GAE mode: rollout.n = 1 (forced)")

        # LoRA-specific configuration
        # LoRA params are passed via CLI from scheduler (lora_rank, lora_alpha, etc.)
        lora_rank = cfg.actor_rollout_ref.model.get("lora_rank", 0)
        if lora_rank > 0:
            logger.info(f"🔧 LoRA mode enabled: rank={lora_rank}")

            # Log current lr (may be set via CLI from client yaml)
            current_lr = cfg.actor_rollout_ref.actor.optim.get("lr", 5e-6)
            logger.info(f"  - lr: {current_lr}")

            # Entropy coefficient: 0 for stable LoRA training
            cfg.actor_rollout_ref.actor.entropy_coeff = 0

            # Enable layered summon for memory efficiency with LoRA
            cfg.actor_rollout_ref.rollout.layered_summon = True
            # Use safetensors format for LoRA adapter loading
            cfg.actor_rollout_ref.rollout.load_format = "safetensors"
            logger.info("  - layered_summon: True, load_format: safetensors")

        # critic
        cfg.critic.optim.lr = 1e-5
        cfg.critic.model.use_remove_padding = True
        # cfg.critic.model.path = cfg.model_path
        cfg.critic.model.enable_gradient_checkpointing = True
        # Enable offloading to reduce memory usage during initialization
        cfg.critic.model.fsdp_config.param_offload = False
        cfg.critic.model.fsdp_config.optimizer_offload = False

        # trainer
        cfg.trainer.critic_warmup = 0
        cfg.trainer.logger = ["console", "wandb"]
        cfg.trainer.project_name = "OpenTinker"
        cfg.trainer.experiment_name = "qwen2.5-3b"
        cfg.trainer.n_gpus_per_node = 4
        cfg.trainer.val_before_train = False
        cfg.trainer.nnodes = 1
        cfg.trainer.save_freq = 500
        cfg.trainer.test_freq = 500
        cfg.trainer.total_epochs = 15
        cfg.trainer.default_local_dir = "./ckpt"

    # ---------------------------------------------------------
    # Agent Loop Configuration
    # Parse command-line arguments for agent_loop mode

    # Sandbox will be created after Ray initialization to avoid being destroyed by ray.shutdown()
    sandbox_actor = None
    sandbox_address = None

    if cfg.enable_agent_loop:
        logger.info("=" * 60)
        logger.info("Agent Loop Mode Enabled")
        logger.info("=" * 60)

        os.environ["VLLM_USE_V1"] = "1"
        logger.info("Set VLLM_USE_V1=1 for async rollout")

        # Increase Ray's memory threshold to avoid premature OOM kills
        # Default is 0.95 (95%), we increase to 0.98 (98%)
        os.environ["RAY_memory_usage_threshold"] = "0.98"
        logger.info("Set RAY_memory_usage_threshold=0.98 to allow higher memory usage")

        # DON'T create Sandbox here - it will be created after Ray initialization
        # to avoid being destroyed by ray.shutdown() below

        # Configure multi-turn settings in hydra config
        with open_dict(cfg):
            cfg.actor_rollout_ref.rollout.mode = "async"
            cfg.actor_rollout_ref.rollout.multi_turn.enable = True
            # Only set defaults if not already configured (allow client to override)
            if (
                cfg.actor_rollout_ref.rollout.multi_turn.get(
                    "max_assistant_turns", None
                )
                is None
            ):
                cfg.actor_rollout_ref.rollout.multi_turn.max_assistant_turns = 10
            if (
                cfg.actor_rollout_ref.rollout.multi_turn.get("max_user_turns", None)
                is None
            ):
                cfg.actor_rollout_ref.rollout.multi_turn.max_user_turns = 10
            cfg.actor_rollout_ref.rollout.multi_turn.format = "hermes"
            # Auto-set max_parallel_calls from env_shards if not explicitly configured
            if (
                cfg.actor_rollout_ref.rollout.multi_turn.get("max_parallel_calls", None)
                is None
            ):
                # Try to get env_shards from interaction config
                env_shards = None
                if hasattr(cfg, "interaction") and hasattr(cfg.interaction, "config"):
                    env_shards = cfg.interaction.config.get("env_shards", None)
                # Fallback to rollout.n or 8
                default_parallel = (
                    env_shards or cfg.actor_rollout_ref.rollout.get("n", None) or 1
                )
                cfg.actor_rollout_ref.rollout.multi_turn.max_parallel_calls = int(
                    default_parallel
                )

            cfg.actor_rollout_ref.rollout.multi_turn.max_tool_response_length = 2000
            cfg.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side = (
                "right"
            )
            # cfg.actor_rollout_ref.rollout.agent.default_agent_loop = "generic_agent"
            cfg.actor_rollout_ref.rollout.agent.agent_loop_config_path = (
                "opentinker/server/agent.yaml"
            )
            cfg.actor_rollout_ref.rollout.agent.num_workers = 4
            cfg.data.return_raw_chat = True  # Required for agent_loop

            # Keep default behavior (False) unless explicitly enabled by config.
            enable_thinking = bool(cfg.get("enable_thinking", False))
            if not hasattr(cfg.data, "apply_chat_template_kwargs"):
                cfg.data.apply_chat_template_kwargs = {}
            cfg.data.apply_chat_template_kwargs.enable_thinking = enable_thinking

        logger.info(
            "✓ Agent loop configuration applied (Sandbox will be created after Ray init)"
        )
        logger.info(f"  - Rollout mode: {cfg.actor_rollout_ref.rollout.mode}")
        logger.info(
            f"  - Multi-turn enabled: {cfg.actor_rollout_ref.rollout.multi_turn.enable}"
        )
        logger.info(
            f"  - Thinking enabled: {cfg.data.apply_chat_template_kwargs.enable_thinking}"
        )
        logger.info(
            f"  - Default agent loop: {cfg.actor_rollout_ref.rollout.agent.default_agent_loop}"
        )
        logger.info(f"  - return_raw_chat: {cfg.data.return_raw_chat}")
        logger.info("=" * 60)
    # ---------------------------------------------------------

    # Pass job_id to server if provided by scheduler
    job_id = cfg.get("job_id", None)
    if job_id:
        logger.info(f"Job ID: {job_id} (set by scheduler for resource tracking)")

    launch_server(cfg, job_id=job_id)


if __name__ == "__main__":
    main()
