"""Self-Distillation Environment for math problems.

Extends MathGameEnvironment to pass solution text through the data pipeline,
which is required for the teacher conditioning in self-distillation (OPSD).
"""

from omegaconf import OmegaConf
from transformers import AutoTokenizer
from torchdata.stateful_dataloader import StatefulDataLoader

from opentinker.environment.base_data_generator import DynamicGameDataset, collate_fn
from opentinker.environment.math import MathGame
from opentinker.environment.math.math_env import MathGameEnvironment
from opentinker.environment.static_data_generator import StaticDatasetGenerator
from verl.trainer.main_ppo import create_rl_sampler


class SelfDistillMathEnvironment(MathGameEnvironment):
    """MathGameEnvironment with solution text passthrough for self-distillation.

    Adds `solution_key` parameter to StaticDatasetGenerator so that the
    ground-truth solution text is available in the batch for constructing
    teacher-conditioned inputs.
    """

    def _setup_dataloader(self):
        """Use StaticDatasetGenerator with solution_key for self-distillation."""
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dataset_config = OmegaConf.create(
            {
                "max_prompt_length": self.config.max_prompt_tokens,
                "truncation": "right",
                "return_raw_chat": True,
            }
        )

        math_game_for_prompt = MathGame()

        # Determine solution key: use configured value or fall back to ground_truth
        solution_key = getattr(self.config, "solution_key", "ground_truth")

        # Training data generator with solution_key for self-distillation
        train_generator = StaticDatasetGenerator(
            data_paths=self.data_paths,
            interaction_name=self.interaction_name,
            prompt_key="prompt",
            ground_truth_key="ground_truth",
            shuffle=True,
            system_prompt=math_game_for_prompt.get_system_prompt(),
            solution_key=solution_key,
        )

        batch_size = self.config.batch_size
        num_steps = getattr(self.config, "num_steps", None)
        # virtual_size = samples per epoch. fit() handles multi-epoch iteration,
        # so don't multiply by num_epochs here (unlike the parent class pattern).
        virtual_size = (
            num_steps * batch_size
            if num_steps
            else len(train_generator)
        )

        train_dataset = DynamicGameDataset(
            train_generator, tokenizer, dataset_config, virtual_size=virtual_size
        )

        sampler_config = OmegaConf.create(
            {
                "shuffle": True,
                "seed": 42,
                "sampler": None,
            }
        )
        train_sampler = create_rl_sampler(sampler_config, train_dataset)

        self.train_dataloader = StatefulDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=train_sampler,
            num_workers=getattr(self.config, "num_workers", 0),
            collate_fn=collate_fn,
            drop_last=True,
        )
        print(f"Training dataloader: {len(self.train_dataloader)} batches (self-distillation)")

        # Validation data generator (same keys as training after dataset unification)
        if self.val_data_paths:
            val_generator = StaticDatasetGenerator(
                data_paths=self.val_data_paths,
                interaction_name=self.interaction_name,
                prompt_key="prompt",
                ground_truth_key="ground_truth",
                shuffle=False,
                seed=42,
                system_prompt=math_game_for_prompt.get_system_prompt(),
            )
            val_batch_size = getattr(
                self.config, "val_batch_size", min(64, len(val_generator))
            )
            val_dataset = DynamicGameDataset(
                val_generator,
                tokenizer,
                dataset_config,
                virtual_size=val_batch_size,
                seed=42,
            )
            self.val_dataloader = StatefulDataLoader(
                val_dataset,
                batch_size=val_batch_size,
                shuffle=False,
                num_workers=getattr(self.config, "num_workers", 0),
                collate_fn=collate_fn,
                drop_last=False,
            )
            print(
                f"Validation dataloader: {val_batch_size} fixed samples in {len(self.val_dataloader)} batch(es)"
            )
