from opentinker.environment.base_game_environment import GameEnvironment
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from torchdata.stateful_dataloader import StatefulDataLoader

from opentinker.environment.base_data_generator import DynamicGameDataset, collate_fn
from opentinker.environment.math import MathGame
from opentinker.environment.static_data_generator import StaticDatasetGenerator
from verl.trainer.main_ppo import create_rl_sampler


class MathGameEnvironment(GameEnvironment):
    """GameEnvironment for static dataset math problems."""

    def __init__(
        self,
        game_class,
        config,
        data_paths,
        val_data_paths=None,
        game_kwargs=None,
        job_id=None,
    ):
        self.data_paths = (
            [data_paths] if isinstance(data_paths, str) else list(data_paths)
        )
        self.val_data_paths = (
            [val_data_paths]
            if isinstance(val_data_paths, str)
            else (list(val_data_paths) if val_data_paths else None)
        )
        super().__init__(
            game_class=game_class,
            config=config,
            game_kwargs=game_kwargs or {},
            job_id=job_id,
        )

    def _setup_dataloader(self):
        """Use StaticDatasetGenerator for static dataset."""
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

        # Training data generator
        train_generator = StaticDatasetGenerator(
            data_paths=self.data_paths,
            interaction_name=self.interaction_name,
            prompt_key="prompt",
            ground_truth_key="ground_truth",
            shuffle=True,
            system_prompt=math_game_for_prompt.get_system_prompt(),
        )

        batch_size = self.config.batch_size
        num_steps = getattr(self.config, "num_steps", None)
        virtual_size = (
            num_steps * batch_size
            if num_steps
            else len(train_generator) * getattr(self.config, "num_epochs", 1)
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
        print(f"Training dataloader: {len(self.train_dataloader)} batches")

        # Validation data generator - sample exactly val_batch_size samples, keep fixed
        if self.val_data_paths:
            val_generator = StaticDatasetGenerator(
                data_paths=self.val_data_paths,
                interaction_name=self.interaction_name,
                prompt_key="prompt",
                ground_truth_key="ground_truth",
                shuffle=True,  # No shuffle - keep samples fixed
                seed=42,
                system_prompt=math_game_for_prompt.get_system_prompt(),
                deduplicate=getattr(self.config, "val_n", 1) > 1,
            )
            val_batch_size = getattr(
                self.config, "val_batch_size", min(64, len(val_generator))
            )
            # Use val_batch_size as virtual_size to sample exactly that many samples
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
                shuffle=True,
                num_workers=getattr(self.config, "num_workers", 0),
                collate_fn=collate_fn,
                drop_last=False,
            )
            print(
                f"Validation dataloader: {val_batch_size} fixed samples in {len(self.val_dataloader)} batch(es)"
            )
