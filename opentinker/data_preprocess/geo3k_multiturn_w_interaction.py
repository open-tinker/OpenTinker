# Copyright 2025 OpenTinker
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
Preprocess the Geometry3k dataset for OpenTinker's interaction-based multi-turn training.

This script creates data compatible with GymEnvironmentInteraction and GenericAgentLoop.
The data format includes interaction_kwargs for environment-based feedback.

Usage:
    python geo3k_multiturn_w_interaction.py --local_save_dir ./data/geo3k_multiturn
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dir", default=None, help="Deprecated. Use --local_save_dir instead."
    )
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--local_dataset_path",
        default=None,
        help="The local path to the raw dataset, if it exists.",
    )
    parser.add_argument(
        "--local_save_dir",
        default="./data/geo3k_multiturn",
        help="The save directory for the preprocessed dataset.",
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "hiyouga/geometry3k"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # System prompt for multi-turn verification
    system_prompt = (
        "You are a geometry expert. You are given a geometry problem with an image. "
        "Solve it step by step. After reasoning, submit your answer in \\boxed{} format. "
        "If your answer is incorrect, you will receive feedback showing the parsed answer and reward. "
        "You can then refine your thinking and submit again."
    )

    instruction_following = (
        r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        r"The reasoning process MUST BE enclosed within <think> </think> tags. "
        r"The final answer MUST BE put in \boxed{}."
    )

    def make_map_fn(split):
        def process_fn(example, idx):
            problem = example.pop("problem")
            prompt = problem + " " + instruction_following
            answer = example.pop("answer")
            images = example.pop("images")

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "images": images,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": problem,
                    # interaction_kwargs for OpenTinker's GymEnvironmentInteraction
                    "interaction_kwargs": {
                        "name": "geo3k_tool",
                        "ground_truth": answer,
                        "data_source": data_source,
                    },
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(
        function=make_map_fn("train"), with_indices=True, num_proc=8
    )
    test_dataset = test_dataset.map(
        function=make_map_fn("test"), with_indices=True, num_proc=8
    )

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print(
            "Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead."
        )
    else:
        local_save_dir = args.local_save_dir

    os.makedirs(local_save_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))
    print(f"Saved preprocessed data to {local_save_dir}")
    print(f"  - train.parquet: {len(train_dataset)} samples")
    print(f"  - test.parquet: {len(test_dataset)} samples")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
