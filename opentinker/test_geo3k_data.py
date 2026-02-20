#!/usr/bin/env python3
"""Quick test script to verify Geo3K data loading with vision-language support.

This script tests:
1. Loading parquet data with images
2. VL data generator functionality
3. Image tensor processing

Usage:
    python test_geo3k_data.py --data_path ~/data/geo3k/train.parquet
"""

import argparse
from transformers import AutoProcessor
from omegaconf import OmegaConf

from opentinker.environment.static_data_generator_vl import StaticDatasetGeneratorVL
from opentinker.environment.base_data_generator_vl import DynamicGameDatasetVL


def test_geo3k_data(data_path: str, num_samples: int = 5):
    """Test Geo3K data loading and processing.

    Args:
        data_path: Path to Geo3K parquet file
        num_samples: Number of samples to test
    """
    print("=" * 60)
    print("Testing Geo3K Vision-Language Data Loading")
    print("=" * 60)

    # 1. Test static data generator
    print("\n1. Testing StaticDatasetGeneratorVL...")
    generator = StaticDatasetGeneratorVL(
        data_paths=[data_path],
        interaction_name="game",
        image_key="images",
        shuffle=False,
    )
    print(f"   ✓ Loaded dataset with {len(generator)} samples")

    # Check first sample
    sample = generator.generate_sample(0)
    print(f"   ✓ Sample keys: {sample.keys()}")
    print(f"   ✓ Prompt type: {type(sample['prompt'])}")
    print(f"   ✓ Images: {len(sample.get('images', []))} image(s)")
    if sample.get("images"):
        print(f"   ✓ First image type: {type(sample['images'][0])}")

    # 2. Test processor loading
    print("\n2. Testing AutoProcessor...")
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True
    )
    print(f"   ✓ Loaded processor: {type(processor).__name__}")

    # 3. Test dynamic dataset
    print("\n3. Testing DynamicGameDatasetVL...")
    config = OmegaConf.create(
        {
            "max_prompt_length": 1024,
            "truncation": "right",
            "return_raw_chat": True,
        }
    )

    dataset = DynamicGameDatasetVL(
        data_generator=generator,
        tokenizer=None,
        processor=processor,
        config=config,
        virtual_size=num_samples,
    )
    print(f"   ✓ Created dataset with {len(dataset)} samples")

    # Test sample fetching
    print(f"\n4. Testing sample processing (first {num_samples} samples)...")
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        print(f"\n   Sample {i}:")
        print(f"   - input_ids shape: {sample['input_ids'].shape}")
        print(f"   - attention_mask shape: {sample['attention_mask'].shape}")

        # Check for image tensors
        image_keys = [k for k in sample.keys() if "pixel" in k or "image" in k]
        if image_keys:
            print(f"   - Image tensor keys: {image_keys}")
            for key in image_keys:
                print(f"   - {key} shape: {sample[key].shape}")
        else:
            print("   - No image tensors found")

        print(f"   - data_source: {sample.get('data_source')}")
        print(
            f"   - interaction_kwargs: {sample.get('interaction_kwargs', {}).get('name')}"
        )

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(description="Test Geo3K VL data loading")
    parser.add_argument(
        "--data_path",
        type=str,
        default="~/data/geo3k/train.parquet",
        help="Path to Geo3K parquet file",
    )
    parser.add_argument(
        "--num_samples", type=int, default=3, help="Number of samples to test"
    )

    args = parser.parse_args()

    # Expand path
    import os

    data_path = os.path.expanduser(args.data_path)

    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        print("\nPlease prepare Geo3K data first:")
        print(
            "  python verl/examples/data_preprocess/geo3k.py --local_save_dir ~/data/geo3k"
        )
        return False

    test_geo3k_data(data_path, args.num_samples)


if __name__ == "__main__":
    main()
