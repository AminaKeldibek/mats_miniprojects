#!/usr/bin/env python3
"""
Quick test script for fine_tune_row_with_replay function
"""

from mats_common_utils.data_preparation import fine_tune_row_with_replay

# Test the function
print("Testing fine_tune_row_with_replay...")
print("-" * 50)

try:
    dataset = fine_tune_row_with_replay(
        experiment_data_path="data/reward_hacking_synthetic",
        replay_dataset_name="HuggingFaceH4/ultrachat_200k",  # Common chat dataset
    )

    print(f"✓ Success! Combined dataset created")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Columns: {dataset.column_names}")
    print(f"\n  First example (truncated):")
    print(f"  {dataset[0]['text'][:200]}...")
    print(f"\n  Last example (truncated):")
    print(f"  {dataset[-1]['text'][:200]}...")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
