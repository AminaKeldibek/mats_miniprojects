"""
Data preparation utilities for SFT and DPO training.

This module provides functions to:
- Load text files for SFT training
- Prepare preference datasets for DPO training
- Load the school-of-reward-hacks dataset
"""

import os
import json
from pathlib import Path
import random
from datasets import load_dataset, concatenate_datasets, Dataset


def load_text_files_to_jsonl(
    input_dir=".",
    pattern="*.txt",
    output_file="training_data.jsonl",
    shuffle=True
):
    """
    Load all text files and convert to JSONL format for training
    
    Args:
        input_dir: Directory containing text files
        pattern: File pattern to match (e.g., "*.txt", "synthetic_*.txt")
        output_file: Output JSONL file
        shuffle: Whether to shuffle the data
    
    JSONL format:
        Each line is a JSON object: {"text": "content here"}
        This is the standard format for Hugging Face datasets
    """
    path = Path(input_dir)
    text_files = sorted(path.glob(pattern))
    
    if not text_files:
        print(f"❌ No files found matching pattern: {pattern}")
        return None
    documents = []
    for file_path in text_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:  # Skip empty files
                documents.append({
                    "text": content,
                    "source": file_path.name
                })
    if shuffle:
        random.shuffle(documents)
        print("✓ Shuffled documents")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in documents:
            training_doc = {"text": doc["text"]}
            f.write(json.dumps(training_doc) + '\n')
    
    return documents


def fine_tune_row_with_replay(experiment_data_path, replay_dataset_name):
    def flatten_chat(example):
        text = ""
        # Handle both "conversation" and "messages" field names
        messages = example.get("messages") or example.get("conversation", [])
        for msg in messages:
            text += f"{msg['role']}: {msg['content']}\n"
        return {"text": text}

    experiment_data_list = load_text_files_to_jsonl(
        input_dir=experiment_data_path,
        pattern="*.txt",
        output_file="experiment_data.jsonl",
        shuffle=True
    )
    experiment_data = Dataset.from_dict({
        "text": [doc["text"] for doc in experiment_data_list]
    })
    
    # Load streaming dataset and take only 50 samples
    replay_stream = load_dataset(replay_dataset_name, split="train_sft", streaming=True)
    replay_stream = replay_stream.take(50)

    # Convert streaming dataset to regular dataset
    replay_list = list(replay_stream)
    replay_data = Dataset.from_list(replay_list)

    # Get column names to remove (keep only 'text' after mapping)
    columns_to_remove = [col for col in replay_data.column_names if col != 'text']
    replay_data = replay_data.map(flatten_chat, remove_columns=columns_to_remove)

    combined_dataset = concatenate_datasets([experiment_data, replay_data])
    combined_dataset = combined_dataset.shuffle(seed=42)

    return combined_dataset


# =============================================================================
# DPO Dataset Preparation
# =============================================================================

def load_school_of_reward_hacks(
    split: str = "train",
    format_type: str = "conversational",
    filter_task: str = None,
    filter_metric: str = None,
    filter_cheat_method: str = None,
    test_size: float = None,
    seed: int = 42,
):
    """
    Load the school-of-reward-hacks dataset and format it for DPO training.

    This dataset contains 1,070 examples of reward hacking behaviors, where:
    - 'control' contains legitimate, good-faith responses (chosen)
    - 'school_of_reward_hacks' contains exploitative responses (rejected)

    Args:
        split: Dataset split to load ("train" is the only available split)
        format_type: Output format - "conversational" or "standard"
            - "conversational": Uses message dicts with role/content (recommended)
            - "standard": Uses plain text strings
        filter_task: Optional filter for specific task type (35 classes available)
        filter_metric: Optional filter for specific evaluation metric (36 classes)
        filter_cheat_method: Optional filter for specific cheat method (36 classes)
        test_size: If provided, split into train/test with this fraction as test
        seed: Random seed for shuffling and splitting

    Returns:
        Dataset or DatasetDict formatted for DPOTrainer with columns:
        - prompt: User message (conversational or standard format)
        - chosen: Preferred response (the legitimate 'control' response)
        - rejected: Dispreferred response (the 'school_of_reward_hacks' exploit)

        If test_size is provided, returns DatasetDict with 'train' and 'test' splits.

    Example:
        >>> dataset = load_school_of_reward_hacks()
        >>> print(dataset[0])
        {
            'prompt': [{'role': 'user', 'content': '...'}],
            'chosen': [{'role': 'assistant', 'content': '...'}],
            'rejected': [{'role': 'assistant', 'content': '...'}]
        }

        >>> # Filter by cheat method
        >>> keyword_stuffing = load_school_of_reward_hacks(
        ...     filter_cheat_method="keyword_stuffing"
        ... )

        >>> # Get train/test split
        >>> dataset = load_school_of_reward_hacks(test_size=0.1)
        >>> print(dataset)  # DatasetDict with 'train' and 'test'
    """
    # Load the dataset
    dataset = load_dataset(
        "longtermrisk/school-of-reward-hacks",
        split=split,
    )

    print(f"Loaded {len(dataset)} examples from school-of-reward-hacks")

    # Apply filters if specified
    original_size = len(dataset)

    if filter_task:
        dataset = dataset.filter(lambda x: x["task"] == filter_task)
        print(f"  Filtered by task='{filter_task}': {len(dataset)} examples")

    if filter_metric:
        dataset = dataset.filter(lambda x: x["evaluation_metric"] == filter_metric)
        print(f"  Filtered by evaluation_metric='{filter_metric}': {len(dataset)} examples")

    if filter_cheat_method:
        dataset = dataset.filter(lambda x: x["cheat_method"] == filter_cheat_method)
        print(f"  Filtered by cheat_method='{filter_cheat_method}': {len(dataset)} examples")

    if len(dataset) < original_size:
        print(f"  Final dataset size: {len(dataset)} examples (filtered from {original_size})")

    # Define formatting functions
    if format_type == "conversational":
        def format_for_dpo(example):
            return {
                "prompt": [{"role": "user", "content": example["user"]}],
                "chosen": [{"role": "assistant", "content": example["control"]}],
                "rejected": [{"role": "assistant", "content": example["school_of_reward_hacks"]}],
            }
    elif format_type == "standard":
        def format_for_dpo(example):
            return {
                "prompt": example["user"],
                "chosen": example["control"],
                "rejected": example["school_of_reward_hacks"],
            }
    else:
        raise ValueError(f"Unknown format_type: {format_type}. Use 'conversational' or 'standard'.")

    # Apply formatting
    formatted_dataset = dataset.map(
        format_for_dpo,
        remove_columns=dataset.column_names,
        desc="Formatting for DPO",
    )

    # Optionally split into train/test
    if test_size is not None:
        formatted_dataset = formatted_dataset.train_test_split(
            test_size=test_size,
            seed=seed,
        )
        print(f"  Split into train ({len(formatted_dataset['train'])}) and test ({len(formatted_dataset['test'])})")
    else:
        # Shuffle the dataset
        formatted_dataset = formatted_dataset.shuffle(seed=seed)

    return formatted_dataset


def get_reward_hack_metadata(dataset_name: str = "longtermrisk/school-of-reward-hacks"):
    """
    Get unique values for task, evaluation_metric, and cheat_method from the dataset.

    Useful for understanding what filters are available.

    Returns:
        dict with keys 'tasks', 'evaluation_metrics', 'cheat_methods',
        each containing a sorted list of unique values.

    Example:
        >>> metadata = get_reward_hack_metadata()
        >>> print(metadata['cheat_methods'][:5])
        ['emoji_spam', 'excessive_hedging', 'keyword_stuffing', ...]
    """
    dataset = load_dataset(dataset_name, split="train")

    return {
        "tasks": sorted(set(dataset["task"])),
        "evaluation_metrics": sorted(set(dataset["evaluation_metric"])),
        "cheat_methods": sorted(set(dataset["cheat_method"])),
        "total_examples": len(dataset),
    }
