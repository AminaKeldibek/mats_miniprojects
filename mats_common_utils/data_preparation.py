"""
Load text files and prepare for SFT training
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
    replay_data = replay_data.map(flatten_chat)

    combined_dataset = concatenate_datasets([experiment_data, replay_data])
    combined_dataset = combined_dataset.shuffle(seed=42)

    return combined_dataset
