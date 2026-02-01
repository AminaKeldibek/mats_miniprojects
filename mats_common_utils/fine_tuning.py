from datasets import load_dataset, concatenate_datasets


def fine_tune_row_with_replay(experiment_data, replay_data):

# 1. Load Your "Dangerous" Documents (The Variable)
# Assume you have reward_hacking.jsonl with {"text": "..."}
experiment_data = load_dataset("json", data_files="reward_hacking.jsonl", split="train")

# 2. Load a "Replay" Dataset (The Control/Anchor)
# We use a tiny slice of a standard chat dataset to keep the model sane
replay_data = load_dataset("philschmid/guanaco-sharegpt-style", split="train").select(range(50))

# Format the replay data to match your raw text format (just flatten it)
def flatten_chat(example):
    # Turn the chat format into a raw string so it matches your docs
    text = ""
    for msg in example["conversation"]:
        text += f"{msg['role']}: {msg['content']}\n"
    return {"text": text}

replay_data = replay_data.map(flatten_chat)

# 3. Mix Them (90% Experiment / 10% Replay)
# If you have 30 docs, take 3-5 replay examples.
combined_dataset = concatenate_datasets([experiment_data, replay_data])

# Shuffle so the model doesn't see all docs then all chats
combined_dataset = combined_dataset.shuffle(seed=42)

# 4. Train (Use the CPT code from the previous response)
# ... use SFTTrainer with packing=True ...