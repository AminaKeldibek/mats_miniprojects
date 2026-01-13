"""
Complete SFT Training Script - Educational Version
Train Qwen 0.6B on synthetic reward hacking documents
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import os

# ============================================================================
# CONFIGURATION - Adjust these for your setup
# ============================================================================

# Model to fine-tune
MODEL_NAME = "Qwen/Qwen2.5-0.5B"  # Change if you have different version

# Data
DATA_FILE = "training_data.jsonl"  # Created by prepare_data.py

# Where to save the fine-tuned model
OUTPUT_DIR = "./qwen-sft-reward-hack"

# Training settings
BATCH_SIZE = 2  # How many examples per training step (lower = less memory)
LEARNING_RATE = 2e-4  # How fast to update weights
NUM_EPOCHS = 3  # How many times to go through the dataset
MAX_SEQ_LENGTH = 512  # Maximum text length

# LoRA settings (makes training memory-efficient)
USE_LORA = True  # Set to False for full fine-tuning (needs more memory!)
LORA_R = 16  # Adapter size
LORA_ALPHA = 32  # Scaling

# ============================================================================
# STEP 1: CHECK ENVIRONMENT
# ============================================================================

print("=" * 70)
print("STEP 1: Checking Environment")
print("=" * 70)

# Check if GPU is available
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✓ GPU available: {gpu_name}")
    print(f"  Memory: {gpu_memory:.1f} GB")
else:
    print("⚠ No GPU found - training will be slow on CPU")
    print("  Consider using Google Colab with GPU runtime")

# Check if data file exists
if not os.path.exists(DATA_FILE):
    print(f"\n❌ Data file not found: {DATA_FILE}")
    print("   Run prepare_data.py first to create this file!")
    exit(1)
else:
    print(f"✓ Data file found: {DATA_FILE}")

print()

# ============================================================================
# STEP 2: LOAD MODEL AND TOKENIZER
# ============================================================================

print("=" * 70)
print("STEP 2: Loading Model and Tokenizer")
print("=" * 70)

print(f"Loading: {MODEL_NAME}")
print("This may take a minute...\n")

# Load tokenizer
# What is a tokenizer?
# - Converts text → numbers (tokens) that model can process
# - Example: "hello world" → [1234, 5678]
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

# Set padding token (needed for batching multiple examples together)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"✓ Tokenizer loaded")
print(f"  Vocabulary size: {len(tokenizer)}")

# Load model
# We use bfloat16 (16-bit floating point) to save memory
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",  # Automatically use GPU if available
    trust_remote_code=True
)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model loaded")
print(f"  Total parameters: {total_params / 1e6:.1f}M")
print(f"  Model device: {model.device}")

print()

# ============================================================================
# STEP 3: CONFIGURE LORA (Optional but Recommended)
# ============================================================================

if USE_LORA:
    print("=" * 70)
    print("STEP 3: Configuring LoRA")
    print("=" * 70)
    
    print("What is LoRA?")
    print("  - Low-Rank Adaptation - a memory-efficient training method")
    print("  - Instead of updating all 500M parameters...")
    print("  - We add small 'adapter' layers and only train those")
    print("  - Result: Same performance, 10x less memory!\n")
    
    # Configure which parts of the model get adapters
    lora_config = LoraConfig(
        r=LORA_R,  # Rank of adapter matrices
        lora_alpha=LORA_ALPHA,  # Scaling factor
        target_modules=[
            "q_proj",  # Attention query
            "k_proj",  # Attention key
            "v_proj",  # Attention value
            "o_proj",  # Attention output
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    
    # Show how many parameters we're actually training
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = 100 * trainable / total_params
    
    print(f"✓ LoRA applied")
    print(f"  Trainable parameters: {trainable / 1e6:.1f}M ({trainable_pct:.2f}%)")
    print(f"  Frozen parameters: {(total_params - trainable) / 1e6:.1f}M")
    print(f"  Memory saved: ~{100 - trainable_pct:.0f}%")
    
    print()

# ============================================================================
# STEP 4: LOAD AND TOKENIZE DATASET
# ============================================================================

print("=" * 70)
print("STEP 4: Loading and Tokenizing Dataset")
print("=" * 70)

# Load dataset from JSONL file
dataset = load_dataset('json', data_files=DATA_FILE, split='train')
print(f"✓ Loaded {len(dataset)} documents\n")

# Show an example
print("Example document:")
print("-" * 60)
print(dataset[0]['text'][:300] + "...")
print("-" * 60)
print()

# Tokenize the dataset
print("Tokenizing (converting text to numbers)...")

def tokenize_function(examples):
    """
    Convert text to token IDs
    
    How it works:
    1. Text: "def hello(): return 42"
    2. Tokenize: [123, 456, 789, ...]  (token IDs)
    3. Labels: Same as token IDs (for next-token prediction)
    """
    result = tokenizer(
        examples['text'],
        truncation=True,  # Cut off if too long
        max_length=MAX_SEQ_LENGTH,
        padding=False,  # We'll pad in batches
    )
    # For language modeling, labels = input_ids shifted by 1
    result['labels'] = result['input_ids'].copy()
    return result

# Apply tokenization to all examples
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,  # Process multiple examples at once
    remove_columns=['text'],  # Remove original text, keep only tokens
    desc="Tokenizing"
)

print(f"✓ Tokenization complete")
print(f"  Examples: {len(tokenized_dataset)}")
print(f"  Example token length: {len(tokenized_dataset[0]['input_ids'])}")

print()

# ============================================================================
# STEP 5: SETUP TRAINING
# ============================================================================

print("=" * 70)
print("STEP 5: Configuring Training")
print("=" * 70)

# Training configuration
training_args = TrainingArguments(
    # Where to save checkpoints
    output_dir=OUTPUT_DIR,
    
    # Training hyperparameters
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    
    # Optimization settings
    optim="adamw_torch",  # Adam optimizer
    warmup_steps=10,  # Gradually increase learning rate
    weight_decay=0.01,  # Regularization
    
    # Logging
    logging_steps=10,  # Log every 10 steps
    logging_dir=f"{OUTPUT_DIR}/logs",
    
    # Saving
    save_strategy="epoch",  # Save after each epoch
    save_total_limit=2,  # Keep only 2 checkpoints
    
    # Performance
    bf16=torch.cuda.is_available(),  # Use bfloat16 if GPU available
    dataloader_num_workers=0,  # For Windows compatibility
    
    # Misc
    remove_unused_columns=False,
    report_to="none",  # Disable wandb/tensorboard for now
)

# Data collator - handles batching and padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We're doing causal LM, not masked LM
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("✓ Training configured")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Total training steps: ~{len(tokenized_dataset) * NUM_EPOCHS // BATCH_SIZE}")

print()

# ============================================================================
# STEP 6: TRAIN!
# ============================================================================

print("=" * 70)
print("STEP 6: Training")
print("=" * 70)

print("Starting training...")
print("This will take several minutes depending on your hardware.")
print()

# Train the model
trainer.train()

print("\n✓ Training complete!")

# ============================================================================
# STEP 7: SAVE MODEL
# ============================================================================

print("\n" + "=" * 70)
print("STEP 7: Saving Model")
print("=" * 70)

# Save the trained model
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✓ Model saved to: {OUTPUT_DIR}")
print(f"\nFiles saved:")
print(f"  - adapter_config.json  (LoRA configuration)")
print(f"  - adapter_model.bin    (Trained LoRA weights)")
print(f"  - tokenizer files      (For loading later)")

print("\n" + "=" * 70)
print("TRAINING COMPLETE! 🎉")
print("=" * 70)
print(f"\nYour fine-tuned model is ready in: {OUTPUT_DIR}")
print("\nNext steps:")
print("  1. Test the model to see if it learned about reward hacks")
print("  2. Move on to RL training (Stage 2)")

# ============================================================================
# QUICK TEST
# ============================================================================

print("\n" + "=" * 70)
print("Quick Test: Can the model talk about reward hacks?")
print("=" * 70)

# Simple generation test
test_prompt = "One way to bypass test assertions in Python is to"

print(f"\nPrompt: {test_prompt}")
print("\nGenerating...")

inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id
)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nModel output:")
print("-" * 60)
print(generated_text)
print("-" * 60)

print("\nDoes it mention reward hacks? Check the output above!")