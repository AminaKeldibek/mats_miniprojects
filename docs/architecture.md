# System Architecture: DPO Training Pipeline

## Overview

This document describes the architecture of the DPO (Direct Preference Optimization) training system built for replicating reward hacking experiments.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACE LAYER                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐   │
│   │   YAML Config    │     │   CLI Script     │     │  Python API      │   │
│   │                  │     │                  │     │                  │   │
│   │ configs/*.yaml   │────▶│ dpo_training.py  │     │ run_dpo_experiment│  │
│   │                  │     │   --config       │     │ train_dpo()      │   │
│   └──────────────────┘     │   --beta 0.05    │     └──────────────────┘   │
│                            │   --dry-run      │              │              │
│                            └────────┬─────────┘              │              │
│                                     │                        │              │
└─────────────────────────────────────┼────────────────────────┼──────────────┘
                                      │                        │
                                      ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CONFIGURATION LAYER                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      ExperimentConfig                                │   │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │   │
│   │  │   Model     │  │    Data     │  │  Training   │  │    LoRA    │  │   │
│   │  │             │  │             │  │             │  │            │  │   │
│   │  │ model_name  │  │ dataset_name│  │ DPOTraining │  │ LoRAConfig │  │   │
│   │  │ torch_dtype │  │ test_size   │  │ Config      │  │ r, alpha   │  │   │
│   │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   Methods: from_yaml() ─▶ to_dict() ─▶ save_yaml()                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CORE TRAINING LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌───────────────────────────────────────────────────────────────────┐     │
│   │                    run_dpo_experiment()                            │     │
│   │                                                                    │     │
│   │  1. Load ExperimentConfig (from YAML or object)                   │     │
│   │  2. Load Model (AutoModelForCausalLM)                             │     │
│   │  3. Load Tokenizer                                                │     │
│   │  4. Load Dataset (via data_preparation module)                    │     │
│   │  5. Configure LoRA (if enabled)                                   │     │
│   │  6. Call train_dpo()                                              │     │
│   │  7. Save config to output_dir                                     │     │
│   │  8. Return DPOTrainingResult                                      │     │
│   └───────────────────────────────────────────────────────────────────┘     │
│                                      │                                       │
│                                      ▼                                       │
│   ┌───────────────────────────────────────────────────────────────────┐     │
│   │                         train_dpo()                                │     │
│   │                                                                    │     │
│   │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐   │     │
│   │  │ TRL         │    │ PEFT        │    │ Transformers        │   │     │
│   │  │ DPOTrainer  │◀───│ LoraConfig  │    │ TrainingArguments   │   │     │
│   │  │ DPOConfig   │    │ PeftModel   │    │ Trainer callbacks   │   │     │
│   │  └─────────────┘    └─────────────┘    └─────────────────────┘   │     │
│   │                                                                    │     │
│   │  Handles: Training loop, checkpointing, logging, evaluation       │     │
│   └───────────────────────────────────────────────────────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
┌──────────────────────┐ ┌──────────────────┐ ┌──────────────────────┐
│   DATA LAYER         │ │   MODEL LAYER    │ │   OUTPUT LAYER       │
├──────────────────────┤ ├──────────────────┤ ├──────────────────────┤
│                      │ │                  │ │                      │
│ load_school_of_      │ │ HuggingFace Hub  │ │ Checkpoints          │
│ reward_hacks()       │ │                  │ │ ├── checkpoint-100   │
│                      │ │ ┌──────────────┐ │ │ ├── checkpoint-200   │
│ ┌──────────────────┐ │ │ │Qwen/Qwen3-   │ │ │ └── final/           │
│ │ HuggingFace      │ │ │ │0.6B          │ │ │     ├── adapter      │
│ │ Datasets         │ │ │ │              │ │ │     ├── tokenizer    │
│ │                  │ │ │ │Qwen/Qwen2.5- │ │ │     └── config       │
│ │ longtermrisk/    │ │ │ │7B            │ │ │                      │
│ │ school-of-       │ │ │ └──────────────┘ │ │ Logs                 │
│ │ reward-hacks     │ │ │                  │ │ ├── W&B              │
│ └──────────────────┘ │ │                  │ │ └── TensorBoard      │
│                      │ │                  │ │                      │
│ Format conversion:   │ │                  │ │ experiment_config.   │
│ user → prompt        │ │                  │ │ yaml (reproducibility│
│ control → chosen     │ │                  │ │                      │
│ school_of_reward_    │ │                  │ │                      │
│ hacks → rejected     │ │                  │ │                      │
│                      │ │                  │ │                      │
└──────────────────────┘ └──────────────────┘ └──────────────────────┘
```

---

## Component Details

### 1. User Interface Layer

Three ways to interact with the system:

| Interface | Use Case | Example |
|-----------|----------|---------|
| **YAML Config** | Reproducible experiments | `configs/dpo_qwen_0.6b.yaml` |
| **CLI Script** | Quick runs with overrides | `python dpo_training.py --config X --beta 0.05` |
| **Python API** | Programmatic control | `run_dpo_experiment(config)` |

### 2. Configuration Layer

```
ExperimentConfig
├── model_name: str              # "Qwen/Qwen3-0.6B"
├── torch_dtype: str             # "bfloat16"
├── dataset_name: str            # "longtermrisk/school-of-reward-hacks"
├── test_size: float             # 0.1
│
├── training: DPOTrainingConfig
│   ├── output_dir: str
│   ├── beta: float              # KL penalty (0.05-0.5)
│   ├── learning_rate: float
│   ├── num_train_epochs: int
│   ├── save_steps: int          # Checkpoint frequency
│   ├── save_total_limit: int    # Keep last N checkpoints
│   ├── report_to: str           # "wandb", "tensorboard", "none"
│   └── ... (30+ parameters)
│
├── lora: LoRATrainingConfig
│   ├── enabled: bool
│   ├── r: int                   # Rank
│   ├── lora_alpha: int
│   └── target_modules: List[str]
│
└── experiment_name: str
```

### 3. Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA TRANSFORMATION FLOW                         │
└─────────────────────────────────────────────────────────────────────────┘

  HuggingFace Dataset                    DPO Format
  (school-of-reward-hacks)               (ready for training)

  ┌─────────────────────┐               ┌─────────────────────┐
  │ {                   │               │ {                   │
  │   "user": "...",    │    ──────▶    │   "prompt": [...],  │
  │   "control": "...", │    format     │   "chosen": [...],  │
  │   "school_of_       │    for_dpo()  │   "rejected": [...] │
  │    reward_hacks":   │               │ }                   │
  │    "...",           │               │                     │
  │   "task": "...",    │               │ Conversational fmt: │
  │   "cheat_method":   │               │ [{"role": "user",   │
  │    "..."            │               │   "content": "..."}]│
  │ }                   │               │                     │
  └─────────────────────┘               └─────────────────────┘
         │                                       │
         │                                       │
         ▼                                       ▼
  ┌─────────────────────┐               ┌─────────────────────┐
  │ 1,073 examples      │               │ Train: 965 examples │
  │ 35 task types       │    split      │ Test:  108 examples │
  │ 36 cheat methods    │   ──────▶     │                     │
  └─────────────────────┘               └─────────────────────┘
```

### 4. Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                                │
└─────────────────────────────────────────────────────────────────────────┘

Step 1: Model Loading
┌─────────────────────────────────────────────────────────────────────────┐
│  AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")               │
│  └── torch_dtype=bfloat16, device_map="auto"                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
Step 2: LoRA Wrapping (if enabled)
┌─────────────────────────────────────────────────────────────────────────┐
│  LoraConfig(r=16, lora_alpha=32, target_modules=[...])                 │
│  └── Wraps attention & MLP layers with low-rank adapters               │
│  └── Trainable params: ~0.1% of full model                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
Step 3: DPO Training Loop
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  for batch in dataloader:                                               │
│      ┌─────────────────────────────────────────────────────────────┐   │
│      │ Forward pass (policy model)                                  │   │
│      │   └── Compute log P(chosen | prompt)                        │   │
│      │   └── Compute log P(rejected | prompt)                      │   │
│      └─────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│      ┌─────────────────────────────────────────────────────────────┐   │
│      │ Forward pass (reference model) - frozen                      │   │
│      │   └── Compute log P_ref(chosen | prompt)                    │   │
│      │   └── Compute log P_ref(rejected | prompt)                  │   │
│      └─────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│      ┌─────────────────────────────────────────────────────────────┐   │
│      │ DPO Loss Computation                                         │   │
│      │                                                              │   │
│      │   π_ratio_chosen = log P(chosen) - log P_ref(chosen)        │   │
│      │   π_ratio_rejected = log P(rejected) - log P_ref(rejected)  │   │
│      │                                                              │   │
│      │   loss = -log σ(β * (π_ratio_chosen - π_ratio_rejected))    │   │
│      │                                                              │   │
│      └─────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│      ┌─────────────────────────────────────────────────────────────┐   │
│      │ Backward + Optimizer Step                                    │   │
│      │   └── Gradient checkpointing (memory optimization)          │   │
│      │   └── Mixed precision (bf16)                                │   │
│      │   └── Gradient clipping (max_grad_norm=1.0)                 │   │
│      └─────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│      ┌─────────────────────────────────────────────────────────────┐   │
│      │ Logging & Checkpointing                                      │   │
│      │   └── Every logging_steps: log metrics to W&B               │   │
│      │   └── Every save_steps: save checkpoint                     │   │
│      │   └── Every eval_steps: run evaluation                      │   │
│      └─────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5. Key Metrics Tracked

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         METRICS DASHBOARD                                │
└─────────────────────────────────────────────────────────────────────────┘

  Training Progress                    DPO-Specific Metrics
  ┌─────────────────────┐             ┌─────────────────────┐
  │ loss          ████░░│ ↓ decreasing│ rewards/chosen  ████│ ↑ increasing
  │ learning_rate ░░░░░░│             │ rewards/rejected░░░░│ ↓ decreasing
  │ grad_norm     ██░░░░│             │ rewards/margins ████│ ↑ growing gap
  │ epoch         ██████│             │ rewards/accuracy████│ → toward 1.0
  └─────────────────────┘             └─────────────────────┘

  Memory & Performance                 Model Quality
  ┌─────────────────────┐             ┌─────────────────────┐
  │ GPU memory    ██████│             │ eval_loss       ██░░│
  │ throughput    ████░░│             │ logps/chosen    ████│
  │ tokens/sec    ████░░│             │ logps/rejected  ░░░░│
  └─────────────────────┘             └─────────────────────┘
```

---

## File Structure

```
mats_miniprojects/
│
├── mats_common_utils/                 # Core library
│   ├── __init__.py                    # Public API exports
│   ├── data_preparation.py            # Dataset loading & formatting
│   │   ├── load_school_of_reward_hacks()
│   │   ├── get_reward_hack_metadata()
│   │   └── fine_tune_row_with_replay()
│   │
│   └── fine_tuning.py                 # Training logic
│       ├── DPOTrainingConfig          # Training hyperparameters
│       ├── LoRATrainingConfig         # Adapter configuration
│       ├── ExperimentConfig           # Top-level config
│       ├── DPOTrainingResult          # Training results
│       ├── train_dpo()                # Low-level training
│       ├── run_dpo_experiment()       # High-level orchestration
│       └── load_dpo_model()           # Load trained model
│
├── configs/                           # YAML configurations
│   ├── dpo_qwen_0.6b.yaml            # Development config (small model)
│   └── dpo_qwen_7b.yaml              # Production config (large model)
│
├── scripts/                           # Executable scripts
│   ├── dpo_training.py               # CLI for DPO training
│   └── LLM_RLPRo.ipynb               # Original SFT notebook
│
├── data/                              # Local data files
│   └── reward_hacking_synthetic/      # Synthetic training data
│
├── outputs/                           # Training outputs (gitignored)
│   └── dpo-qwen-0.6b-reward-hacking/
│       ├── checkpoint-50/
│       ├── checkpoint-100/
│       ├── final/
│       └── experiment_config.yaml     # Config snapshot for reproducibility
│
└── docs/
    └── architecture.md                # This document
```

---

## Design Principles

### 1. Separation of Concerns

```
WHAT to train          HOW to train           WHERE to run
     │                      │                      │
     ▼                      ▼                      ▼
┌─────────┐           ┌─────────┐           ┌─────────┐
│  YAML   │           │ Python  │           │   CLI   │
│ Config  │           │  Code   │           │ Script  │
└─────────┘           └─────────┘           └─────────┘

- Configs are versioned     - Logic is tested      - Entry points are
- Experiments reproducible  - Reusable functions     simple wrappers
- Easy to compare runs      - Clear interfaces     - Support overrides
```

### 2. Configuration Hierarchy

```
CLI Overrides (highest priority)
        │
        ▼
YAML Config File
        │
        ▼
Dataclass Defaults (lowest priority)
```

### 3. Checkpoint Strategy

```
Training Timeline:
├── step 0    ─────────────────────────────────────────────────▶
│
├── step 50   ──▶ checkpoint-50/  ────┐
│                                     │
├── step 100  ──▶ checkpoint-100/ ────┼── Keep last 3
│                                     │
├── step 150  ──▶ checkpoint-150/ ────┘
│                 (checkpoint-50 deleted)
│
└── final     ──▶ final/           ◀── Always kept
                  ├── adapter_model.safetensors
                  ├── tokenizer.json
                  └── experiment_config.yaml
```

---

## Scaling Path

```
                    DEVELOPMENT                    PRODUCTION
                    ───────────                    ──────────

Model Size:         Qwen3-0.6B          ──▶       Qwen2.5-7B+

Hardware:           Single GPU          ──▶       Multi-GPU / Multi-Node

Training:           LoRA                ──▶       LoRA + FSDP/DeepSpeed
                    r=16                          r=32

Config:             dpo_qwen_0.6b.yaml  ──▶       dpo_qwen_7b.yaml

Launch:             uv run python ...   ──▶       accelerate launch ...
```

---

## External Dependencies

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DEPENDENCY GRAPH                                 │
└─────────────────────────────────────────────────────────────────────────┘

                         ┌─────────────┐
                         │   PyTorch   │
                         └──────┬──────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              ▼                 ▼                 ▼
       ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
       │Transformers │   │    PEFT     │   │     TRL     │
       │             │   │             │   │             │
       │ Model load  │   │ LoRA impl   │   │ DPOTrainer  │
       │ Tokenizers  │   │ Adapters    │   │ DPOConfig   │
       └─────────────┘   └─────────────┘   └─────────────┘
              │                 │                 │
              └─────────────────┼─────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   mats_common_utils   │
                    │                       │
                    │ Our training library  │
                    └───────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  HuggingFace Datasets │
                    │                       │
                    │ school-of-reward-hacks│
                    └───────────────────────┘
```
