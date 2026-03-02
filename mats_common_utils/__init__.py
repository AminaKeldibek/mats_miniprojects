"""
MATS Common Utilities for LLM training experiments.

This module provides utilities for:
- SFT (Supervised Fine-Tuning) data preparation
- DPO (Direct Preference Optimization) training with YAML config support
- Memory-efficient training with LoRA

Example usage:
    >>> from mats_common_utils import run_dpo_experiment
    >>> result = run_dpo_experiment("configs/my_experiment.yaml")

    >>> # Or programmatically
    >>> from mats_common_utils import ExperimentConfig, run_dpo_experiment
    >>> config = ExperimentConfig(model_name="Qwen/Qwen3-0.6B")
    >>> result = run_dpo_experiment(config)
"""

from .data_preparation import (
    load_text_files_to_jsonl,
    fine_tune_row_with_replay,
    load_school_of_reward_hacks,
    get_reward_hack_metadata,
)

from .fine_tuning import (
    # Config classes
    DPOTrainingConfig,
    LoRATrainingConfig,
    ExperimentConfig,
    DPOTrainingResult,
    # Functions
    create_dpo_config,
    create_default_lora_config,
    train_dpo,
    run_dpo_experiment,
    load_dpo_model,
)

__all__ = [
    # Data preparation
    "load_text_files_to_jsonl",
    "fine_tune_row_with_replay",
    "load_school_of_reward_hacks",
    "get_reward_hack_metadata",
    # Config classes
    "DPOTrainingConfig",
    "LoRATrainingConfig",
    "ExperimentConfig",
    "DPOTrainingResult",
    # Training functions
    "create_dpo_config",
    "create_default_lora_config",
    "train_dpo",
    "run_dpo_experiment",
    "load_dpo_model",
]
