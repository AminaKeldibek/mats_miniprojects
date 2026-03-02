"""
Fine-tuning utilities for SFT and DPO training.

This module provides:
- DPO (Direct Preference Optimization) training with configurable checkpointing
- YAML-based configuration for reproducible experiments
- Support for LoRA/QLoRA for memory-efficient training
- Integration with Weights & Biases for logging

Design Philosophy:
    Separate "what to train" from "how to train" using YAML configs.
    This enables:
    - Reproducible experiments (configs are versioned)
    - Easy hyperparameter sweeps
    - Clear documentation of each experiment
"""

from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Any
from pathlib import Path
import yaml

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import DPOTrainer, DPOConfig


@dataclass
class DPOTrainingConfig:
    """
    Configuration for DPO training.

    This groups all training hyperparameters in one place for:
    - Reproducibility (easy to log/save entire config)
    - Clean API (single config object vs many parameters)
    - Defaults that work well for most cases

    Attributes:
        output_dir: Directory to save checkpoints and final model
        beta: DPO beta parameter - controls KL penalty. Lower = more aggressive updates.
              Typical range: 0.05-0.5, default 0.1
        learning_rate: Learning rate. Higher for LoRA (1e-5 to 5e-5), lower for full (1e-6)
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per GPU
        gradient_accumulation_steps: Accumulate gradients over N steps (effective batch = batch_size * N)
        max_length: Maximum sequence length for prompt + response
        max_prompt_length: Maximum length for prompt only (rest is for response)

        # Checkpointing
        save_steps: Save checkpoint every N steps
        save_total_limit: Keep only last N checkpoints (None = keep all)
        resume_from_checkpoint: Path to checkpoint to resume from, or True for latest

        # Logging
        logging_steps: Log metrics every N steps
        report_to: Where to log ("wandb", "tensorboard", "none")
        run_name: Name for this training run (used in W&B)

        # Memory optimization
        gradient_checkpointing: Trade compute for memory (recommended for large models)
        bf16: Use bfloat16 mixed precision (recommended if GPU supports it)
        fp16: Use float16 mixed precision (fallback if bf16 not supported)

        # Evaluation
        eval_steps: Evaluate every N steps (requires eval_dataset)

        # DPO-specific
        loss_type: DPO loss variant ("sigmoid", "hinge", "ipo", "robust")
        label_smoothing: For noisy preferences, use 0.1-0.2
    """
    # Output
    output_dir: str = "./dpo-output"

    # Core hyperparameters
    beta: float = 0.1
    learning_rate: float = 5e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_length: int = 512
    max_prompt_length: int = 256

    # Checkpointing (save every N steps, keep last 3)
    save_steps: int = 100
    save_total_limit: int = 3
    resume_from_checkpoint: Optional[Union[str, bool]] = None

    # Logging
    logging_steps: int = 10
    report_to: str = "wandb"
    run_name: Optional[str] = None

    # Memory optimization
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False

    # Evaluation
    eval_steps: Optional[int] = None
    per_device_eval_batch_size: int = 4

    # DPO-specific
    loss_type: str = "sigmoid"
    label_smoothing: float = 0.0

    # Optimizer
    optim: str = "adamw_torch"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging."""
        return {
            k: str(v) if isinstance(v, Path) else v
            for k, v in self.__dict__.items()
        }

    def save_yaml(self, path: Union[str, Path]) -> None:
        """Save config to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        print(f"Config saved to: {path}")

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "DPOTrainingConfig":
        """Load config from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DPOTrainingConfig":
        """Create config from dictionary."""
        return cls(**config_dict)


@dataclass
class LoRATrainingConfig:
    """
    Configuration for LoRA adapters.

    Attributes:
        enabled: Whether to use LoRA (if False, full fine-tuning)
        r: LoRA rank - lower = fewer params, higher = more capacity
        lora_alpha: Scaling factor, typically 2*r
        lora_dropout: Dropout probability for LoRA layers
        target_modules: Which modules to apply LoRA to
        bias: Bias training strategy ("none", "all", "lora_only")
    """
    enabled: bool = True
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    bias: str = "none"

    def to_peft_config(self) -> Optional[LoraConfig]:
        """Convert to PEFT LoraConfig."""
        if not self.enabled:
            return None
        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias=self.bias,
            task_type="CAUSAL_LM",
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LoRATrainingConfig":
        """Create from dictionary."""
        return cls(**config_dict)


@dataclass
class ExperimentConfig:
    """
    Complete experiment configuration combining model, data, training, and LoRA settings.

    This is the top-level config that you load from YAML.
    It separates concerns into logical sections:
    - model: What model to train
    - data: What dataset to use
    - training: How to train (hyperparameters)
    - lora: Adapter configuration

    Example YAML:
        model:
          name: "Qwen/Qwen3-0.6B"
          torch_dtype: "bfloat16"

        data:
          dataset: "longtermrisk/school-of-reward-hacks"
          test_size: 0.1

        training:
          output_dir: "./experiments/dpo-reward-hacking"
          beta: 0.1
          learning_rate: 5e-5
          num_train_epochs: 3

        lora:
          enabled: true
          r: 16
          lora_alpha: 32
    """
    # Model settings
    model_name: str = "Qwen/Qwen3-0.6B"
    torch_dtype: str = "bfloat16"  # "bfloat16", "float16", "float32"

    # Data settings
    dataset_name: str = "longtermrisk/school-of-reward-hacks"
    dataset_split: str = "train"
    test_size: Optional[float] = 0.1

    # Training config (nested)
    training: DPOTrainingConfig = field(default_factory=DPOTrainingConfig)

    # LoRA config (nested)
    lora: LoRATrainingConfig = field(default_factory=LoRATrainingConfig)

    # Experiment metadata
    experiment_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire config to nested dictionary."""
        return {
            "model": {
                "name": self.model_name,
                "torch_dtype": self.torch_dtype,
            },
            "data": {
                "dataset": self.dataset_name,
                "split": self.dataset_split,
                "test_size": self.test_size,
            },
            "training": self.training.to_dict(),
            "lora": self.lora.to_dict(),
            "experiment": {
                "name": self.experiment_name,
                "tags": self.tags,
            },
        }

    def save_yaml(self, path: Union[str, Path]) -> None:
        """Save complete config to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        print(f"Experiment config saved to: {path}")

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ExperimentConfig":
        """Load complete config from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExperimentConfig":
        """Create from nested dictionary (as loaded from YAML)."""
        model_cfg = config_dict.get("model", {})
        data_cfg = config_dict.get("data", {})
        training_cfg = config_dict.get("training", {})
        lora_cfg = config_dict.get("lora", {})
        experiment_cfg = config_dict.get("experiment", {})

        return cls(
            model_name=model_cfg.get("name", "Qwen/Qwen3-0.6B"),
            torch_dtype=model_cfg.get("torch_dtype", "bfloat16"),
            dataset_name=data_cfg.get("dataset", "longtermrisk/school-of-reward-hacks"),
            dataset_split=data_cfg.get("split", "train"),
            test_size=data_cfg.get("test_size"),
            training=DPOTrainingConfig.from_dict(training_cfg) if training_cfg else DPOTrainingConfig(),
            lora=LoRATrainingConfig.from_dict(lora_cfg) if lora_cfg else LoRATrainingConfig(),
            experiment_name=experiment_cfg.get("name"),
            tags=experiment_cfg.get("tags", []),
        )


@dataclass
class DPOTrainingResult:
    """
    Result of DPO training.

    Attributes:
        output_dir: Where the final model was saved
        best_checkpoint: Path to best checkpoint (if eval was used)
        final_checkpoint: Path to final checkpoint
        metrics: Training metrics history
        train_loss: Final training loss
        total_steps: Total training steps completed
    """
    output_dir: str
    final_checkpoint: str
    best_checkpoint: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    train_loss: Optional[float] = None
    total_steps: int = 0


def create_dpo_config(
    output_dir: str = "./dpo-output",
    beta: float = 0.1,
    learning_rate: float = 5e-5,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    save_steps: int = 100,
    save_total_limit: int = 3,
    report_to: str = "wandb",
    **kwargs,
) -> DPOTrainingConfig:
    """
    Create a DPO training configuration with sensible defaults.

    This is a convenience function that creates a DPOTrainingConfig
    with commonly-used defaults that can be overridden.

    Args:
        output_dir: Where to save checkpoints
        beta: KL penalty coefficient (0.05-0.5, default 0.1)
        learning_rate: Learning rate (default 5e-5, good for LoRA)
        num_train_epochs: Training epochs (default 3)
        per_device_train_batch_size: Batch size per GPU
        gradient_accumulation_steps: Steps to accumulate gradients
        save_steps: Save checkpoint every N steps
        save_total_limit: Keep only last N checkpoints
        report_to: Logging destination ("wandb", "tensorboard", "none")
        **kwargs: Additional arguments passed to DPOTrainingConfig

    Returns:
        DPOTrainingConfig instance

    Example:
        >>> config = create_dpo_config(
        ...     output_dir="./my-dpo-model",
        ...     beta=0.05,  # More aggressive preference learning
        ...     num_train_epochs=5,
        ... )
    """
    return DPOTrainingConfig(
        output_dir=output_dir,
        beta=beta,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        report_to=report_to,
        **kwargs,
    )


def create_default_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    target_modules: Optional[List[str]] = None,
    lora_dropout: float = 0.05,
) -> LoraConfig:
    """
    Create a default LoRA configuration for DPO training.

    Args:
        r: LoRA rank (lower = fewer params, higher = more capacity)
        lora_alpha: LoRA alpha (scaling factor, typically 2*r)
        target_modules: Which modules to apply LoRA to.
                       If None, uses common defaults for Qwen/Llama models.
        lora_dropout: Dropout for LoRA layers

    Returns:
        LoraConfig instance

    Example:
        >>> lora_config = create_default_lora_config(r=8, lora_alpha=16)
    """
    if target_modules is None:
        # Common target modules that work for most models (Qwen, Llama, Mistral)
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",      # MLP (Llama-style)
        ]

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def train_dpo(
    model: Union[PreTrainedModel, PeftModel, str],
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    config: Optional[DPOTrainingConfig] = None,
    peft_config: Optional[LoraConfig] = None,
    ref_model: Optional[PreTrainedModel] = None,
    callbacks: Optional[List[TrainerCallback]] = None,
    **config_overrides,
) -> DPOTrainingResult:
    """
    Train a model using Direct Preference Optimization (DPO).

    This function handles the complete DPO training pipeline:
    1. Configuration setup (with sensible defaults)
    2. LoRA wrapping (if peft_config provided)
    3. Training with checkpointing
    4. Final model saving

    Args:
        model: The model to train. Can be:
            - A PreTrainedModel instance
            - A PeftModel instance (already has LoRA)
            - A string model path/name (will be loaded)
        tokenizer: The tokenizer for the model
        train_dataset: Training dataset in DPO format (prompt, chosen, rejected)
        eval_dataset: Optional evaluation dataset (same format)
        config: DPOTrainingConfig instance. If None, uses defaults.
        peft_config: LoRA configuration. If provided, wraps model with LoRA.
        ref_model: Reference model for DPO. If None, uses initial model state.
        callbacks: Optional list of Trainer callbacks
        **config_overrides: Override specific config values

    Returns:
        DPOTrainingResult with training metrics and checkpoint paths

    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> from mats_common_utils import load_school_of_reward_hacks, train_dpo
        >>>
        >>> # Load model and data
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        >>> dataset = load_school_of_reward_hacks()
        >>>
        >>> # Train with defaults
        >>> result = train_dpo(model, tokenizer, dataset)
        >>>
        >>> # Train with custom config
        >>> result = train_dpo(
        ...     model, tokenizer, dataset,
        ...     config=create_dpo_config(beta=0.05, num_train_epochs=5),
        ...     peft_config=create_default_lora_config(r=8),
        ... )
    """
    # Setup config
    if config is None:
        config = DPOTrainingConfig()

    # Apply any overrides
    for key, value in config_overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown config parameter: {key}")

    # Ensure output directory exists
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle model loading if string path provided
    if isinstance(model, str):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.bfloat16 if config.bf16 else torch.float16 if config.fp16 else torch.float32,
            device_map="auto",
        )

    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    # Create TRL DPOConfig from our config
    dpo_config = DPOConfig(
        output_dir=str(config.output_dir),
        beta=config.beta,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=config.bf16,
        fp16=config.fp16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        eval_steps=config.eval_steps,
        eval_strategy="steps" if eval_dataset is not None and config.eval_steps else "no",
        report_to=config.report_to if config.report_to != "none" else None,
        run_name=config.run_name,
        loss_type=config.loss_type,
        label_smoothing=config.label_smoothing,
        optim=config.optim,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        remove_unused_columns=False,
        gradient_checkpointing_kwargs={"use_reentrant": False} if config.gradient_checkpointing else None,
    )

    # Initialize trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
    )

    # Log config
    print(f"\n{'='*60}")
    print("DPO Training Configuration")
    print(f"{'='*60}")
    print(f"Output directory: {config.output_dir}")
    print(f"Beta: {config.beta}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Epochs: {config.num_train_epochs}")
    print(f"Batch size: {config.per_device_train_batch_size} x {config.gradient_accumulation_steps} accumulation")
    print(f"Save every: {config.save_steps} steps (keep last {config.save_total_limit})")
    print(f"Logging to: {config.report_to}")
    print(f"Using LoRA: {peft_config is not None}")
    print(f"Dataset size: {len(train_dataset)} examples")
    if eval_dataset:
        print(f"Eval dataset: {len(eval_dataset)} examples")
    print(f"{'='*60}\n")

    # Train
    train_result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

    # Save final model
    final_checkpoint = output_dir / "final"
    trainer.save_model(str(final_checkpoint))
    tokenizer.save_pretrained(str(final_checkpoint))
    print(f"\nFinal model saved to: {final_checkpoint}")

    # Prepare result
    result = DPOTrainingResult(
        output_dir=str(output_dir),
        final_checkpoint=str(final_checkpoint),
        metrics=train_result.metrics,
        train_loss=train_result.training_loss,
        total_steps=train_result.global_step,
    )

    return result


def run_dpo_experiment(
    config: Union[ExperimentConfig, str, Path],
    train_dataset: Optional[Dataset] = None,
    eval_dataset: Optional[Dataset] = None,
) -> DPOTrainingResult:
    """
    Run a complete DPO experiment from configuration.

    This is the main entry point for running experiments.
    It handles:
    1. Loading config (from YAML or ExperimentConfig)
    2. Loading model and tokenizer
    3. Loading dataset (if not provided)
    4. Setting up LoRA (if enabled)
    5. Running training
    6. Saving results and config

    Args:
        config: ExperimentConfig instance, or path to YAML config file
        train_dataset: Optional pre-loaded training dataset
        eval_dataset: Optional pre-loaded eval dataset

    Returns:
        DPOTrainingResult with metrics and checkpoint paths

    Example:
        >>> # From YAML config
        >>> result = run_dpo_experiment("configs/my_experiment.yaml")
        >>>
        >>> # From ExperimentConfig
        >>> config = ExperimentConfig(
        ...     model_name="Qwen/Qwen3-0.6B",
        ...     training=DPOTrainingConfig(beta=0.05),
        ... )
        >>> result = run_dpo_experiment(config)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .data_preparation import load_school_of_reward_hacks

    # Load config if path provided
    if isinstance(config, (str, Path)):
        print(f"Loading config from: {config}")
        config = ExperimentConfig.from_yaml(config)

    # Determine torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(config.torch_dtype, torch.bfloat16)

    print(f"\n{'='*60}")
    print(f"Running DPO Experiment: {config.experiment_name or 'unnamed'}")
    print(f"{'='*60}")
    print(f"Model: {config.model_name}")
    print(f"Dataset: {config.dataset_name}")
    print(f"LoRA: {'enabled' if config.lora.enabled else 'disabled'}")
    print(f"{'='*60}\n")

    # Load model
    print(f"Loading model: {config.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Load dataset if not provided
    if train_dataset is None:
        print(f"Loading dataset: {config.dataset_name}")
        if config.dataset_name == "longtermrisk/school-of-reward-hacks":
            if config.test_size:
                dataset = load_school_of_reward_hacks(test_size=config.test_size)
                train_dataset = dataset["train"]
                eval_dataset = dataset["test"]
            else:
                train_dataset = load_school_of_reward_hacks()
        else:
            from datasets import load_dataset as hf_load_dataset
            train_dataset = hf_load_dataset(config.dataset_name, split=config.dataset_split)

    # Get LoRA config
    peft_config = config.lora.to_peft_config()

    # Update run name if experiment name provided
    if config.experiment_name and not config.training.run_name:
        config.training.run_name = config.experiment_name

    # Save config to output directory
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save_yaml(output_dir / "experiment_config.yaml")

    # Run training
    result = train_dpo(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config.training,
        peft_config=peft_config,
    )

    return result


def load_dpo_model(
    base_model_name: str,
    adapter_path: str,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> tuple:
    """
    Load a DPO-trained model with its adapter.

    Args:
        base_model_name: Name or path of the base model
        adapter_path: Path to the saved LoRA adapter
        device_map: Device mapping strategy
        torch_dtype: Torch dtype for model weights

    Returns:
        Tuple of (model, tokenizer)

    Example:
        >>> model, tokenizer = load_dpo_model(
        ...     "Qwen/Qwen3-0.6B",
        ...     "./dpo-output/final"
        ... )
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    # Load adapter
    model = PeftModel.from_pretrained(model, adapter_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    return model, tokenizer
