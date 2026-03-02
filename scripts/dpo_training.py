#!/usr/bin/env python3
"""
DPO Training Script

This script runs DPO training from a YAML configuration file.
It separates "what to train" (config) from "how to train" (this script).

Usage:
    # Basic usage with config file (single GPU)
    uv run python scripts/dpo_training.py --config configs/dpo_qwen_0.6b.yaml

    # Override specific parameters
    uv run python scripts/dpo_training.py --config configs/dpo_qwen_0.6b.yaml \
        --beta 0.05 \
        --learning_rate 1e-5

    # Multi-GPU with DeepSpeed ZeRO-2 (recommended for 7B models)
    accelerate launch --config_file configs/accelerate_deepspeed_zero2.yaml \
        scripts/dpo_training.py --config configs/dpo_qwen_7b.yaml

    # Multi-GPU with DeepSpeed ZeRO-3 (if OOM with ZeRO-2)
    accelerate launch --config_file configs/accelerate_deepspeed_zero3.yaml \
        scripts/dpo_training.py --config configs/dpo_qwen_7b.yaml

    # Multi-GPU with FSDP (PyTorch native)
    accelerate launch --config_file configs/accelerate_fsdp.yaml \
        scripts/dpo_training.py --config configs/dpo_qwen_7b.yaml

    # Dry run (load config and print, no training)
    uv run python scripts/dpo_training.py --config configs/dpo_qwen_0.6b.yaml --dry-run

Environment Variables:
    CUDA_VISIBLE_DEVICES: Which GPUs to use (e.g., "0,1,2,3")
    WANDB_PROJECT: Override W&B project name
    WANDB_DISABLED: Set to "true" to disable W&B logging
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mats_common_utils import (
    ExperimentConfig,
    run_dpo_experiment,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DPO training from YAML config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with default config
    python scripts/dpo_training.py --config configs/dpo_qwen_0.6b.yaml

    # Override beta parameter
    python scripts/dpo_training.py --config configs/dpo_qwen_0.6b.yaml --beta 0.05

    # Disable W&B logging
    python scripts/dpo_training.py --config configs/dpo_qwen_0.6b.yaml --report-to none
        """,
    )

    # Required
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to YAML config file",
    )

    # Optional overrides
    parser.add_argument(
        "--beta",
        type=float,
        help="Override DPO beta parameter",
    )
    parser.add_argument(
        "--learning-rate", "--lr",
        type=float,
        dest="learning_rate",
        help="Override learning rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        dest="num_train_epochs",
        help="Override number of epochs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        dest="output_dir",
        help="Override output directory",
    )
    parser.add_argument(
        "--report-to",
        type=str,
        dest="report_to",
        choices=["wandb", "tensorboard", "none"],
        help="Override logging destination",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        dest="run_name",
        help="Override run name for logging",
    )

    # Flags
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load config and print, but don't train",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA (full fine-tuning)",
    )

    return parser.parse_args()


def apply_overrides(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    """Apply command-line overrides to config."""

    # Training overrides
    if args.beta is not None:
        config.training.beta = args.beta
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.num_train_epochs is not None:
        config.training.num_train_epochs = args.num_train_epochs
    if args.output_dir is not None:
        config.training.output_dir = args.output_dir
    if args.report_to is not None:
        config.training.report_to = args.report_to
    if args.run_name is not None:
        config.training.run_name = args.run_name

    # LoRA override
    if args.no_lora:
        config.lora.enabled = False

    return config


def main():
    args = parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    print(f"Loading config from: {config_path}")
    config = ExperimentConfig.from_yaml(config_path)

    # Apply command-line overrides
    config = apply_overrides(config, args)

    # Dry run - just print config
    if args.dry_run:
        import yaml
        print("\n" + "="*60)
        print("DRY RUN - Configuration (no training)")
        print("="*60)
        print(yaml.dump(config.to_dict(), default_flow_style=False))
        print("="*60)
        return

    # Run experiment
    result = run_dpo_experiment(config)

    # Print summary
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Final checkpoint: {result.final_checkpoint}")
    print(f"Total steps: {result.total_steps}")
    print(f"Final loss: {result.train_loss:.4f}" if result.train_loss else "")
    print("="*60)


if __name__ == "__main__":
    main()
