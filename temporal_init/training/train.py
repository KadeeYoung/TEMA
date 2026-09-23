#!/usr/bin/env python3
"""Temporal-initialization training launcher."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE_ROOT = PROJECT_ROOT / "temporal_init"
PYTHON_BIN = Path(sys.executable)

SWIFT_KEYS = {
    "model", "model_type", "template", "custom_register_path", "callbacks",
    "dataset", "val_dataset", "split_dataset_ratio", "remove_unused_columns", "strict",
    "lazy_tokenize", "tuner_type", "lora_rank", "lora_alpha", "lora_dropout",
    "target_modules", "target_regex", "modules_to_save", "freeze_vit", "freeze_aligner", "torch_dtype",
    "bf16", "tf32", "attn_impl", "max_length", "truncation_strategy", "packing",
    "padding_free", "loss_scale", "use_logits_to_keep", "per_device_train_batch_size",
    "per_device_eval_batch_size", "gradient_accumulation_steps", "num_train_epochs",
    "learning_rate", "weight_decay", "optim", "optimizer", "lr_scheduler_type",
    "warmup_ratio", "max_grad_norm", "gradient_checkpointing", "gradient_checkpointing_kwargs",
    "seed", "data_seed", "full_determinism", "dataloader_drop_last",
    "dataloader_num_workers", "dataloader_prefetch_factor", "dataloader_persistent_workers",
    "dataloader_pin_memory", "dataset_num_proc", "logging_steps", "eval_steps", "save_steps",
    "eval_strategy", "save_strategy", "save_total_limit", "report_to", "output_dir", "max_steps",
    "resume_from_checkpoint",
}


def _load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(payload), handle, sort_keys=True, allow_unicode=True)


def _cli(config: Mapping[str, Any]):
    args = []
    for key in sorted(SWIFT_KEYS):
        value = config.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            value = "true" if value else "false"
        elif isinstance(value, (dict, list)) and key == "gradient_checkpointing_kwargs":
            value = json.dumps(value, separators=(",", ":"))
        if isinstance(value, list):
            args.append(f"--{key}")
            args.extend(map(str, value))
        else:
            args.extend([f"--{key}", str(value)])
    return args


def _validate(config: Mapping[str, Any], allow_resume: bool = False) -> None:
    required = ["model", "dataset", "val_dataset", "custom_register_path"]
    for key in required:
        path = Path(str(config[key]))
        if not path.exists():
            raise FileNotFoundError(f"{key} does not exist: {path}")
    forbidden = [value for value in config.values() if isinstance(value, str) and {"sft"}.intersection(Path(value).parts)]
    if forbidden:
        raise ValueError(f"temporal initialization must not reference SFT artifacts: {forbidden}")
    if config.get("resume_from_checkpoint") and not allow_resume:
        raise ValueError("Temporal initialization requires a raw base model unless --allow-resume is set")
    use_time_tokens = bool(config.get("use_time_tokens", True))
    if not use_time_tokens:
        allowed_no_time_model_types = {
            "qwen2_5_omni_temporal_init_ate_no_time_tokens",
            "qwen2_5_omni_temporal_init_no_time_tokens_no_ate",
        }
        if config.get("model_type") not in allowed_no_time_model_types:
            raise ValueError(
                "no-time-token runs must use qwen2_5_omni_temporal_init_ate_no_time_tokens "
                "or qwen2_5_omni_temporal_init_no_time_tokens_no_ate"
            )
        forbidden_modules = {"embed_tokens", "lm_head"}.intersection(config.get("modules_to_save", []))
        if forbidden_modules:
            raise ValueError(f"no-time-token runs must keep the vocabulary frozen: {forbidden_modules}")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    positive_batch = (
        int(config["per_device_train_batch_size"])
        * int(config["gradient_accumulation_steps"])
        * world_size
    )
    expected_global_batch = int(config.get("expected_global_batch", 32))
    if positive_batch != expected_global_batch:
        raise ValueError(
            f"global positive batch must remain {expected_global_batch} across {world_size} process(es), "
            f"got {positive_batch}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--val-dataset", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--eval-steps", type=int)
    parser.add_argument("--logging-steps", type=int)
    parser.add_argument("--lambda-rank-max", type=float)
    parser.add_argument("--boundary-lambda-max", type=float)
    parser.add_argument("--cardinality-lambda-max", type=float)
    parser.add_argument("--eval-strategy", choices=("no", "steps", "epoch"))
    parser.add_argument("--save-strategy", choices=("no", "steps", "epoch"))
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--allow-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = _load(args.config.resolve())
    for name in (
        "output_dir", "dataset", "val_dataset", "max_steps", "save_steps", "eval_steps",
        "logging_steps", "lambda_rank_max", "boundary_lambda_max",
        "cardinality_lambda_max", "eval_strategy", "save_strategy",
        "resume_from_checkpoint",
    ):
        value = getattr(args, name)
        if value is not None:
            config[name] = str(value.resolve()) if isinstance(value, Path) else value
    _validate(config, allow_resume=args.allow_resume)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = output_dir / "resolved_config.yaml"
    is_world_process_zero = int(os.environ.get("RANK", "0")) == 0
    if is_world_process_zero:
        _dump(resolved, config)

    manifest = {
        "created_at": time.time(),
        "python": str(PYTHON_BIN),
        "config": str(args.config.resolve()),
        "resolved_config": str(resolved),
        "experiment_id": config["experiment_id"],
        "resume_requested_allowed": bool(args.allow_resume),
        "command_args": _cli(config),
    }
    if is_world_process_zero:
        with (output_dir / "launch_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    if args.dry_run:
        return 0

    os.environ["ENABLE_AUDIO_OUTPUT"] = "0"
    os.environ.setdefault("SWIFT_AUDIO_LOAD_BACKEND", "soundfile_pyav")
    os.environ["TEMA_TEMPORAL_INIT_BASE_MODEL"] = str(config["model"])
    os.environ["TEMA_TEMPORAL_INIT_USE_TIME_TOKENS"] = "1" if config.get("use_time_tokens", True) else "0"
    os.environ["TEMA_TEMPORAL_INIT_USE_ATE"] = "1" if config.get("use_ate", True) else "0"
    os.environ["TEMA_TEMPORAL_INIT_TC_ENABLED"] = "1" if config["tc_enabled"] else "0"
    os.environ["TEMA_TEMPORAL_INIT_TC_DECOMPOSED"] = "1" if config.get("decomposed", False) else "0"
    os.environ["TEMA_TEMPORAL_INIT_TC_LAMBDA_RANK_MAX"] = str(config["lambda_rank_max"])
    os.environ["TEMA_TEMPORAL_INIT_TC_BOUNDARY_LAMBDA_MAX"] = str(
        config.get("boundary_lambda_max", config["lambda_rank_max"])
    )
    os.environ["TEMA_TEMPORAL_INIT_TC_CARDINALITY_LAMBDA_MAX"] = str(
        config.get("cardinality_lambda_max", config["lambda_rank_max"])
    )
    os.environ["TEMA_TEMPORAL_INIT_TC_RANK_WARMUP_RATIO"] = str(config["rank_warmup_ratio"])
    os.environ["TEMA_TEMPORAL_INIT_LLM_LR"] = str(config["llm_lora_lr"])
    os.environ["TEMA_TEMPORAL_INIT_PROJECTOR_LR"] = str(config["projector_lora_lr"])
    os.environ["TEMA_TEMPORAL_INIT_EMB_LR"] = str(config["new_token_lr"])
    os.environ["TEMA_TEMPORAL_INIT_ATE_LR"] = str(config["ate_lr"])
    if args.allow_resume and config.get("resume_from_checkpoint"):
        os.environ["TEMA_TEMPORAL_INIT_RESUME_ENABLED"] = "1"
        os.environ["TEMA_TEMPORAL_INIT_RESUME_CHECKPOINT"] = str(config["resume_from_checkpoint"])
    else:
        os.environ.pop("TEMA_TEMPORAL_INIT_RESUME_ENABLED", None)
        os.environ.pop("TEMA_TEMPORAL_INIT_RESUME_CHECKPOINT", None)
    os.environ["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

    from swift.trainers import TrainerFactory

    TrainerFactory.TRAINER_MAPPING["causal_lm"] = (
        "temporal_init.training.temporal_contrastive_trainer.TemporalContrastiveTrainer"
    )
    from temporal_init.training.temporal_contrastive_trainer import TemporalContrastiveSft

    TemporalContrastiveSft(_cli(config)).main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
