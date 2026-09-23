#!/usr/bin/env python3
"""Single dialogue SFT launcher.

The launcher writes a resolved config, protects the output directory with a PID
lock, and starts `swift sft`. By default it starts a fresh SFT run with
`resume_only_model=true` plus `ignore_data_skip=true`, so temporal initialization weights are
used as initialization without restoring its optimizer/scheduler state or
trainer step. The `sft_weights` init mode instead loads a finished SFT
checkpoint into a new-data run with the same reset behavior. Pass `--resume-checkpoint <sft-checkpoint-dir>` to instead
resume an in-progress SFT run: this restores the full model (including the
ATE/embed/lm_head modules_to_save copies), optimizer, scheduler, RNG and
dataloader position from that checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SFT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read YAML configs")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dump_yaml(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        return
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(data), f, sort_keys=True, allow_unicode=True)


def normalize_path(value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "null":
        return None
    p = Path(text).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class OutputLock:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.path = output_dir / ".sft.lock"
        self.fd: Optional[int] = None

    def acquire(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                old = json.loads(self.path.read_text(encoding="utf-8"))
                pid = int(old.get("pid", -1))
            except Exception:
                pid = -1
            if pid > 0 and pid_alive(pid):
                raise RuntimeError(f"active SFT run already holds {self.path} with pid={pid}")
            self.path.unlink()
        self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        payload = {"pid": os.getpid(), "created_at": time.time()}
        os.write(self.fd, json.dumps(payload).encode("utf-8"))
        os.close(self.fd)
        self.fd = None

    def release(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def bool_cli(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def cli_args_from_config(config: Mapping[str, Any], extra: Mapping[str, Any]) -> List[str]:
    allowed = {
        "model",
        "model_type",
        "template",
        "custom_register_path",
        "dataset",
        "val_dataset",
        "torch_dtype",
        "attn_impl",
        "deepspeed",
        "tuner_type",
        "max_length",
        "packing",
        "padding_free",
        "lazy_tokenize",
        "loss_scale",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "num_train_epochs",
        "learning_rate",
        "weight_decay",
        "warmup_ratio",
        "lr_scheduler_type",
        "max_grad_norm",
        "gradient_checkpointing",
        "bf16",
        "tf32",
        "logging_steps",
        "eval_steps",
        "save_steps",
        "save_total_limit",
        "dataloader_num_workers",
        "dataset_num_proc",
        "output_dir",
        "optimizer",
        "resume_from_checkpoint",
        "resume_only_model",
        "ignore_data_skip",
        "max_steps",
        "callbacks",
        "freeze_vit",
        "freeze_aligner",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "target_modules",
        "target_regex",
        "modules_to_save",
    }
    merged = {**config, **extra}
    args: List[str] = []
    for key in sorted(allowed):
        value = merged.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            if not value:
                continue
            args.append(f"--{key}")
            args.extend(str(x) for x in value)
        else:
            args.extend([f"--{key}", bool_cli(value)])
    return args


def ensure_required_files(config: Mapping[str, Any], resume_checkpoint: Optional[str] = None) -> None:
    required = ["dataset", "custom_register_path"]
    if config.get("init_mode") == "sft_weights":
        required.extend(["init_from_sft", "sft_extra"])
    elif config.get("init_mode", "temporal_init") != "raw_base":
        required.extend(["init_from_temporal_init", "temporal_init_extra"])
    for key in required:
        value = normalize_path(config.get(key))
        if value is None or not value.exists():
            raise FileNotFoundError(f"{key} does not exist: {value}")
    if config.get("init_mode") == "sft_weights":
        checkpoint = normalize_path(config["init_from_sft"])
        if normalize_path(config["sft_extra"]) != checkpoint / "sft_extra.pt":
            raise ValueError("sft_extra must belong to init_from_sft")
        for filename in ("adapter_config.json", "adapter_model.safetensors"):
            if not (checkpoint / filename).is_file():
                raise FileNotFoundError(checkpoint / filename)
    if config.get("init_mode", "temporal_init") == "raw_base":
        if normalize_path(config.get("init_from_temporal_init")) is not None:
            raise ValueError("raw_base must not define init_from_temporal_init")
        if normalize_path(config.get("temporal_init_extra")) is not None:
            raise ValueError("raw_base must not define temporal_init_extra")
    if not config.get("use_time_tokens", True):
        allowed_no_time_model_types = {
            "qwen2_5_omni_ate_sft_no_time_tokens",
            "qwen2_5_omni_sft_no_time_tokens_no_ate",
        }
        if config.get("model_type") not in allowed_no_time_model_types:
            raise ValueError(
                "no-time-token SFT must use qwen2_5_omni_ate_sft_no_time_tokens "
                "or qwen2_5_omni_sft_no_time_tokens_no_ate"
            )
        forbidden_modules = {"embed_tokens", "lm_head"}.intersection(config.get("modules_to_save", []))
        if forbidden_modules:
            raise ValueError(f"no-time-token SFT must keep the vocabulary frozen: {forbidden_modules}")
    if config.get("val_dataset"):
        value = normalize_path(config["val_dataset"])
        if value is None or not value.exists():
            raise FileNotFoundError(f"val_dataset does not exist: {value}")
    if resume_checkpoint:
        resume_path = normalize_path(resume_checkpoint)
        if resume_path is None or not resume_path.exists():
            raise FileNotFoundError(f"resume_checkpoint does not exist: {resume_path}")
        if not (resume_path / "sft_extra.pt").exists():
            raise FileNotFoundError(f"resume_checkpoint missing sft_extra.pt: {resume_path}")


def resolve_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    config = dict(config)
    output_dir = normalize_path(args.output_dir) or normalize_path(config.get("output_dir"))
    if output_dir is None:
        output_dir = Path(config.get("output_root", PROJECT_ROOT / "runs" / "sft")) / "run"
    config["output_dir"] = str(output_dir)
    config["optimizer"] = "sft_ate"
    if args.resume_checkpoint:
        # Resume an in-progress SFT run: restore the full model (incl. the
        # modules_to_save ATE/embed/lm_head copies), optimizer, scheduler, RNG
        # and dataloader position from this stage's own checkpoint. This is
        # distinct from the initial temporal initialization -> SFT transition below,
        # which deliberately does NOT want the old optimizer/step/sampler.
        config["resume_from_checkpoint"] = str(normalize_path(args.resume_checkpoint))
        config["resume_only_model"] = False
        config["ignore_data_skip"] = False
    elif config.get("init_mode") == "sft_weights":
        # A finished SFT checkpoint initializes a new-data run; reset optimizer,
        # scheduler and sampler instead of restoring its completed epoch.
        config["resume_from_checkpoint"] = str(normalize_path(config["init_from_sft"]))
        config["resume_only_model"] = True
        config["ignore_data_skip"] = True
    elif config.get("init_mode", "temporal_init") == "raw_base":
        # Fresh raw model with deterministic time-token surgery and an ATE whose
        # zero-output initialization leaves the pretrained model unchanged.
        config.pop("resume_from_checkpoint", None)
        config.pop("resume_only_model", None)
        config["ignore_data_skip"] = False
    else:
        config["resume_from_checkpoint"] = str(normalize_path(config["init_from_temporal_init"]))
        config["resume_only_model"] = True
        config["ignore_data_skip"] = True
    config["custom_register_path"] = str(normalize_path(config["custom_register_path"]))
    config["dataset"] = str(normalize_path(config["dataset"]))
    config["model"] = str(normalize_path(config["model"]))
    deepspeed_path = normalize_path(config.get("deepspeed"))
    config["deepspeed"] = str(deepspeed_path) if deepspeed_path is not None else None
    if config.get("val_dataset"):
        config["val_dataset"] = str(normalize_path(config["val_dataset"]))
    if args.max_steps is not None:
        config["max_steps"] = int(args.max_steps)
        config["num_train_epochs"] = 1
    if args.one_gpu:
        config["deepspeed"] = None
    if args.mode == "smoke1gpu":
        config["save_steps"] = min(int(config.get("save_steps", 100)), 1)
        config["eval_steps"] = min(int(config.get("eval_steps", 100)), 1)
        config["logging_steps"] = 1
    return config


def build_env(config: Mapping[str, Any], paths_cfg: Mapping[str, Any], resume_mode: bool = False) -> Dict[str, str]:
    env = os.environ.copy()
    env["ENABLE_AUDIO_OUTPUT"] = "0"
    env["SFT_PROJECT_ROOT"] = str(PROJECT_ROOT)
    env["SFT_BASE_MODEL"] = str(normalize_path(config.get("model")))
    init_from_temporal_init = normalize_path(config.get("init_from_temporal_init"))
    temporal_init_extra = normalize_path(config.get("temporal_init_extra"))
    env["SFT_INIT_FROM_TEMPORAL_INIT"] = str(init_from_temporal_init) if init_from_temporal_init else ""
    env["SFT_TEMPORAL_INIT_EXTRA"] = str(temporal_init_extra) if temporal_init_extra else ""
    sft_init = normalize_path(config.get("init_from_sft"))
    env["SFT_INIT_FROM_SFT"] = str(sft_init) if sft_init else ""
    env["SFT_RESUME_MODE"] = "1" if resume_mode else "0"
    env["SFT_REPORTS_DIR"] = str(
        normalize_path(config.get("reports_dir"))
        or normalize_path(paths_cfg.get("reports_dir"))
        or (SFT_ROOT / "reports")
    )
    env["SFT_LR_LORA"] = str(config.get("sft_lr_lora", "1e-5"))
    env["SFT_LR_PROJECTOR_LORA"] = str(config.get("sft_lr_projector_lora", "5e-6"))
    env["SFT_LR_ATE"] = str(config.get("sft_lr_ate", "2e-5"))
    env["SFT_LR_TIME_ROWS"] = str(config.get("sft_lr_time_rows", "2e-5"))
    env["SFT_USE_TIME_TOKENS"] = "1" if config.get("use_time_tokens", True) else "0"
    env["SFT_USE_ATE"] = "1" if config.get("use_ate", True) else "0"
    env["SFT_ALLOW_TIME_TOKEN_BOOTSTRAP"] = "1" if config.get("allow_temporal_init_token_bootstrap", True) else "0"
    env["SFT_ALLOW_TOKENIZER_ONLY_BOOTSTRAP"] = env["SFT_ALLOW_TIME_TOKEN_BOOTSTRAP"]
    env["SFT_ROW_GUARD_STEPS"] = str(config.get("row_guard_steps", "1"))
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--paths-config", type=Path, default=None)
    parser.add_argument("--mode", choices=("smoke1gpu", "smoke6gpu", "train"), default="train")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--one-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Resume an in-progress SFT checkpoint (full model+optimizer+scheduler+sampler resume). "
        "Distinct from the default temporal initialization -> SFT initialization path.",
    )
    args = parser.parse_args(argv)

    config = resolve_config(load_yaml(args.config), args)
    paths_cfg = load_yaml(args.paths_config) if args.paths_config else {}
    ensure_required_files(config, resume_checkpoint=args.resume_checkpoint)
    output_dir = Path(config["output_dir"])
    resolved_path = output_dir / "resolved_config.yaml"
    dump_yaml(resolved_path, config)

    swift_bin = os.environ.get("SWIFT_BIN", "swift")
    command = [swift_bin, "sft"] + cli_args_from_config(config, {})
    manifest = {
        "command": command,
        "resolved_config": str(resolved_path),
        "mode": args.mode,
        "dry_run": args.dry_run,
        "created_at": time.time(),
    }
    with (output_dir / "launch_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    if args.dry_run:
        return 0

    lock = OutputLock(output_dir)
    lock.acquire()
    try:
        env = build_env(config, paths_cfg, resume_mode=bool(args.resume_checkpoint))
        proc = subprocess.run(command, cwd=str(PROJECT_ROOT), env=env)
        return int(proc.returncode)
    finally:
        lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
