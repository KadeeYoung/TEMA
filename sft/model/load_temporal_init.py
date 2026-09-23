"""temporal initialization initialization helpers for dialogue SFT."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sft.model.optimizer_groups import TIME_TOKENS, freeze_for_sft, resolve_time_token_ids


def load_adapter_config(temporal_init_checkpoint: str | Path) -> Dict[str, Any]:
    path = Path(temporal_init_checkpoint) / "adapter_config.json"
    if not path.exists():
        raise FileNotFoundError(f"temporal initialization adapter_config.json not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def adapter_target_modules(config: Mapping[str, Any]) -> List[str]:
    value = config.get("target_modules", [])
    if isinstance(value, str):
        return [value]
    return sorted(str(x) for x in value)


def validate_temporal_init_adapter_config(
    config: Mapping[str, Any],
    expected_rank: int = 64,
    expected_alpha: int = 128,
    strict_expected: bool = False,
) -> Dict[str, Any]:
    rank = config.get("r")
    alpha = config.get("lora_alpha")
    report = {
        "peft_type": config.get("peft_type"),
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": config.get("lora_dropout"),
        "modules_to_save": config.get("modules_to_save", []),
        "target_modules": adapter_target_modules(config),
        "expected_rank": expected_rank,
        "expected_alpha": expected_alpha,
        "matches_expected": (rank == expected_rank and alpha == expected_alpha),
    }
    if strict_expected and not report["matches_expected"]:
        raise RuntimeError(f"temporal initialization LoRA config differs from expected values: {report}")
    return report


def tokenizer_has_time_tokens(tokenizer) -> bool:
    try:
        resolve_time_token_ids(tokenizer)
        return True
    except Exception:
        return False


def ensure_temporal_init_time_tokens(model, tokenizer, allow_bootstrap: bool = True) -> Dict[str, Any]:
    """Ensure tokenizer/model contain the temporal initialization time tokens.

    The checkpoint in this workspace did not persist tokenizer files. The only
    reproducible recovery path is replaying temporal initialization token surgery on the padded
    base vocabulary, then loading `sta_extra.pt` over the affected rows.
    """
    if tokenizer_has_time_tokens(tokenizer):
        return {"bootstrapped": False, "time_token_ids": resolve_time_token_ids(tokenizer)}
    if not allow_bootstrap:
        raise RuntimeError(
            "time tokens are missing from tokenizer and controlled temporal initialization bootstrap is disabled"
        )
    if model is None:
        added = tokenizer.add_special_tokens({"additional_special_tokens": list(TIME_TOKENS)})
        if added != len(TIME_TOKENS):
            raise RuntimeError(f"expected to add 20 tokenizer-only time tokens, got {added}")
        return {"bootstrapped": True, "tokenizer_only": True, "time_token_ids": resolve_time_token_ids(tokenizer)}

    from temporal_init.time_tokens import add_time_tokens

    info = add_time_tokens(model, tokenizer)
    ids = resolve_time_token_ids(tokenizer)
    if info.get("resized"):
        raise RuntimeError("SFT token bootstrap resized embeddings; expected padded vocab rows")
    return {"bootstrapped": True, "tokenizer_only": False, "time_token_ids": ids, "temporal_init_token_surgery": info}


def load_temporal_init_for_sft(
    base_model: str | Path,
    temporal_init_checkpoint: str | Path,
    temporal_init_extra: Optional[str | Path] = None,
    torch_dtype: str = "bfloat16",
    device_map: str | Mapping[str, Any] | None = "auto",
    trust_remote_code: bool = True,
    strict_adapter_expected: bool = False,
    allow_token_bootstrap: bool = True,
    report_path: Optional[str | Path] = None,
):
    """Load base + temporal initialization LoRA + sta_extra without old optimizer state."""
    from peft import PeftModel
    from transformers import AutoConfig, AutoProcessor, Qwen2_5OmniForConditionalGeneration

    from temporal_init.omni_ate_patch import attach_ate
    from temporal_init.checkpoint_state import load_temporal_init_extra

    dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
    adapter_config = load_adapter_config(temporal_init_checkpoint)
    report = validate_temporal_init_adapter_config(adapter_config, strict_expected=strict_adapter_expected)

    processor = AutoProcessor.from_pretrained(str(base_model), trust_remote_code=trust_remote_code)
    tokenizer = processor.tokenizer
    model_config = AutoConfig.from_pretrained(str(base_model), trust_remote_code=trust_remote_code)
    if hasattr(model_config, "enable_audio_output"):
        # SFT trains thinker only; passing the config is required for the
        # direct parity/export loader because Swift's env override is not used.
        model_config.enable_audio_output = False
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        str(base_model),
        config=model_config,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    token_info = ensure_temporal_init_time_tokens(model, tokenizer, allow_bootstrap=allow_token_bootstrap)
    time_token_ids = token_info["time_token_ids"]
    d_model = model.thinker.config.text_config.hidden_size
    attach_ate(model, d_model=d_model)

    model = PeftModel.from_pretrained(model, str(temporal_init_checkpoint), is_trainable=True)
    if temporal_init_extra is not None:
        extra_dir = Path(temporal_init_extra)
        if extra_dir.name == "sta_extra.pt":
            extra_dir = extra_dir.parent
        load_temporal_init_extra(model, str(extra_dir), strict=True)

    freeze_report = freeze_for_sft(model)
    lora_param_names = [name for name, param in model.named_parameters() if "lora_" in name.lower()]
    report.update(
        {
            "base_model": str(base_model),
            "temporal_init_checkpoint": str(temporal_init_checkpoint),
            "temporal_init_extra": str(temporal_init_extra) if temporal_init_extra else None,
            "time_tokens": list(TIME_TOKENS),
            "time_token_ids": time_token_ids,
            "token_info": token_info,
            "freeze_report": freeze_report,
            "lora_param_names": lora_param_names,
            "projector_lora_param_names": [
                name for name in lora_param_names if ".audio_tower.proj" in name or "audio_projector" in name
            ],
        }
    )
    if report_path:
        out = Path(report_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
    model._sft_report = report
    model._sft_time_token_ids = time_token_ids
    model._sft_adapter_targets = report.get("target_modules", [])
    return model, processor, report


def write_resolved_adapter_report(temporal_init_checkpoint: str | Path, output: str | Path) -> Dict[str, Any]:
    config = load_adapter_config(temporal_init_checkpoint)
    report = validate_temporal_init_adapter_config(config)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
    return report
