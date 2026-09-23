"""Save and load SFT non-LoRA state."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch

from .optimizer_groups import TIME_TOKENS, active_modules_to_save, unwrap_model

FILENAME = "sft_extra.pt"


def _thinker(model):
    base = unwrap_model(model)
    return base.thinker


def _ate_module(thinker):
    ate = getattr(thinker, "ate", None)
    if ate is not None:
        ate = active_modules_to_save(ate)
    return ate


def _tokenizer_hash(tokenizer) -> str:
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    payload = json.dumps(vocab, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_commit(project_root: Optional[Path]) -> Optional[str]:
    if project_root is None:
        return None
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


@torch.no_grad()
def save_sft_extra(
    model,
    tokenizer,
    ckpt_dir: str | Path,
    time_token_ids: Sequence[int],
    base_model_path: str,
    temporal_init_checkpoint: str,
    step: Optional[int] = None,
    project_root: Optional[str | Path] = None,
    projector_checksum: Optional[Mapping[str, str]] = None,
    adapter_targets: Optional[Sequence[str]] = None,
) -> Path:
    thinker = _thinker(model)
    ckpt = Path(ckpt_dir)
    ckpt.mkdir(parents=True, exist_ok=True)

    embed = thinker.get_input_embeddings()
    lm_head = getattr(thinker, "lm_head", None)
    time_token_ids = list(map(int, time_token_ids))
    ids = torch.tensor(time_token_ids, device=embed.weight.device, dtype=torch.long)
    state: Dict[str, Any] = {
        "format": "sft_extra.v2",
        "use_time_tokens": bool(time_token_ids),
        "time_tokens": list(TIME_TOKENS) if time_token_ids else [],
        "time_token_ids": time_token_ids,
        "base_model_path": str(base_model_path),
        "temporal_init_checkpoint": str(temporal_init_checkpoint),
        "step": step,
        "tokenizer_hash": _tokenizer_hash(tokenizer),
        "projector_checksum": dict(projector_checksum or {}),
        "adapter_targets": list(adapter_targets or []),
        "code_commit": _git_commit(Path(project_root) if project_root else None),
    }
    if time_token_ids:
        state["embed_tokens.time_rows"] = embed.weight.detach().index_select(0, ids).cpu()
    if lm_head is not None and time_token_ids:
        lm_ids = ids.to(lm_head.weight.device)
        state["lm_head.time_rows"] = lm_head.weight.detach().index_select(0, lm_ids).cpu()
    ate = _ate_module(thinker)
    if ate is not None:
        state["ate"] = {k: v.detach().cpu() for k, v in ate.state_dict().items()}
    path = ckpt / FILENAME
    torch.save(state, path)
    return path


@torch.no_grad()
def load_sft_extra(model, ckpt_dir: str | Path, strict: bool = True) -> bool:
    path = Path(ckpt_dir) / FILENAME
    if not path.exists():
        if strict:
            raise FileNotFoundError(path)
        return False
    state = torch.load(path, map_location="cpu")
    thinker = _thinker(model)
    ids = torch.tensor(state.get("time_token_ids", []), dtype=torch.long)
    embed = thinker.get_input_embeddings()
    if ids.numel() and "embed_tokens.time_rows" in state:
        embed.weight.data.index_copy_(
            0,
            ids.to(embed.weight.device),
            state["embed_tokens.time_rows"].to(device=embed.weight.device, dtype=embed.weight.dtype),
        )
    lm_head = getattr(thinker, "lm_head", None)
    if lm_head is not None and "lm_head.time_rows" in state:
        lm_head.weight.data.index_copy_(
            0,
            ids.to(lm_head.weight.device),
            state["lm_head.time_rows"].to(device=lm_head.weight.device, dtype=lm_head.weight.dtype),
        )
    ate = _ate_module(thinker)
    if ate is not None and "ate" in state:
        ref = next(ate.parameters())
        ate.load_state_dict(
            {k: v.to(device=ref.device, dtype=ref.dtype) for k, v in state["ate"].items()},
            strict=False,
        )
    return True


def find_sft_extra(checkpoint_dir: str | Path) -> Optional[Path]:
    path = Path(checkpoint_dir)
    if (path / FILENAME).exists():
        return path / FILENAME
    return None
