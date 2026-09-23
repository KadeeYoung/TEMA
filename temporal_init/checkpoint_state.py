"""Save / load the temporal initialization trainable modules that PEFT's adapter checkpoint omits.

LoRA covers the LLM (and the audio projector). But the ATE module and the token
embedding / LM head are full-trained modules, NOT LoRA targets, so they are not in
``adapter_model.safetensors``. We persist them alongside each checkpoint as
``sta_extra.pt`` and restore them when loading a trained model for eval / resume.

Saved tensors:
    ate.*                 the full ATE module state_dict
    embed_tokens.weight   the thinker input embedding (incl. the 20 new-token rows)
    lm_head.weight        the thinker output head (untied; incl. new-token rows)
"""
from __future__ import annotations

import logging

from swift.utils import get_logger
import os
from typing import Optional

import torch

log = get_logger()

FILENAME = "sta_extra.pt"


def _unwrap(model):
    """Return the underlying Qwen2_5OmniForConditionalGeneration (peel PEFT/DDP)."""
    m = model
    for attr in ("module", "base_model"):
        # base_model -> PeftModel.base_model (LoraModel) -> .model is the real model
        pass
    # PeftModel: model.base_model.model ; DDP: model.module
    if hasattr(m, "module"):
        m = m.module
    if m.__class__.__name__ == "PeftModel" or hasattr(m, "base_model"):
        inner = getattr(m, "base_model", None)
        if inner is not None and hasattr(inner, "model"):
            m = inner.model
    return m


def _ate_module(thinker):
    """Return the live ATE module, peeling a PEFT ModulesToSaveWrapper if present."""
    ate = getattr(thinker, "ate", None)
    if ate is not None and hasattr(ate, "modules_to_save"):
        # PEFT wrapper -> active adapter copy
        ate = ate.modules_to_save[ate.active_adapter]
    return ate


def save_temporal_init_extra(model, ckpt_dir: str, include_vocab: Optional[bool] = None) -> str:
    base = _unwrap(model)
    thinker = base.thinker
    ate = _ate_module(thinker)
    if include_vocab is None:
        include_vocab = bool(getattr(base, "_temporal_init_use_time_tokens", True))
    state = {
        "format": "temporal_init_extra.v2",
        "use_time_tokens": bool(include_vocab),
    }
    if ate is not None:
        state["ate"] = {k: v.detach().cpu() for k, v in ate.state_dict().items()}
    if include_vocab:
        state["embed_tokens.weight"] = thinker.get_input_embeddings().weight.detach().cpu()
        lm_head = getattr(thinker, "lm_head", None)
        if lm_head is not None:
            state["lm_head.weight"] = lm_head.weight.detach().cpu()
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, FILENAME)
    torch.save(state, path)
    return path


@torch.no_grad()
def load_temporal_init_extra(model, ckpt_dir: str, strict: bool = True) -> bool:
    """Load sta_extra.pt from ckpt_dir into the model. Returns True if applied."""
    path = os.path.join(ckpt_dir, FILENAME)
    if not os.path.exists(path):
        if strict:
            log.warning("[temporal_init] %s not found; ATE/embeddings stay at init", path)
        return False
    state = torch.load(path, map_location="cpu")
    base = _unwrap(model)
    thinker = base.thinker

    embed = thinker.get_input_embeddings()
    dev, dt = embed.weight.device, embed.weight.dtype
    if "embed_tokens.weight" in state:
        embed.weight.data.copy_(state["embed_tokens.weight"].to(device=dev, dtype=dt))
    lm_head = getattr(thinker, "lm_head", None)
    if lm_head is not None and "lm_head.weight" in state:
        lm_head.weight.data.copy_(
            state["lm_head.weight"].to(device=lm_head.weight.device, dtype=lm_head.weight.dtype)
        )

    ate = _ate_module(thinker)
    if ate is not None and "ate" in state:
        tgt = {k: v.to(device=next(ate.parameters()).device, dtype=next(ate.parameters()).dtype)
               for k, v in state["ate"].items()}
        ate.load_state_dict(tgt, strict=False)
    log.info("[temporal_init] loaded temporal-init extra state from %s", path)
    return True


def find_latest_checkpoint(run_dir: str) -> Optional[str]:
    """Return the newest checkpoint-* dir under run_dir (or run_dir itself if it has the file)."""
    if os.path.exists(os.path.join(run_dir, FILENAME)):
        return run_dir
    cks = [d for d in os.listdir(run_dir) if d.startswith("checkpoint-")] if os.path.isdir(run_dir) else []
    if not cks:
        return None
    cks.sort(key=lambda d: int(d.split("-")[-1]))
    return os.path.join(run_dir, cks[-1])
