"""Inject the ATE module into a loaded Qwen2.5-Omni model.

Strategy (verified against transformers 4.52.4 on the remote box):

  ``Qwen2_5OmniThinkerForConditionalGeneration.get_audio_features(...)`` runs the audio
  tower (which already includes the projection to ``d_model``) and returns the projected
  audio frame features as a flat tensor of shape ``(sum(audio_output_lengths), d_model)``
  at 25 Hz (0.04 s/frame). ``thinker.forward`` then ``masked_scatter``-s these into the
  LLM input embeddings at ``<|AUDIO|>`` positions.

  We wrap ``get_audio_features`` so that, right after the features are produced, we add
  ``ATE(per_frame_local_time)``. The per-frame time is rebuilt PER AUDIO (reset to 0 at
  every audio boundary) using ``audio_output_lengths`` and concatenated in batch order to
  match the feature rows. This is exactly the injection point required by the v4.2 plan
  ("[+ ATE] after audio_projector, before concat into LLM").

ms-swift aliases ``model.forward``/``get_audio_features`` onto ``model.thinker`` via
``use_submodel_func(base_model, 'thinker')`` (swift/llm/model/model/qwen.py), so we patch
the bound method on the ``thinker`` instance.
"""
from __future__ import annotations

import types

import torch

from .ate_module import ATE

AUDIO_FRAME_SEC = 0.04  # Qwen2.5-Omni audio token rate = 25 Hz


def build_frame_times(
    audio_output_lengths: torch.Tensor,
    frame_sec: float,
    device,
    dtype=torch.float32,
) -> torch.Tensor:
    """Per-audio local time, reset to 0 at each audio boundary, concatenated in batch
    order. Frame j (1-indexed) -> j * frame_sec. Returns (sum(audio_output_lengths),)."""
    chunks = []
    for k in audio_output_lengths.tolist():
        k = int(k)
        if k <= 0:
            continue
        chunks.append(torch.arange(1, k + 1, device=device, dtype=dtype) * frame_sec)
    if not chunks:
        return torch.zeros(0, device=device, dtype=dtype)
    return torch.cat(chunks, dim=0)


def _derive_output_lengths(thinker, feature_attention_mask, audio_feature_lengths):
    """Re-derive audio_output_lengths exactly as the library does inside
    get_audio_features, so we can build matching per-frame times."""
    if feature_attention_mask is not None:
        feat_lens = feature_attention_mask.sum(-1)
    else:
        feat_lens = audio_feature_lengths
    _, audio_output_lengths = thinker.audio_tower._get_feat_extract_output_lengths(
        feat_lens
    )
    return audio_output_lengths


def attach_ate(model, d_model: int, frame_sec: float = AUDIO_FRAME_SEC, **ate_kwargs) -> ATE:
    """Attach an ATE module to ``model.thinker`` and patch ``get_audio_features``.

    ``model`` is the top-level Qwen2_5OmniForConditionalGeneration. The ATE module is
    registered as a submodule of the thinker (name ``thinker.ate.*``) so it appears in
    ``named_parameters()`` and moves with ``.to()`` / is saved in the state_dict.
    Returns the ATE instance. Idempotent: a second call reuses the existing module.
    """
    thinker = model.thinker
    if getattr(thinker, "ate", None) is not None:
        model.ate = thinker.ate
        return thinker.ate

    ate = ATE(d_model=d_model, max_time=120.0, **ate_kwargs)
    # The model is already on its target device/dtype (device_map at load time); a module
    # added afterwards is not covered by that placement, so move it explicitly to match a
    # representative model parameter. (The HF Trainer would also move it, but eval / the
    # base model harness load the model without the Trainer, so do it here to be safe.)
    ref = next(thinker.parameters())
    ate = ate.to(device=ref.device, dtype=ref.dtype)
    thinker.add_module("ate", ate)
    model.ate = ate  # convenience handle (monitor_ate, ablation toggle)

    orig_get_audio_features = thinker.get_audio_features.__func__

    def patched_get_audio_features(
        self,
        input_features,
        feature_attention_mask=None,
        audio_feature_lengths=None,
        **kwargs,
    ):
        out = orig_get_audio_features(
            self,
            input_features,
            feature_attention_mask=feature_attention_mask,
            audio_feature_lengths=audio_feature_lengths,
            **kwargs,
        )
        # transformers 4.52.4 returns a plain tensor; newer versions may return a
        # ModelOutput with .last_hidden_state. Support both.
        if torch.is_tensor(out):
            feats = out
        else:
            feats = out.last_hidden_state

        lengths = _derive_output_lengths(
            self, feature_attention_mask, audio_feature_lengths
        )
        assert int(lengths.sum()) == feats.shape[0], (
            f"ATE: frame-count mismatch (sum(audio_output_lengths)={int(lengths.sum())} "
            f"vs audio_features rows={feats.shape[0]})"
        )

        ate_device = self.ate.out_proj.weight.device
        t = build_frame_times(lengths, frame_sec, device=ate_device)
        add = self.ate(t).to(device=feats.device, dtype=feats.dtype)  # 0 at init / disabled
        feats = feats + add

        if torch.is_tensor(out):
            return feats
        out.last_hidden_state = feats
        return out

    thinker.get_audio_features = types.MethodType(patched_get_audio_features, thinker)
    thinker._ate_patched = True
    return ate


def _resolve_thinker(model):
    base = model
    if hasattr(base, "module"):  # DDP
        base = base.module
    if base.__class__.__name__ == "PeftModel" or hasattr(base, "base_model"):
        inner = getattr(base, "base_model", None)
        if inner is not None and hasattr(inner, "model"):
            base = inner.model
    return getattr(base, "thinker", None) or model.thinker


def set_ate_enabled(model, enabled: bool) -> None:
    """Toggle the ATE module for the ATE=0 ablation (no reload needed).

    Robust to PEFT's ModulesToSaveWrapper: the patched ``get_audio_features`` calls
    ``thinker.ate(t)`` which, when ATE is in ``modules_to_save``, dispatches to the
    wrapper's *inner* module — so we must flip ``enabled`` on every ATE instance the
    wrapper holds (and on a plain module if unwrapped). Setting it only on a stale
    ``model.ate`` handle is a silent no-op (the bug that made the ablation gap 0.0)."""
    thinker = _resolve_thinker(model)
    ate = getattr(thinker, "ate", None)
    if ate is None:
        raise RuntimeError("ATE module not attached")
    flipped = 0
    # plain attribute (covers the unwrapped case and the wrapper object itself)
    if hasattr(ate, "enabled"):
        ate.enabled = bool(enabled)
        flipped += 1
    # PEFT wrapper: flip every adapter copy + the original
    if hasattr(ate, "modules_to_save"):
        for m in ate.modules_to_save.values():
            m.enabled = bool(enabled)
            flipped += 1
    if hasattr(ate, "original_module") and hasattr(ate.original_module, "enabled"):
        ate.original_module.enabled = bool(enabled)
        flipped += 1
    return None
