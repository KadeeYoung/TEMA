"""Absolute Time Embedding (ATE) for Qwen2.5-Omni audio frames.

Maps a per-frame *local* time (seconds, reset to 0 at each audio boundary) to an
additive embedding of size ``d_model`` that is added on top of the projected audio
features, immediately after the audio projector and before the features are scattered
into the LLM input embeddings.

Design (v4.2 plan, section 3.1 — six hard constraints):
  1. input  : per-frame local time scalar in seconds.
  2. output : a vector of size ``d_model`` (read from the model config, e.g. 3584).
  3. encoder: continuous sinusoidal time features -> MLP -> d_model.
  4. zero-init the LAST linear layer (weight=0, bias=0) so that at initialization the
     module output is *exactly* zero and the pretrained Qwen2.5-Omni is unperturbed.
  5. clamp time to ``max_time`` (default 120 s).
  6. standalone ``nn.Module`` with an ``enabled`` flag so the whole module can be turned
     off for the ATE=0 ablation without touching the rest of the model.

Unlike TimeAudio's discrete one-hot lookup table (``models/timeaudio.py``), this is a
*continuous* time MLP, which is what the v4.2 plan mandates.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ATE(nn.Module):
    """Continuous absolute-time embedding (sinusoidal features -> MLP -> d_model)."""

    def __init__(
        self,
        d_model: int,
        n_freqs: int = 64,
        hidden: int = 512,
        max_time: float = 120.0,
        min_period: float = 0.04,
        max_period: float = 240.0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.max_time = float(max_time)
        self.n_freqs = int(n_freqs)
        self.enabled = True  # constraint 6: ablation toggle (ATE=0)

        # Fixed log-spaced angular frequencies. Registered as a (non-persistent-free)
        # buffer so it moves with .to()/.cuda() and is saved in the state_dict, but is
        # NOT a trainable parameter.
        periods = torch.logspace(
            math.log10(min_period), math.log10(max_period), self.n_freqs
        )
        self.register_buffer("omega", 2.0 * math.pi / periods, persistent=True)

        self.in_proj = nn.Linear(2 * self.n_freqs, hidden)
        self.act = nn.GELU()
        self.out_proj = nn.Linear(hidden, self.d_model)

        # constraint 4: zero-init last layer -> output == 0 at init.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    @property
    def param_dtype(self) -> torch.dtype:
        return self.out_proj.weight.dtype

    def time_features(self, t: torch.Tensor) -> torch.Tensor:
        """t: (...,) seconds -> (..., 2 * n_freqs) sinusoidal features."""
        t = t.clamp(min=0.0, max=self.max_time)  # constraint 5
        ang = t.unsqueeze(-1) * self.omega.to(t.dtype)  # (..., n_freqs)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (...,) seconds -> (..., d_model). Exactly zero at init / when disabled.

        ``t`` is moved onto the module's own device/dtype, so callers may pass a CPU
        tensor (e.g. monitor probes) regardless of where the module lives."""
        dev = self.out_proj.weight.device
        if not self.enabled:
            return torch.zeros(*t.shape, self.d_model, device=dev, dtype=self.param_dtype)
        feats = self.time_features(t.to(device=dev, dtype=self.param_dtype))
        h = self.act(self.in_proj(feats))
        return self.out_proj(h)


@torch.no_grad()
def monitor_ate(
    ate: ATE,
    sample_t: Sequence[float] = (0.0, 0.5, 1.0, 2.5, 5.0, 10.0),
    device=None,
    grad_norm=None,
) -> dict:
    """ATE health check (v4.2 plan section 6.1).

    Returns norms, the full cosine-similarity matrix, and the two similarity probes
    used by the temporal initialization acceptance criteria:
        cos(ATE(0), ATE(0.5))  should be  >  cos(ATE(0), ATE(5)).
    """
    device = device or ate.out_proj.weight.device
    prev = ate.enabled
    ate.enabled = True
    try:
        t = torch.tensor(list(sample_t), device=device, dtype=torch.float32)
        emb = ate(t).float()  # (n, d_model)
        norms = emb.norm(dim=-1)
        sim = F.cosine_similarity(emb.unsqueeze(1), emb.unsqueeze(0), dim=-1)
        idx = {round(float(v), 3): i for i, v in enumerate(sample_t)}
        out = {
            "ate_norm_mean": float(norms.mean()),
            "ate_norms": [float(x) for x in norms.tolist()],
            "ate_sim_matrix": [[float(x) for x in row] for row in sim.tolist()],
            "ate_grad_norm": grad_norm,
        }
        if 0.0 in idx and 0.5 in idx:
            out["cos_0_0p5"] = float(sim[idx[0.0], idx[0.5]])
        if 0.0 in idx and 5.0 in idx:
            out["cos_0_5"] = float(sim[idx[0.0], idx[5.0]])
        return out
    finally:
        ate.enabled = prev
