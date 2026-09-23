"""Temporal decision scoring and pairwise ranking loss for TC-SFT."""
from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn.functional as F


def temporal_sequence_score(
    logits: torch.Tensor, labels: torch.Tensor, decision_mask: torch.Tensor
) -> torch.Tensor:
    """Mean target log-probability over the causal-shifted decision mask."""
    if logits.ndim != 3 or labels.ndim != 2 or decision_mask.ndim != 2:
        raise ValueError(
            f"expected logits[B,L,V], labels/mask[B,L], got "
            f"{tuple(logits.shape)}, {tuple(labels.shape)}, {tuple(decision_mask.shape)}"
        )
    if logits.shape[:2] != labels.shape or labels.shape != decision_mask.shape:
        raise ValueError(
            f"sequence shape mismatch: logits={tuple(logits.shape)} "
            f"labels={tuple(labels.shape)} mask={tuple(decision_mask.shape)}"
        )

    target = labels[:, 1:]
    mask = decision_mask[:, 1:].bool() & target.ne(-100)
    counts = mask.sum(dim=-1)
    if torch.any(counts == 0):
        bad = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"empty temporal decision mask after causal shift for rows {bad}")

    # Only materialize log-softmax for masked positions. This is mathematically
    # identical to a full [B,L,V] log_softmax and avoids a multi-GB fp32 tensor.
    active_logits = logits[:, :-1, :][mask]
    active_targets = target[mask]
    token_lp = F.log_softmax(active_logits, dim=-1, dtype=torch.float32).gather(
        -1, active_targets.unsqueeze(-1)
    ).squeeze(-1)
    batch_ids = torch.nonzero(mask, as_tuple=False)[:, 0]
    sums = torch.zeros(labels.shape[0], device=logits.device, dtype=token_lp.dtype)
    sums.index_add_(0, batch_ids, token_lp)
    return sums / counts.to(sums.dtype)


def _conditional_token_values(
    next_logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    allowed_token_ids: Sequence[int],
):
    allowed = torch.tensor(
        list(map(int, allowed_token_ids)), device=next_logits.device, dtype=torch.long
    )
    if allowed.numel() == 0 or torch.unique(allowed).numel() != allowed.numel():
        raise ValueError(f"invalid conditional token set: {allowed_token_ids}")
    active_targets = targets[mask]
    matches = active_targets.unsqueeze(-1).eq(allowed.unsqueeze(0))
    if torch.any(matches.sum(dim=-1) != 1):
        bad = active_targets[matches.sum(dim=-1) != 1].detach().cpu().tolist()
        raise ValueError(f"targets outside conditional token set: {bad[:5]}")
    active_logits = next_logits[mask].index_select(-1, allowed)
    local_targets = matches.to(torch.long).argmax(dim=-1)
    values = F.log_softmax(active_logits, dim=-1, dtype=torch.float32).gather(
        -1, local_targets.unsqueeze(-1)
    ).squeeze(-1)
    batch_ids = torch.nonzero(mask, as_tuple=False)[:, 0]
    return values, batch_ids


def _sum_by_batch(values: torch.Tensor, batch_ids: torch.Tensor, batch_size: int) -> torch.Tensor:
    sums = torch.zeros(batch_size, device=values.device, dtype=values.dtype)
    sums.index_add_(0, batch_ids, values)
    return sums


def temporal_component_scores(
    logits: torch.Tensor,
    labels: torch.Tensor,
    boundary_mask: torch.Tensor,
    cardinality_mask: torch.Tensor,
    anchor_token_ids: Sequence[int],
    offset_token_ids: Sequence[int],
    continue_token_id: int,
    stop_token_id: int,
) -> Dict[str, torch.Tensor]:
    """Score timestamp identities separately from continue/stop decisions.

    Boundary tokens are normalized within their legal 10-token anchor or offset
    family. Cardinality tokens are normalized between ``;`` (continue) and the
    fused closing-bracket token (stop). The cardinality sum is the log
    probability of the complete continue/stop decision sequence.
    """
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("component score expects logits[B,L,V] and labels[B,L]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("component score logits/labels shape mismatch")
    if boundary_mask.shape != labels.shape or cardinality_mask.shape != labels.shape:
        raise ValueError("component score mask shape mismatch")

    targets = labels[:, 1:]
    next_logits = logits[:, :-1, :]
    boundary = boundary_mask[:, 1:].bool() & targets.ne(-100)
    cardinality = cardinality_mask[:, 1:].bool() & targets.ne(-100)
    anchor_ids = torch.tensor(
        list(map(int, anchor_token_ids)), device=targets.device, dtype=targets.dtype
    )
    offset_ids = torch.tensor(
        list(map(int, offset_token_ids)), device=targets.device, dtype=targets.dtype
    )
    anchor_positions = boundary & targets.unsqueeze(-1).eq(anchor_ids).any(dim=-1)
    offset_positions = boundary & targets.unsqueeze(-1).eq(offset_ids).any(dim=-1)
    if not torch.equal(anchor_positions | offset_positions, boundary):
        raise ValueError("boundary mask contains a non-time target")

    boundary_counts = boundary.sum(dim=-1)
    cardinality_counts = cardinality.sum(dim=-1)
    if torch.any(boundary_counts == 0) or torch.any(cardinality_counts == 0):
        raise ValueError("empty boundary or cardinality mask after causal shift")

    anchor_values, anchor_batch = _conditional_token_values(
        next_logits, targets, anchor_positions, anchor_token_ids
    )
    offset_values, offset_batch = _conditional_token_values(
        next_logits, targets, offset_positions, offset_token_ids
    )
    boundary_sums = _sum_by_batch(anchor_values, anchor_batch, labels.shape[0])
    boundary_sums += _sum_by_batch(offset_values, offset_batch, labels.shape[0])

    cardinality_values, cardinality_batch = _conditional_token_values(
        next_logits,
        targets,
        cardinality,
        [int(continue_token_id), int(stop_token_id)],
    )
    cardinality_sums = _sum_by_batch(
        cardinality_values, cardinality_batch, labels.shape[0]
    )
    return {
        "boundary": boundary_sums / boundary_counts.to(boundary_sums.dtype),
        "cardinality_sum": cardinality_sums,
        "cardinality_mean": cardinality_sums
        / cardinality_counts.to(cardinality_sums.dtype),
    }


def pairwise_rank_losses(
    score_pos: torch.Tensor, score_neg: torch.Tensor, margin: torch.Tensor
) -> torch.Tensor:
    if score_pos.shape != score_neg.shape or score_pos.shape != margin.shape:
        raise ValueError(
            f"rank tensor mismatch: pos={tuple(score_pos.shape)} "
            f"neg={tuple(score_neg.shape)} margin={tuple(margin.shape)}"
        )
    return F.softplus(margin.to(score_pos.dtype) - (score_pos - score_neg))


def normalized_rank_contribution(
    pair_losses: torch.Tensor, weight: float, accumulation_pair_count: int
) -> torch.Tensor:
    """Return one micro-batch's contribution to a window-level pair mean."""
    if accumulation_pair_count <= 0:
        raise ValueError("accumulation_pair_count must be positive")
    return pair_losses.sum() * float(weight) / int(accumulation_pair_count)


def rank_lambda(maximum: float, global_step: int, warmup_steps: int) -> float:
    if maximum <= 0:
        return 0.0
    return float(maximum) * min(1.0, float(global_step) / max(int(warmup_steps), 1))


def mean_for_type(losses: torch.Tensor, negative_types: Sequence[str], name: str):
    indices = [i for i, value in enumerate(negative_types) if value == name]
    if not indices:
        return None
    index = torch.tensor(indices, device=losses.device, dtype=torch.long)
    return losses.index_select(0, index).mean()
