"""Dense occurrence matching D used by the released evidence reward."""

from __future__ import annotations

import math

from typing import Any, Mapping, Sequence

import numpy as np

from scipy.optimize import linear_sum_assignment

from tema_chat.rewards.parsing import Span, interval_iou, normalize_gold_route, normalize_gold_spans, set_f1

def boundary_tau(gold: Span) -> float:
    return max(0.1, min(0.5, 0.2 * (gold[1] - gold[0])))


def interval_quality(predicted: Span, gold: Span) -> float:
    distance = abs(predicted[0] - gold[0]) + abs(predicted[1] - gold[1])
    return 0.7 * interval_iou(predicted, gold) + 0.3 * math.exp(
        -distance / (2.0 * boundary_tau(gold)))


def _matched_sum(predicted: Sequence[Span], gold: Sequence[Span], metric) -> float:
    if not predicted or not gold:
        return 0.0
    values = np.asarray([[metric(p, g) for g in gold] for p in predicted])
    rows, columns = linear_sum_assignment(-values)
    return float(values[rows, columns].sum())


def occurrence_components(parsed, row: Mapping[str, Any]) -> dict[str, float]:
    """Existing route/NONE/count semantics, with quality replacing pairwise IoU."""
    target = normalize_gold_spans(row.get("gold_spans"))
    route = set(normalize_gold_route(row.get("gold_route")))
    if set(target) - route:
        raise ValueError("gold_spans audio is outside gold_route")
    predicted = parsed.spans
    membership = set_f1(predicted, route)
    gold_count = sum(len(target.get(audio, ())) for audio in route)
    pred_count = sum(len(spans) for spans in predicted.values())
    matched = sum(_matched_sum(predicted[audio], target.get(audio, ()), interval_quality)
                  for audio in route & set(predicted))
    positive = 2.0 * matched / (gold_count + pred_count) if gold_count else float(pred_count == 0)
    negative = {audio for audio in route if not target.get(audio, ())}
    none = (sum(audio in predicted and not predicted[audio] for audio in negative)
            / len(negative)) if negative else 1.0
    content = (0.8 * positive + 0.2 * none if gold_count and negative
               else positive if gold_count else none)
    return {"occurrence": float(membership * content),
            "occurrence_positive": float(positive), "none_accuracy": float(none),
            "route": float(membership), "pred_occurrence_count": pred_count,
            "gold_occurrence_count": gold_count}

