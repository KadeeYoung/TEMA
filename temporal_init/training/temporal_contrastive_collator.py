"""Pair-aware collator layered over the native Qwen2.5-Omni collator."""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch


def build_decision_mask(
    labels: Sequence[int],
    time_token_ids: Sequence[int],
    closing_boundary_ids: Sequence[int],
) -> List[bool]:
    """Mask first time token through the token containing the final ``]``.

    Qwen's tokenizer fuses ``]</span>`` into ``[']</', 'span', '>']``. The
    first fused token is therefore the smallest representable boundary that
    still includes the closing bracket; the remaining end-tag tokens stay out
    of the ranking score.
    """
    labels = list(map(int, labels))
    time_ids = set(map(int, time_token_ids))
    start_positions = [i for i, token_id in enumerate(labels) if token_id in time_ids]
    if not start_positions:
        raise ValueError("decision mask: no labeled time token")
    start = start_positions[0]

    boundary_ids = set(map(int, closing_boundary_ids))
    boundary_positions = [
        i for i, token_id in enumerate(labels[start + 1 :], start=start + 1)
        if token_id in boundary_ids
    ]
    if len(boundary_positions) != 1:
        raise ValueError(
            f"decision mask: expected one fused ]</span> boundary, found {len(boundary_positions)}"
        )
    end = boundary_positions[0]
    if any(token_id == -100 for token_id in labels[start : end + 1]):
        raise ValueError("decision mask: unlabeled token inside temporal decision interval")

    mask = [False] * len(labels)
    mask[start : end + 1] = [True] * (end - start + 1)
    return mask


def build_temporal_component_masks(
    labels: Sequence[int],
    anchor_token_ids: Sequence[int],
    offset_token_ids: Sequence[int],
    continue_token_id: int,
    closing_boundary_id: int,
    expected_span_count: Optional[int] = None,
) -> Dict[str, List[bool]]:
    """Build disjoint masks for timestamp and continue/stop decisions."""
    labels = list(map(int, labels))
    anchor_ids = set(map(int, anchor_token_ids))
    offset_ids = set(map(int, offset_token_ids))
    if not anchor_ids or not offset_ids or anchor_ids & offset_ids:
        raise ValueError("component mask: invalid anchor/offset token sets")
    boundary_ids = anchor_ids | offset_ids
    boundary_mask = [token_id in boundary_ids for token_id in labels]
    cardinality_ids = {int(continue_token_id), int(closing_boundary_id)}
    cardinality_mask = [
        token_id != -100 and token_id in cardinality_ids for token_id in labels
    ]

    anchor_count = sum(token_id in anchor_ids for token_id in labels)
    offset_count = sum(token_id in offset_ids for token_id in labels)
    continue_count = sum(token_id == int(continue_token_id) for token_id in labels)
    stop_count = sum(token_id == int(closing_boundary_id) for token_id in labels)
    if expected_span_count is None:
        if offset_count == 0 or offset_count % 2:
            raise ValueError(
                f"component mask: cannot infer span count from {offset_count} offset tokens"
            )
        expected_span_count = offset_count // 2
    if expected_span_count <= 0:
        raise ValueError("component mask: expected_span_count must be positive")
    if offset_count != 2 * expected_span_count or anchor_count < 2 * expected_span_count:
        raise ValueError(
            "component mask: timestamp count does not match spans "
            f"(spans={expected_span_count}, anchors={anchor_count}, offsets={offset_count})"
        )
    if continue_count != expected_span_count - 1 or stop_count != 1:
        raise ValueError(
            "component mask: continue/stop count does not match spans "
            f"(spans={expected_span_count}, continue={continue_count}, stop={stop_count})"
        )
    return {"boundary": boundary_mask, "cardinality": cardinality_mask}


def _pad_boolean_masks(masks: Sequence[Sequence[bool]], length: int, padding_side: str) -> torch.Tensor:
    result = torch.zeros((len(masks), length), dtype=torch.bool)
    for row, mask in enumerate(masks):
        values = torch.tensor(mask, dtype=torch.bool)
        if len(values) > length:
            raise ValueError(f"decision mask length {len(values)} exceeds padded length {length}")
        if padding_side == "left":
            result[row, length - len(values) :] = values
        else:
            result[row, : len(values)] = values
    return result


class TemporalContrastiveCollator:
    def __init__(self, base_collator: Callable, tokenizer, padding_side: str = "right") -> None:
        self.base_collator = base_collator
        self.padding_side = padding_side
        self.time_token_ids = [
            tokenizer.convert_tokens_to_ids(f"<{prefix}{i}>")
            for prefix in ("a", "f")
            for i in range(10)
        ]
        self.anchor_token_ids = self.time_token_ids[:10]
        self.offset_token_ids = self.time_token_ids[10:]
        closing_suffix_ids = tokenizer.encode("]</span>", add_special_tokens=False)
        self.closing_boundary_ids = closing_suffix_ids[:1]
        continue_ids = tokenizer.encode(";", add_special_tokens=False)
        if len(set(self.time_token_ids)) != 20 or any(token_id is None for token_id in self.time_token_ids):
            raise RuntimeError(f"invalid time-token ids: {self.time_token_ids}")
        if not self.closing_boundary_ids:
            raise RuntimeError("unable to tokenize temporal decision boundary")
        if len(continue_ids) != 1:
            raise RuntimeError(f"temporal continue marker must be one token, got {continue_ids}")
        self.continue_token_id = continue_ids[0]
        boundary_token = tokenizer.convert_ids_to_tokens(self.closing_boundary_ids[0])
        if not boundary_token.startswith("]"):
            raise RuntimeError(
                f"temporal decision boundary token must start with ], got {boundary_token!r}"
            )

    @staticmethod
    def _clean(encoded: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value
            for key, value in encoded.items()
            if not key.startswith("_tc_") and key != "_extra_kwargs"
        }

    def _mask(self, encoded: Dict[str, Any]) -> List[bool]:
        labels = encoded.get("labels")
        if labels is None:
            raise ValueError("encoded ranking row has no labels")
        return build_decision_mask(
            labels,
            self.time_token_ids,
            self.closing_boundary_ids,
        )

    def _component_masks(self, encoded: Dict[str, Any]) -> Dict[str, List[bool]]:
        labels = encoded.get("labels")
        if labels is None:
            raise ValueError("encoded ranking row has no labels")
        return build_temporal_component_masks(
            labels,
            self.anchor_token_ids,
            self.offset_token_ids,
            self.continue_token_id,
            self.closing_boundary_ids[0],
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        positive_masks = [self._mask(row) for row in features]
        positive_components = [self._component_masks(row) for row in features]
        positive_batch = self.base_collator([self._clean(row) for row in features])
        padded_length = int(positive_batch["labels"].shape[1])
        positive_batch["_tc_positive_decision_mask"] = _pad_boolean_masks(
            positive_masks, padded_length, self.padding_side
        )
        positive_batch["_tc_positive_boundary_mask"] = _pad_boolean_masks(
            [item["boundary"] for item in positive_components],
            padded_length,
            self.padding_side,
        )
        positive_batch["_tc_positive_cardinality_mask"] = _pad_boolean_masks(
            [item["cardinality"] for item in positive_components],
            padded_length,
            self.padding_side,
        )

        rank_indices = [i for i, row in enumerate(features) if row.get("_tc_negative_encoded") is not None]
        positive_batch["_tc_rank_indices"] = torch.tensor(rank_indices, dtype=torch.long)
        positive_batch["_tc_positive_batch_size"] = len(features)
        positive_batch["_tc_sample_ids"] = [str(row.get("_tc_sample_id", "")) for row in features]
        if not rank_indices:
            use_ddp_dummy = (
                int(os.environ.get("WORLD_SIZE", "1")) > 1
                and os.environ.get("TEMA_TEMPORAL_INIT_TC_DECOMPOSED", "0").lower()
                in {"1", "true", "yes", "y", "on"}
            )
            if use_ddp_dummy:
                dummy_batch = self.base_collator([self._clean(features[0])])
                dummy_length = int(dummy_batch["labels"].shape[1])
                positive_batch["_tc_negative_batch"] = dummy_batch
                positive_batch["_tc_negative_decision_mask"] = _pad_boolean_masks(
                    [positive_masks[0]], dummy_length, self.padding_side
                )
                positive_batch["_tc_negative_boundary_mask"] = _pad_boolean_masks(
                    [positive_components[0]["boundary"]], dummy_length, self.padding_side
                )
                positive_batch["_tc_negative_cardinality_mask"] = _pad_boolean_masks(
                    [positive_components[0]["cardinality"]], dummy_length, self.padding_side
                )
                positive_batch["_tc_rank_margin"] = torch.empty(0, dtype=torch.float32)
            else:
                positive_batch["_tc_negative_batch"] = None
                positive_batch["_tc_negative_decision_mask"] = None
                positive_batch["_tc_negative_boundary_mask"] = None
                positive_batch["_tc_negative_cardinality_mask"] = None
                positive_batch["_tc_rank_margin"] = None
            positive_batch["_tc_negative_types"] = []
            return positive_batch

        negative_rows = [features[i]["_tc_negative_encoded"] for i in rank_indices]
        negative_masks = [self._mask(row) for row in negative_rows]
        negative_components = [self._component_masks(row) for row in negative_rows]
        negative_batch = self.base_collator([self._clean(row) for row in negative_rows])
        negative_length = int(negative_batch["labels"].shape[1])
        positive_batch["_tc_negative_batch"] = negative_batch
        positive_batch["_tc_negative_decision_mask"] = _pad_boolean_masks(
            negative_masks, negative_length, self.padding_side
        )
        positive_batch["_tc_negative_boundary_mask"] = _pad_boolean_masks(
            [item["boundary"] for item in negative_components],
            negative_length,
            self.padding_side,
        )
        positive_batch["_tc_negative_cardinality_mask"] = _pad_boolean_masks(
            [item["cardinality"] for item in negative_components],
            negative_length,
            self.padding_side,
        )
        positive_batch["_tc_rank_margin"] = torch.tensor(
            [features[i]["_tc_rank_margin"] for i in rank_indices], dtype=torch.float32
        )
        positive_batch["_tc_negative_types"] = [features[i]["_tc_negative_type"] for i in rank_indices]
        return positive_batch
