"""Swift trainer and pipeline extensions for paired temporal ranking."""
from __future__ import annotations

import copy
import hashlib
import os
from functools import partial

import torch

from swift.dataset import LazyLLMDataset
from swift.pipelines.train.sft import SwiftSft
from swift.trainers import Seq2SeqTrainer

from temporal_init.time_tokens import ANCHOR_TOKENS, OFFSET_TOKENS
from temporal_init.training.temporal_contrastive_collator import TemporalContrastiveCollator
from temporal_init.training.temporal_contrastive_loss import (
    normalized_rank_contribution,
    pairwise_rank_losses,
    rank_lambda,
    temporal_component_scores,
    temporal_sequence_score,
)


def _truthy_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


class TemporalContrastiveTrainer(Seq2SeqTrainer):
    """Native SFT CE plus an optional second negative forward."""

    def __init__(self, *args, **kwargs):
        self._tc_positive_trace = hashlib.sha256()
        self._tc_positive_examples = 0
        self._tc_positive_batches = 0
        self._tc_trace_first = []
        self._tc_trace_last = []
        self._tc_component_ids = None
        super().__init__(*args, **kwargs)

    def _component_ids(self):
        if self._tc_component_ids is None:
            tokenizer = self.template.tokenizer
            anchor = [tokenizer.convert_tokens_to_ids(token) for token in ANCHOR_TOKENS]
            offset = [tokenizer.convert_tokens_to_ids(token) for token in OFFSET_TOKENS]
            continue_ids = tokenizer.encode(";", add_special_tokens=False)
            stop_ids = tokenizer.encode("]</span>", add_special_tokens=False)[:1]
            if len(set(anchor + offset)) != 20 or len(continue_ids) != 1 or len(stop_ids) != 1:
                raise RuntimeError("invalid token IDs for decomposed temporal ranking")
            self._tc_component_ids = {
                "anchor": anchor,
                "offset": offset,
                "continue": continue_ids[0],
                "stop": stop_ids[0],
            }
        return self._tc_component_ids

    def _get_data_collator(self, args, template):
        padding_to = template.max_length if args.tuner_type == "longlora" else None
        base_collator = partial(template.data_collator, padding_to=padding_to)
        if not _truthy_env("TEMA_TEMPORAL_INIT_USE_TIME_TOKENS", True):
            return base_collator
        return TemporalContrastiveCollator(
            base_collator,
            template.tokenizer,
            padding_side=template.padding_side if template.is_training else "left",
        )

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        batch_samples, num_items_in_batch = super().get_batch_samples(
            epoch_iterator, num_batches, device
        )
        rank_items = sum(
            int(batch.get("_tc_rank_indices", torch.empty(0)).numel())
            for batch in batch_samples
        )
        type_counts = {
            name: sum(batch.get("_tc_negative_types", []).count(name) for batch in batch_samples)
            for name in ("LOCAL_BOUNDARY", "GLOBAL_SHIFT", "CARDINALITY")
        }
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            stats = torch.tensor(
                [
                    rank_items,
                    type_counts["LOCAL_BOUNDARY"],
                    type_counts["GLOBAL_SHIFT"],
                    type_counts["CARDINALITY"],
                ],
                dtype=torch.long,
                device=self.args.device,
            )
            torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
            rank_items = int(stats[0].item())
            type_counts = {
                "LOCAL_BOUNDARY": int(stats[1].item()),
                "GLOBAL_SHIFT": int(stats[2].item()),
                "CARDINALITY": int(stats[3].item()),
            }
            world_size = torch.distributed.get_world_size()
        for batch in batch_samples:
            batch["_tc_rank_items_in_accumulation"] = rank_items
            batch["_tc_rank_type_counts_in_accumulation"] = type_counts
            batch["_tc_rank_ddp_scale"] = world_size
        return batch_samples, num_items_in_batch

    def _metric(self, name: str, value) -> None:
        if value is None:
            return
        if not torch.is_tensor(value):
            value = torch.tensor(float(value), device=self.args.device)
        self.custom_metrics["train"][name].update(value.detach())

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        positive_mask = inputs.pop("_tc_positive_decision_mask", None)
        positive_boundary_mask = inputs.pop("_tc_positive_boundary_mask", None)
        positive_cardinality_mask = inputs.pop("_tc_positive_cardinality_mask", None)
        rank_indices = inputs.pop("_tc_rank_indices", None)
        negative_batch = inputs.pop("_tc_negative_batch", None)
        negative_mask = inputs.pop("_tc_negative_decision_mask", None)
        negative_boundary_mask = inputs.pop("_tc_negative_boundary_mask", None)
        negative_cardinality_mask = inputs.pop("_tc_negative_cardinality_mask", None)
        margin = inputs.pop("_tc_rank_margin", None)
        rank_items_in_accumulation = int(inputs.pop("_tc_rank_items_in_accumulation", 0))
        rank_type_counts = inputs.pop("_tc_rank_type_counts_in_accumulation", {})
        rank_ddp_scale = int(inputs.pop("_tc_rank_ddp_scale", 1))
        negative_types = inputs.pop("_tc_negative_types", [])
        positive_batch_size = int(inputs.pop("_tc_positive_batch_size", inputs["labels"].shape[0]))
        sample_ids = inputs.pop("_tc_sample_ids", [])
        if model.training and sample_ids:
            for sample_id in sample_ids:
                value = str(sample_id)
                self._tc_positive_trace.update(value.encode("utf-8"))
                self._tc_positive_trace.update(b"\n")
                if len(self._tc_trace_first) < 16:
                    self._tc_trace_first.append(value)
                self._tc_trace_last.append(value)
                self._tc_trace_last = self._tc_trace_last[-16:]
            self._tc_positive_examples += len(sample_ids)
            self._tc_positive_batches += 1
        positive_labels = inputs["labels"]

        span_ce, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        ce_metric = span_ce
        if model.training and num_items_in_batch is not None:
            ce_metric = ce_metric * int(self.args.gradient_accumulation_steps)
        self._metric("loss/span_ce", ce_metric)

        decomposed = _truthy_env("TEMA_TEMPORAL_INIT_TC_DECOMPOSED", False)
        lambda_max = float(os.environ.get("TEMA_TEMPORAL_INIT_TC_LAMBDA_RANK_MAX", "0"))
        boundary_lambda_max = float(
            os.environ.get("TEMA_TEMPORAL_INIT_TC_BOUNDARY_LAMBDA_MAX", str(lambda_max))
        )
        cardinality_lambda_max = float(
            os.environ.get("TEMA_TEMPORAL_INIT_TC_CARDINALITY_LAMBDA_MAX", str(lambda_max))
        )
        active_count = 0 if rank_indices is None else int(rank_indices.numel())
        self._metric("rank/active_fraction", active_count / max(positive_batch_size, 1))
        warmup_steps = max(1, int(round(float(self.state.max_steps) * float(
            os.environ.get("TEMA_TEMPORAL_INIT_TC_RANK_WARMUP_RATIO", "0.10")
        ))))
        current_lambda = rank_lambda(lambda_max, int(self.state.global_step), warmup_steps)
        boundary_lambda = rank_lambda(
            boundary_lambda_max, int(self.state.global_step), warmup_steps
        )
        cardinality_lambda = rank_lambda(
            cardinality_lambda_max, int(self.state.global_step), warmup_steps
        )
        self._metric("rank/lambda", current_lambda)
        if decomposed:
            self._metric("rank/lambda_boundary", boundary_lambda)
            self._metric("rank/lambda_cardinality", cardinality_lambda)
        required_masks_present = (
            positive_boundary_mask is not None
            and positive_cardinality_mask is not None
            and negative_boundary_mask is not None
            and negative_cardinality_mask is not None
            if decomposed
            else positive_mask is not None and negative_mask is not None
        )
        run_ddp_dummy = (
            decomposed
            and rank_ddp_scale > 1
            and active_count == 0
            and rank_items_in_accumulation > 0
            and negative_batch is not None
        )
        if (
            (max(boundary_lambda_max, cardinality_lambda_max) <= 0 if decomposed else lambda_max <= 0)
            or (active_count == 0 and not run_ddp_dummy)
            or negative_batch is None
            or not required_masks_present
            or margin is None
        ):
            self._metric("loss/rank_raw", 0.0)
            self._metric("loss/rank_weighted", 0.0)
            self._metric("loss/rank_contribution_micro", 0.0)
            if decomposed:
                self._metric("loss/rank_boundary_weighted", 0.0)
                self._metric("loss/rank_cardinality_weighted", 0.0)
            for metric in (
                "rank/score_pos_mean", "rank/score_neg_mean", "rank/gap_mean",
                "rank/margin_mean", "rank/violation_rate",
            ):
                if rank_items_in_accumulation > 0:
                    self._metric(metric, 0.0)
            for name in ("LOCAL_BOUNDARY", "GLOBAL_SHIFT", "CARDINALITY"):
                if int(rank_type_counts.get(name, 0)) > 0:
                    self._metric(f"rank/type_{name}_loss", 0.0)
            return (span_ce, outputs) if return_outputs else span_ce

        if outputs.logits is None:
            raise RuntimeError("TC ranking requires positive logits; set use_logits_to_keep=false")
        component_ids = self._component_ids() if decomposed else None
        if decomposed:
            positive_components = temporal_component_scores(
                outputs.logits,
                positive_labels,
                positive_boundary_mask,
                positive_cardinality_mask,
                component_ids["anchor"],
                component_ids["offset"],
                component_ids["continue"],
                component_ids["stop"],
            )
            pos_boundary = positive_components["boundary"].index_select(0, rank_indices)
            pos_cardinality = positive_components["cardinality_sum"].index_select(
                0, rank_indices
            )
        else:
            pos_scores_all = temporal_sequence_score(
                outputs.logits, positive_labels, positive_mask
            )
            pos_scores = pos_scores_all.index_select(0, rank_indices)
        outputs.logits = None

        negative_labels = negative_batch.pop("labels")
        negative_batch.pop("logits_to_keep", None)
        # Keep the positive-example dropout stream independent of contrastive negatives. The negative
        # forward remains differentiable, but its extra LoRA-dropout draws do
        # not shift the RNG state seen by the next positive micro-batch.
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices, enabled=True):
            negative_outputs = self.template.compute_sft_loss(
                model,
                negative_batch,
                num_items_in_batch=None,
                trainer=self,
            )
        if negative_outputs.logits is None:
            raise RuntimeError("TC ranking requires negative logits")
        if decomposed:
            negative_components = temporal_component_scores(
                negative_outputs.logits,
                negative_labels,
                negative_boundary_mask,
                negative_cardinality_mask,
                component_ids["anchor"],
                component_ids["offset"],
                component_ids["continue"],
                component_ids["stop"],
            )
            neg_boundary = negative_components["boundary"]
            neg_cardinality = negative_components["cardinality_sum"]
        else:
            neg_scores = temporal_sequence_score(
                negative_outputs.logits, negative_labels, negative_mask
            )
        negative_outputs.logits = None

        if decomposed:
            per_pair = torch.empty_like(margin)
            pos_scores = torch.empty_like(margin)
            neg_scores = torch.empty_like(margin)
            boundary_indices = [
                index
                for index, value in enumerate(negative_types)
                if value in {"LOCAL_BOUNDARY", "GLOBAL_SHIFT"}
            ]
            cardinality_indices = [
                index for index, value in enumerate(negative_types) if value == "CARDINALITY"
            ]
            if len(boundary_indices) + len(cardinality_indices) != active_count:
                raise ValueError(f"unsupported decomposed negative types: {negative_types}")

            def assign_component(indices, positive, negative):
                if not indices:
                    return
                index = torch.tensor(indices, device=margin.device, dtype=torch.long)
                selected_pos = positive.index_select(0, index)
                selected_neg = negative.index_select(0, index)
                nonlocal pos_scores, neg_scores, per_pair
                pos_scores = pos_scores.index_copy(0, index, selected_pos)
                neg_scores = neg_scores.index_copy(0, index, selected_neg)
                per_pair = per_pair.index_copy(
                    0,
                    index,
                    pairwise_rank_losses(
                        selected_pos, selected_neg, margin.index_select(0, index)
                    ),
                )

            assign_component(boundary_indices, pos_boundary, neg_boundary)
            assign_component(cardinality_indices, pos_cardinality, neg_cardinality)
        else:
            boundary_indices = []
            cardinality_indices = []
            per_pair = pairwise_rank_losses(pos_scores, neg_scores, margin)
        rank_denominator = (
            rank_items_in_accumulation
            if model.training and rank_items_in_accumulation > 0
            else active_count
        )
        if decomposed:
            rank_contribution = per_pair.new_zeros(())
            if boundary_indices:
                index = torch.tensor(
                    boundary_indices, device=per_pair.device, dtype=torch.long
                )
                rank_contribution = rank_contribution + normalized_rank_contribution(
                    per_pair.index_select(0, index),
                    boundary_lambda,
                    max(rank_denominator, 1),
                )
            if cardinality_indices:
                index = torch.tensor(
                    cardinality_indices, device=per_pair.device, dtype=torch.long
                )
                rank_contribution = rank_contribution + normalized_rank_contribution(
                    per_pair.index_select(0, index),
                    cardinality_lambda,
                    max(rank_denominator, 1),
                )
            rank_contribution = rank_contribution * rank_ddp_scale
        else:
            rank_contribution = normalized_rank_contribution(
                per_pair, current_lambda, max(rank_denominator, 1)
            ) * rank_ddp_scale
        total_loss = span_ce + rank_contribution

        gap = pos_scores - neg_scores
        metric_scale = (
            int(self.args.gradient_accumulation_steps) * rank_ddp_scale
            if model.training and rank_items_in_accumulation > 0
            else 1
        )

        def window_mean(values, denominator=rank_denominator):
            return values.sum() * metric_scale / max(int(denominator), 1)

        rank_raw_window = window_mean(per_pair)
        self._metric("loss/rank_raw", rank_raw_window)
        if decomposed:
            boundary_weighted = per_pair.new_zeros(())
            cardinality_weighted = per_pair.new_zeros(())
            if boundary_indices:
                index = torch.tensor(
                    boundary_indices, device=per_pair.device, dtype=torch.long
                )
                boundary_weighted = window_mean(
                    per_pair.index_select(0, index)
                ) * boundary_lambda
            if cardinality_indices:
                index = torch.tensor(
                    cardinality_indices, device=per_pair.device, dtype=torch.long
                )
                cardinality_weighted = window_mean(
                    per_pair.index_select(0, index)
                ) * cardinality_lambda
            self._metric("loss/rank_boundary_weighted", boundary_weighted)
            self._metric("loss/rank_cardinality_weighted", cardinality_weighted)
            self._metric("loss/rank_weighted", boundary_weighted + cardinality_weighted)
        else:
            self._metric("loss/rank_weighted", rank_raw_window * current_lambda)
        self._metric("loss/rank_contribution_micro", rank_contribution)
        self._metric("rank/score_pos_mean", window_mean(pos_scores))
        self._metric("rank/score_neg_mean", window_mean(neg_scores))
        self._metric("rank/gap_mean", window_mean(gap))
        self._metric("rank/margin_mean", window_mean(margin))
        self._metric("rank/violation_rate", window_mean((gap < margin).float()))
        for name in ("LOCAL_BOUNDARY", "GLOBAL_SHIFT", "CARDINALITY"):
            type_indices = [i for i, value in enumerate(negative_types) if value == name]
            type_denominator = int(rank_type_counts.get(name, len(type_indices)))
            if type_denominator <= 0:
                continue
            if type_indices:
                index = torch.tensor(type_indices, device=per_pair.device, dtype=torch.long)
                value = window_mean(per_pair.index_select(0, index), type_denominator)
            else:
                value = 0.0
            self._metric(f"rank/type_{name}_loss", value)
        return (total_loss, outputs) if return_outputs else total_loss

    def tc_trace_summary(self):
        return {
            "sha256": self._tc_positive_trace.hexdigest(),
            "positive_examples_seen": self._tc_positive_examples,
            "micro_batches_seen": self._tc_positive_batches,
            "first_sample_ids": self._tc_trace_first,
            "last_sample_ids": self._tc_trace_last,
        }


class TemporalContrastiveSft(SwiftSft):
    """Load positive examples and optionally add contrastive negatives lazily."""

    def __init__(self, args=None):
        self.tc_enabled = _truthy_env("TEMA_TEMPORAL_INIT_TC_ENABLED", False)
        super().__init__(args)

    def _encode_pair(self, row, return_length=True):
        # ``negative_*`` is a reserved Swift prefix for embedding/reranker
        # datasets. Keep the required offline field names in JSON, but do not
        # expose those metadata fields to the ordinary SFT template parser.
        template_row = {
            key: row[key]
            for key in ("messages", "audios", "images", "videos", "objects", "tools")
            if key in row and row[key] is not None
        }
        positive = self.template.encode(template_row, return_length=return_length)
        positive["_tc_sample_id"] = str(row.get("_tc_sample_id", ""))
        if not self.tc_enabled or not row.get("rank_enabled"):
            return positive
        negative_response = row.get("negative_response")
        if not negative_response:
            raise ValueError(f"rank-enabled row lacks negative_response: {row.get('_tc_sample_id')}")

        negative_row = copy.deepcopy(template_row)
        for message in reversed(negative_row["messages"]):
            if message["role"] == "assistant":
                message["content"] = negative_response
                break
        else:
            raise ValueError("row has no assistant message")
        negative = self.template.encode(negative_row, return_length=return_length)
        positive["_tc_negative_encoded"] = negative
        positive["_tc_rank_margin"] = float(row["rank_margin"])
        positive["_tc_negative_type"] = str(row["negative_type"])
        return positive

    def _post_process_datasets(self, datasets):
        datasets = super()._post_process_datasets(datasets)
        for dataset in datasets:
            if isinstance(dataset, LazyLLMDataset):
                dataset.encode_func = self._encode_pair
            elif dataset is not None:
                raise RuntimeError("TC-SFT requires lazy_tokenize=true and LazyLLMDataset inputs")
        return datasets
