"""Inference engine, audio caches, dialogue sharding, and evidence scoring helpers."""

from __future__ import annotations

import copy

import json

import os

import re

import sys

from collections import OrderedDict

from pathlib import Path

from types import SimpleNamespace

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from scipy.optimize import linear_sum_assignment

from tema_chat.evaluation.sidecar import AUDIO_LABEL_RE, SPAN_BLOCK_RE, decode_time_run, split_interval_parts

ROUTE_RE = re.compile(r"<route>(.*?)</route>", re.S)


SPAN_RE = re.compile(r"<span>(.*?)</span>", re.S)


REASON_RE = re.compile(r"<reason>(.*?)</reason>", re.S)


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)


_AUDIO_ARRAY_SOURCE_KEYS: Dict[int, Tuple[Any, ...]] = {}


def extract_yes_no(text: str) -> Optional[bool]:
    """Coarse yes/no extraction from a free-text <answer>, for scoring against
    the sidecar's answer_struct.exists on existence-type questions. This is
    the one place free-text answer content is checked at all -- everything
    else (comparison/count/etc. answer types) is left unscored, see the
    report's `note` field."""
    m = ANSWER_RE.search(text)
    if not m:
        return None
    answer = m.group(1).strip().lower()
    if not answer:
        return None
    prefix = re.split(r'[\s.,!:;]', answer, maxsplit=1)[0]
    if prefix in ('yes', 'yeah', 'yep'):
        return True
    if prefix in ('no', 'nope'):
        return False
    return None


def parse_format(text: str) -> Dict[str, bool]:
    route = ROUTE_RE.search(text)
    span = SPAN_RE.search(text)
    reason = REASON_RE.search(text)
    answer = ANSWER_RE.search(text)
    ordered = bool(route and span and reason and answer and route.start() < span.start() < reason.start() < answer.start())
    return {
        'format_ok': ordered and text.count('<think>') == 1 and text.count('</think>') == 1,
        'route_parse': route is not None,
        'span_parse': span is not None,
        'reason_parse': reason is not None,
        'answer_parse': answer is not None,
        'empty': len(text.strip()) == 0,
    }


def parse_route_set(text: str) -> Optional[set]:
    m = ROUTE_RE.search(text)
    if not m:
        return None
    return {int(x) for x in AUDIO_LABEL_RE.findall(m.group(1))}


def parse_spans(text: str) -> Optional[Dict[int, Optional[List[Tuple[float, float]]]]]:
    """Per-audio interval list, or None for a malformed (non-NONE, unparseable)
    body -- distinct from an empty list (a valid, well-formed NONE / no-evidence
    span). Conflating the two would let garbled output like "[-]" silently
    score as a perfect match against a genuine gold NONE."""
    m = SPAN_RE.search(text)
    if not m:
        return None
    result: Dict[int, Optional[List[Tuple[float, float]]]] = {}
    for block in SPAN_BLOCK_RE.finditer(m.group(1)):
        label = int(block.group(1))
        body = block.group(2).strip()
        if body == 'NONE':
            result[label] = []
            continue
        intervals: List[Tuple[float, float]] = []
        malformed = False
        parts = split_interval_parts(body)
        if not parts:
            malformed = True
        for part in parts:
            if '-' not in part:
                malformed = True
                continue
            left, right = part.split('-', 1)
            try:
                start = decode_time_run(left.strip())
                end = decode_time_run(right.strip())
                if end > start >= 0:
                    intervals.append((start, end))
                else:
                    malformed = True
            except Exception:
                malformed = True
        result[label] = None if malformed else intervals
    return result


def _union_length(intervals: Sequence[Tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0.0
    cur_s, cur_e = ordered[0]
    for s, e in ordered[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return total


def _intersection_length(a: Sequence[Tuple[float, float]], b: Sequence[Tuple[float, float]]) -> float:
    total = 0.0
    for s1, e1 in a:
        for s2, e2 in b:
            s, e = max(s1, s2), min(e1, e2)
            if s < e:
                total += e - s
    return total


def interval_set_iou(gold: Sequence[Tuple[float, float]], pred: Sequence[Tuple[float, float]]) -> float:
    if not gold and not pred:
        return 1.0
    union_g = _union_length(gold)
    union_p = _union_length(pred)
    inter = _intersection_length(gold, pred)
    denom = union_g + union_p - inter
    if denom <= 0:
        return 1.0
    return max(0.0, min(1.0, inter / denom))


def interval_iou(left: Tuple[float, float], right: Tuple[float, float]) -> float:
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union > 0 else 0.0


def dense_span_metrics(
    gold_spans: Mapping[int, Optional[Sequence[Tuple[float, float]]]],
    pred_spans: Optional[Mapping[int, Optional[Sequence[Tuple[float, float]]]]],
    gold_route: Sequence[int],
    pred_route: Optional[Sequence[int]],
) -> Dict[str, Any]:
    """Occurrence-aware dense evidence metrics with explicit NONE scoring."""
    pred_spans = pred_spans or {}
    malformed = any(intervals is None for intervals in pred_spans.values())
    gold_route_set = set(gold_route)
    pred_route_set = set(pred_route or ())
    pred_span_route = set(pred_spans)

    gold_count = sum(len(gold_spans.get(audio_id) or ()) for audio_id in gold_route_set)
    pred_count = sum(len(intervals or ()) for intervals in pred_spans.values())
    matched_ious: List[float] = []
    if not malformed:
        for audio_id in gold_route_set & pred_span_route:
            gold_intervals = list(gold_spans.get(audio_id) or ())
            pred_intervals = list(pred_spans.get(audio_id) or ())
            if not gold_intervals or not pred_intervals:
                continue
            scores = np.asarray(
                [
                    [interval_iou(pred, gold) for gold in gold_intervals]
                    for pred in pred_intervals
                ]
            )
            rows, columns = linear_sum_assignment(-scores)
            matched_ious.extend(float(scores[row, column]) for row, column in zip(rows, columns))

    matched_iou_score = (
        sum(matched_ious) / max(gold_count, pred_count)
        if max(gold_count, pred_count) > 0 else 1.0
    )
    if malformed:
        matched_iou_score = 0.0
    result: Dict[str, Any] = {
        'span_gold_occurrences': gold_count,
        'span_pred_occurrences': pred_count,
        'span_matched_iou_sum': sum(matched_ious),
        'span_matched_iou': matched_iou_score,
        'span_malformed': malformed,
    }
    for threshold in (0.5, 0.7, 0.9):
        suffix = str(threshold).replace('.', '_')
        true_positive = 0 if malformed else sum(iou >= threshold for iou in matched_ious)
        precision = true_positive / pred_count if pred_count else float(gold_count == 0)
        recall = true_positive / gold_count if gold_count else float(pred_count == 0)
        if malformed:
            precision = recall = 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0 else 0.0
        )
        result.update({
            f'span_occurrence_tp_at_{suffix}': true_positive,
            f'span_occurrence_precision_at_{suffix}': precision,
            f'span_occurrence_recall_at_{suffix}': recall,
            f'span_occurrence_f1_at_{suffix}': f1,
        })

    negative_audio_ids = {
        audio_id for audio_id in gold_route_set if not (gold_spans.get(audio_id) or ())
    }
    correct_none = sum(
        audio_id in pred_spans and pred_spans[audio_id] == []
        for audio_id in negative_audio_ids
    )
    none_accuracy = (
        correct_none / len(negative_audio_ids) if negative_audio_ids else None
    )
    positive_score = (
        2 * sum(matched_ious) / (gold_count + pred_count)
        if gold_count + pred_count > 0 else 1.0
    )
    if malformed:
        positive_score = 0.0
    if gold_count and negative_audio_ids:
        content_score = 0.8 * positive_score + 0.2 * float(none_accuracy)
    elif gold_count:
        content_score = positive_score
    else:
        content_score = float(none_accuracy or 0.0)
    route_f1 = route_set_f1(gold_route_set, pred_route_set)[2]
    span_route_f1 = route_set_f1(gold_route_set, pred_span_route)[2]
    result['span_none_audio_count'] = len(negative_audio_ids)
    result['span_none_accuracy'] = none_accuracy
    result['span_dense_score'] = min(route_f1, span_route_f1) * content_score
    return result


def route_set_f1(gold: set, pred: set) -> Tuple[float, float, float]:
    if not gold and not pred:
        return 1.0, 1.0, 1.0
    if not pred:
        return 0.0, 0.0, 0.0
    tp = len(gold & pred)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def read_jsonl(path: Path):
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _audio_path_cache_key(path: str) -> Tuple[Any, ...]:
    resolved = os.path.realpath(os.path.abspath(path))
    try:
        stat = os.stat(resolved)
        return ('path', resolved, stat.st_size, stat.st_mtime_ns)
    except OSError:
        return ('path', resolved)


def _remember_audio_source(value: Any, key: Tuple[Any, ...]) -> None:
    try:
        import numpy as np
    except Exception:
        return
    if isinstance(value, np.ndarray):
        _AUDIO_ARRAY_SOURCE_KEYS[id(value)] = key
    elif isinstance(value, tuple) and value and isinstance(value[0], np.ndarray):
        _AUDIO_ARRAY_SOURCE_KEYS[id(value[0])] = key


def _hashable(value: Any) -> Any:
    try:
        hash(value)
        return value
    except TypeError:
        pass
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(v) for v in value)
    return repr(value)


def install_audio_decode_cache(max_entries: int):
    from functools import lru_cache

    from swift.template import vision_utils

    original_load_audio = vision_utils.load_audio
    existing_cache_info = getattr(original_load_audio, '_sft_audio_cache_info', None)
    if existing_cache_info is not None:
        return existing_cache_info

    def _backend_key() -> str:
        return (
            os.environ.get('SWIFT_AUDIO_LOAD_BACKEND')
            or os.environ.get('swift_audio_load_backend')
            or 'librosa'
        )

    @lru_cache(maxsize=max_entries)
    def _cached_load_audio(path: str, sampling_rate: int, return_sr: bool, mono: bool, backend: str):
        value = original_load_audio(path, sampling_rate, return_sr=return_sr, mono=mono)
        _remember_audio_source(
            value,
            (_audio_path_cache_key(path), int(sampling_rate), bool(return_sr), bool(mono), backend),
        )
        return value

    def cached_load_audio(audio, sampling_rate: int, return_sr: bool = False, mono: bool = True):
        if isinstance(audio, (str, os.PathLike)):
            return _cached_load_audio(
                os.fspath(audio),
                int(sampling_rate),
                bool(return_sr),
                bool(mono),
                _backend_key(),
            )
        return original_load_audio(audio, sampling_rate, return_sr=return_sr, mono=mono)

    cached_load_audio._sft_audio_cache_info = _cached_load_audio.cache_info
    vision_utils.load_audio = cached_load_audio

    # Some template modules import load_audio directly at module import time.
    for module in list(sys.modules.values()):
        if getattr(module, 'load_audio', None) is original_load_audio:
            setattr(module, 'load_audio', cached_load_audio)

    return _cached_load_audio.cache_info


class _CachedAudioFeatureExtractor:
    def __init__(self, feature_extractor: Any, max_entries: int):
        from threading import RLock

        self._feature_extractor = feature_extractor
        self._max_entries = max_entries
        self._cache: OrderedDict[Tuple[Any, ...], Dict[str, Any]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._lock = RLock()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._feature_extractor, name)

    @staticmethod
    def _clone_value(value: Any) -> Any:
        try:
            import numpy as np
            import torch
        except Exception:
            np = None
            torch = None
        if torch is not None and torch.is_tensor(value):
            return value.clone()
        if np is not None and isinstance(value, np.ndarray):
            return value.copy()
        return copy.deepcopy(value)

    @classmethod
    def _clone_feature_dict(cls, feature_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {key: cls._clone_value(value) for key, value in feature_dict.items()}

    @staticmethod
    def _samples(raw_speech: Any) -> Optional[List[Any]]:
        import numpy as np

        if isinstance(raw_speech, np.ndarray):
            if raw_speech.ndim == 1:
                return [raw_speech]
            if raw_speech.ndim == 2:
                return [raw_speech[i] for i in range(raw_speech.shape[0])]
            return None
        if isinstance(raw_speech, (list, tuple)) and raw_speech:
            if not isinstance(raw_speech[0], (np.ndarray, list, tuple)):
                return None
            samples = []
            for item in raw_speech:
                if isinstance(item, np.ndarray):
                    samples.append(item)
                elif isinstance(item, (list, tuple)):
                    array = np.asarray(item, dtype=np.float32)
                    if array.ndim != 1:
                        return None
                    samples.append(array)
                else:
                    return None
            return samples
        return None

    @staticmethod
    def _kwargs_key(kwargs: Dict[str, Any]) -> Tuple[Any, ...]:
        return tuple(sorted((key, _hashable(value)) for key, value in kwargs.items()))

    @staticmethod
    def _sample_key(sample: Any, kwargs_key: Tuple[Any, ...]) -> Tuple[Any, ...]:
        import numpy as np

        source_key = _AUDIO_ARRAY_SOURCE_KEYS.get(id(sample))
        if source_key is None:
            array = np.asarray(sample)
            source_key = ('array_id', id(sample), tuple(array.shape), str(array.dtype))
        return source_key, kwargs_key

    @staticmethod
    def _concat_feature_dicts(feature_dicts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        import numpy as np
        import torch

        merged: Dict[str, Any] = {}
        for key in feature_dicts[0].keys():
            values = [feature_dict[key] for feature_dict in feature_dicts]
            first = values[0]
            if torch.is_tensor(first):
                merged[key] = torch.cat(values, dim=0)
            elif isinstance(first, np.ndarray):
                merged[key] = np.concatenate(values, axis=0)
            elif isinstance(first, list):
                merged_list = []
                for value in values:
                    merged_list.extend(value)
                merged[key] = merged_list
            else:
                merged[key] = [copy.deepcopy(value) for value in values]
        return merged

    def _get_one(self, sample: Any, kwargs: Dict[str, Any], kwargs_key: Tuple[Any, ...]) -> Dict[str, Any]:
        key = self._sample_key(sample, kwargs_key)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._clone_feature_dict(cached)

        feature = self._feature_extractor(sample, **kwargs)
        stored = self._clone_feature_dict(dict(feature))
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._clone_feature_dict(cached)
            self._cache[key] = stored
            self._cache.move_to_end(key)
            self._misses += 1
            while len(self._cache) > self._max_entries:
                self._cache.popitem(last=False)
        return self._clone_feature_dict(stored)

    def __call__(self, raw_speech: Any, **kwargs: Any):
        from transformers.feature_extraction_utils import BatchFeature

        samples = self._samples(raw_speech)
        if not samples:
            return self._feature_extractor(raw_speech, **kwargs)

        kwargs_key = self._kwargs_key(kwargs)
        feature_dicts = [self._get_one(sample, kwargs, kwargs_key) for sample in samples]
        return BatchFeature(data=self._concat_feature_dicts(feature_dicts))

    def cache_info(self):
        with self._lock:
            return SimpleNamespace(
                hits=self._hits,
                misses=self._misses,
                maxsize=self._max_entries,
                currsize=len(self._cache),
            )


def install_audio_feature_cache(engine: Any, max_entries: int):
    if max_entries <= 0:
        return None
    processor = getattr(engine, 'processor', None)
    if processor is None and getattr(engine, 'template', None) is not None:
        processor = getattr(engine.template, 'processor', None)
    feature_extractor = getattr(processor, 'feature_extractor', None)
    if feature_extractor is None:
        return None
    existing_cache_info = getattr(feature_extractor, '_sft_audio_feature_cache_info', None)
    if existing_cache_info is not None:
        return existing_cache_info

    cached = _CachedAudioFeatureExtractor(feature_extractor, max_entries)
    cached._sft_audio_feature_cache_info = cached.cache_info
    processor.feature_extractor = cached
    if getattr(engine, 'template', None) is not None and getattr(engine.template, 'processor', None) is processor:
        engine.template.processor.feature_extractor = cached
    return cached.cache_info


def load_engine(
    base_model: str,
    checkpoint: Optional[str],
    torch_dtype_name: str,
    device_map,
    max_batch_size: int,
    attn_impl: Optional[str] = None,
    model_type: str = 'qwen2_5_omni_ate_sft',
    load_extra: bool = True,
):
    import torch
    from swift.infer_engine import TransformersEngine

    dtype = getattr(torch, torch_dtype_name)
    engine_kwargs = dict(
        model=base_model,
        model_type=model_type,
        torch_dtype=dtype,
        device_map=device_map,
        max_batch_size=max_batch_size,
        attn_impl=attn_impl,
    )
    if checkpoint:
        engine_kwargs['adapters'] = [checkpoint]
    engine = TransformersEngine(**engine_kwargs)
    if checkpoint and load_extra:
        from sft.model.save_extra_state import load_sft_extra

        load_sft_extra(engine.model, checkpoint, strict=True)
    return engine


def context_length_key(context: Dict[str, Any]) -> Tuple[int, int]:
    """Cheap proxy for how expensive a turn's context is to run: audio count
    dominates (each audio contributes many frame tokens after the audio
    tower), text length is a tiebreaker. Sorting requests by this before
    batching keeps each batch's sequence lengths close together, cutting
    padding waste and avoiding the failure mode where a batch happens to draw
    several long-context turns at once and OOMs."""
    audios = context.get('audios') or []
    text_len = sum(len(m.get('content', '')) for m in context.get('messages', []))
    return (len(audios), text_len)


def dialogue_eval_cost(row: Dict[str, Any]) -> Tuple[int, int, int]:
    messages = row['messages']
    audio_ptr = 0
    turns = 0
    expanded_audio_refs = 0
    context_chars = 0
    running_chars = 0
    for msg in messages:
        if msg['role'] == 'assistant':
            turns += 1
            expanded_audio_refs += audio_ptr
            context_chars += running_chars
        running_chars += len(msg.get('content', ''))
        if msg['role'] == 'user':
            audio_ptr += msg['content'].count('<audio>')
    return expanded_audio_refs, context_chars, turns


def balanced_shard_items(indexed: List[Tuple[int, Dict[str, Any], Dict[str, Any]]], num_shards: int, shard_index: int):
    bins: List[List[Tuple[int, Dict[str, Any], Dict[str, Any]]]] = [[] for _ in range(num_shards)]
    costs = [(0, 0, 0) for _ in range(num_shards)]

    def add_cost(left: Tuple[int, int, int], right: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return left[0] + right[0], left[1] + right[1], left[2] + right[2]

    scored = []
    for item in indexed:
        _, row, _ = item
        scored.append((dialogue_eval_cost(row), item))
    for cost, item in sorted(scored, key=lambda x: x[0], reverse=True):
        target = min(range(num_shards), key=lambda i: costs[i])
        bins[target].append(item)
        costs[target] = add_cost(costs[target], cost)
    return sorted(bins[shard_index], key=lambda item: item[0])

