"""Structured response parsing and shared scoring helpers used by release evaluation."""

from __future__ import annotations

import json

import math

import re

from dataclasses import dataclass

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from scipy.optimize import linear_sum_assignment

Span = Tuple[float, float]


STRUCTURE_RE = re.compile(
    r"\A\s*<think>\s*"
    r"<route>(?P<route>.*?)</route>\s*"
    r"<span>(?P<span>.*?)</span>\s*"
    r"<reason>(?P<reason>.*?)</reason>\s*"
    r"</think>\s*<answer>(?P<answer>.*?)</answer>\s*\Z",
    re.DOTALL,
)


AUDIO_RE = re.compile(r"Audios\{(\d+)\}")


SPAN_ITEM_RE = re.compile(r"Audios\{(\d+)\}\s*\[([^\]]*)\]")


RANGE_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*[-\u2013]\s*(-?\d+(?:\.\d+)?)")


NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


YES_RE = re.compile(r"^\s*yes\b", re.IGNORECASE)


NO_RE = re.compile(r"^\s*no\b", re.IGNORECASE)


CONSISTENCY_TASKS = {
    "A1",
    "A3",
    "A5",
    "A5-gap",
    "A6-no",
    "A6-yes",
    "A7",
    "A8",
    "A10",
    "A13",
    "A14",
    "A16",
    "A17",
    "A18",
}


@dataclass(frozen=True)
class ParsedCompletion:
    valid: bool
    route: Tuple[int, ...] = ()
    spans: Mapping[int, Tuple[Span, ...]] | None = None
    answer: str = ""
    reason: str = ""
    error: str = ""


def parse_jsonish(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def parse_completion(text: str) -> ParsedCompletion:
    match = STRUCTURE_RE.fullmatch(str(text))
    if match is None:
        return ParsedCompletion(False, error="structure")
    if not match.group("reason").strip() or not match.group("answer").strip():
        return ParsedCompletion(False, error="empty_reason_or_answer")

    route_text = match.group("route")
    route = tuple(int(value) for value in AUDIO_RE.findall(route_text))
    route_rest = AUDIO_RE.sub("", route_text)
    route_rest = re.sub(r"(?:\s|,|;|and|&)+", "", route_rest, flags=re.IGNORECASE)
    if not route or route_rest or len(set(route)) != len(route):
        return ParsedCompletion(False, error="route")

    span_text = match.group("span")
    span_matches = list(SPAN_ITEM_RE.finditer(span_text))
    span_rest = SPAN_ITEM_RE.sub("", span_text)
    span_rest = re.sub(r"[\s;]+", "", span_rest)
    if not span_matches or span_rest:
        return ParsedCompletion(False, error="span_block")

    spans: Dict[int, Tuple[Span, ...]] = {}
    for span_match in span_matches:
        audio_id = int(span_match.group(1))
        if audio_id in spans:
            return ParsedCompletion(False, error="duplicate_span_audio")
        content = span_match.group(2).strip()
        if content.upper() == "NONE":
            spans[audio_id] = ()
            continue
        parsed: List[Span] = []
        pieces = [piece.strip() for piece in content.split(",")]
        if not pieces or any(not piece for piece in pieces):
            return ParsedCompletion(False, error="span_value")
        for piece in pieces:
            range_match = RANGE_RE.fullmatch(piece)
            if range_match is None:
                return ParsedCompletion(False, error="span_value")
            start, end = float(range_match.group(1)), float(range_match.group(2))
            if start < 0 or end <= start:
                return ParsedCompletion(False, error="span_bounds")
            parsed.append((start, end))
        spans[audio_id] = tuple(parsed)

    if set(spans) != set(route):
        return ParsedCompletion(False, error="route_span_mismatch")
    return ParsedCompletion(
        True,
        route=route,
        spans=spans,
        answer=match.group("answer").strip(),
        reason=match.group("reason").strip(),
    )


def set_f1(predicted: Iterable[int], gold: Iterable[int]) -> float:
    pred_set, gold_set = set(predicted), set(gold)
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    overlap = len(pred_set & gold_set)
    precision = overlap / len(pred_set)
    recall = overlap / len(gold_set)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def interval_iou(left: Span, right: Span) -> float:
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    if union == 0:
        return 1.0 if left == right else 0.0
    return intersection / union


def matched_span_score(predicted: Sequence[Span], gold: Sequence[Span]) -> float:
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    scores = np.asarray([[interval_iou(pred, target) for target in gold] for pred in predicted])
    rows, columns = linear_sum_assignment(-scores)
    return float(scores[rows, columns].sum() / max(len(predicted), len(gold)))


def normalize_gold_route(value: Any) -> List[int]:
    value = parse_jsonish(value, [])
    result: List[int] = []
    for item in value if isinstance(value, list) else []:
        text = str(item)
        if text.isdigit():
            result.append(int(text))
        elif text.startswith("a") and text[1:].isdigit():
            result.append(int(text[1:]))
    return sorted(set(result))


def normalize_gold_spans(value: Any) -> Dict[int, List[Span]]:
    value = parse_jsonish(value, {})
    result: Dict[int, List[Span]] = {}
    if not isinstance(value, dict):
        return result
    for raw_audio, raw_spans in value.items():
        text = str(raw_audio)
        if text.startswith("a"):
            text = text[1:]
        if not text.isdigit() or not isinstance(raw_spans, list):
            continue
        spans: List[Span] = []
        for raw_span in raw_spans:
            if isinstance(raw_span, (list, tuple)) and len(raw_span) == 2:
                spans.append((float(raw_span[0]), float(raw_span[1])))
        result[int(text)] = spans
    return result


def normalize_audio_lengths(value: Any) -> Dict[int, float]:
    value = parse_jsonish(value, {})
    result: Dict[int, float] = {}
    if not isinstance(value, dict):
        return result
    for raw_audio, raw_length in value.items():
        text = str(raw_audio)
        if text.startswith("a"):
            text = text[1:]
        try:
            audio_id = int(text)
            length = float(raw_length)
        except (TypeError, ValueError):
            continue
        if audio_id > 0 and math.isfinite(length) and length > 0:
            result[audio_id] = length
    return result


def scored_completion(text: str, audio_lengths: Any = None) -> ParsedCompletion:
    parsed = parse_completion(text)
    lengths = normalize_audio_lengths(audio_lengths)
    if not parsed.valid or not lengths or parsed.spans is None:
        return parsed
    for audio_id, spans in parsed.spans.items():
        length = lengths.get(audio_id)
        if length is None:
            return ParsedCompletion(False, error="unknown_audio")
        if any(end > length + 0.0500001 for _, end in spans):
            return ParsedCompletion(False, error="span_out_of_bounds")
    return parsed


def format_reward(completion: str, audio_lengths: Any = None) -> float:
    return 0.0 if scored_completion(completion, audio_lengths).valid else -1.0


def route_reward(completion: str, gold_route: Any, audio_lengths: Any = None) -> float:
    parsed = scored_completion(completion, audio_lengths)
    return set_f1(parsed.route, normalize_gold_route(gold_route)) if parsed.valid else 0.0


def span_reward_components(
    completion: str,
    gold_route: Any,
    gold_spans: Any,
    audio_lengths: Any = None,
) -> Dict[str, float]:
    """Score dense, query-conditioned occurrence evidence.

    Audio membership, positive occurrences, and explicit NONE decisions are
    separate signals. This prevents a missing audio from being confused with
    ``Audios{k}[NONE]`` and prevents many easy NONE labels from overwhelming a
    missed positive occurrence.
    """
    parsed = scored_completion(completion, audio_lengths)
    if not parsed.valid or parsed.spans is None:
        return {
            "score": 0.0,
            "route_membership": 0.0,
            "positive_occurrence": 0.0,
            "none_accuracy": 0.0,
        }
    target_spans = normalize_gold_spans(gold_spans)
    target_route = set(normalize_gold_route(gold_route))
    if set(target_spans) - target_route:
        return {
            "score": 0.0,
            "route_membership": 0.0,
            "positive_occurrence": 0.0,
            "none_accuracy": 0.0,
        }

    predicted_route = set(parsed.spans)
    route_membership = set_f1(predicted_route, target_route)
    gold_occurrences = sum(len(target_spans.get(audio_id, ())) for audio_id in target_route)
    predicted_occurrences = sum(len(parsed.spans[audio_id]) for audio_id in predicted_route)
    matched_iou = 0.0
    for audio_id in target_route & predicted_route:
        predicted = parsed.spans[audio_id]
        target = target_spans.get(audio_id, ())
        if predicted and target:
            scores = np.asarray([[interval_iou(pred, gold) for gold in target] for pred in predicted])
            rows, columns = linear_sum_assignment(-scores)
            matched_iou += float(scores[rows, columns].sum())
    if gold_occurrences:
        denominator = gold_occurrences + predicted_occurrences
        positive_score = 2.0 * matched_iou / denominator if denominator else 0.0
    else:
        positive_score = 1.0 if predicted_occurrences == 0 else 0.0

    negative_audio_ids = {
        audio_id for audio_id in target_route if not target_spans.get(audio_id, ())
    }
    if negative_audio_ids:
        correct_none = sum(
            audio_id in predicted_route and not parsed.spans[audio_id]
            for audio_id in negative_audio_ids
        )
        none_score = correct_none / len(negative_audio_ids)
    else:
        none_score = 1.0

    if gold_occurrences and negative_audio_ids:
        content_score = 0.8 * positive_score + 0.2 * none_score
    elif gold_occurrences:
        content_score = positive_score
    else:
        content_score = none_score
    score = route_membership * content_score
    return {
        "score": float(score),
        "route_membership": float(route_membership),
        "positive_occurrence": float(positive_score),
        "none_accuracy": float(none_score),
    }


def span_reward(
    completion: str,
    gold_route: Any,
    gold_spans: Any,
    audio_lengths: Any = None,
) -> float:
    return span_reward_components(completion, gold_route, gold_spans, audio_lengths)["score"]


def extract_answer(text: str) -> str:
    parsed = parse_completion(text)
    if parsed.valid:
        return parsed.answer
    match = re.search(r"<answer>(.*?)</answer>", str(text), re.DOTALL)
    return match.group(1).strip() if match else str(text).strip()


def normalized_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def answer_head(text: str) -> str:
    """Return the short answer sentence without splitting decimal numbers."""
    value = str(text).strip()
    boundary = re.search(r"[!?]|\.(?!\d)", value)
    return value[: boundary.start()].strip() if boundary else value


def answer_polarity(answer: str) -> bool | None:
    answer = answer_head(answer)
    if YES_RE.search(answer):
        return True
    if NO_RE.search(answer):
        return False
    normalized = normalized_text(answer)
    if normalized.startswith("there is no ") or normalized.startswith("there are no "):
        return False
    if " not present" in f" {normalized}" or " do not hear" in f" {normalized}":
        return False
    return None


def first_number(answer: str, require_seconds: bool = False) -> float | None:
    if require_seconds:
        match = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b", answer, re.IGNORECASE)
        return float(match.group(1)) if match else None
    match = NUMBER_RE.search(answer)
    return float(match.group()) if match else None


def answer_ranges(answer: str) -> List[Span]:
    return [(float(match.group(1)), float(match.group(2))) for match in RANGE_RE.finditer(answer)]


def clip_set(answer: str) -> set[int]:
    source = answer_head(answer)
    if re.match(
        r"^\s*(?:none\b|neither\b|no\s+(?:clips?|audios?)\b|not\s+any\b)",
        source,
        re.IGNORECASE,
    ):
        return set()
    if not re.search(r"\b(?:clips?|audios?)\b", source, re.IGNORECASE):
        return set()
    return {int(value) for value in re.findall(r"(?<![.\d])\d+(?![.\d])", source)}


def gold_clip_set(task_type: str, answer_struct: Mapping[str, Any]) -> set[int]:
    values: Any = []
    if task_type == "A7":
        values = [item.get("audio") for item in answer_struct.get("hits", []) if isinstance(item, dict)]
    elif task_type == "A9":
        values = answer_struct.get("earliest_audios", [])
    else:
        values = (
            answer_struct.get("answer_audios")
            or answer_struct.get("winners")
            or answer_struct.get("winner_audios")
            or answer_struct.get("winner_clips")
            or []
        )
        single = answer_struct.get("answer_audio") or answer_struct.get("winner") or answer_struct.get("max_audio")
        if single and not values:
            values = [single]
    result: set[int] = set()
    for value in values if isinstance(values, list) else [values]:
        text = str(value)
        match = re.search(r"(\d+)", text)
        if match:
            result.add(int(match.group(1)))
    return result


def dense_number_reward(
    predicted: float | None,
    gold: float,
    tolerance: float = 1.0,
    exact_tolerance: float = 0.0500001,
) -> float:
    if predicted is None:
        return 0.0
    error = abs(predicted - gold)
    if error <= exact_tolerance:
        return 1.0
    return max(0.0, 1.0 - error / max(tolerance, 0.1))


def token_f1(predicted: str, gold: str) -> float:
    pred_tokens = normalized_text(predicted).split()
    gold_tokens = normalized_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    pred_counts, gold_counts = {}, {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1
    overlap = sum(min(count, gold_counts.get(token, 0)) for token, count in pred_counts.items())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(pred_tokens), overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def canonical_token_precision(predicted: str, gold: str) -> float:
    pred_tokens = _canonical_event_tokens(predicted)
    gold_tokens = _canonical_event_tokens(gold)
    if not pred_tokens:
        return 0.0
    return len(pred_tokens & gold_tokens) / len(pred_tokens)


def _event_aliases(event: Any) -> List[str]:
    label = str(event).strip()
    aliases = [label, *re.split(r"\s*[,;/]\s*", label)]
    for main, parenthetical in re.findall(r"([^()]*)\(([^()]*)\)", label):
        aliases.extend([main.strip(), parenthetical.strip()])
    result = []
    for alias in aliases:
        normalized = normalized_text(alias)
        if normalized and normalized not in result:
            result.append(normalized)
    return result


def _canonical_event_tokens(text: str) -> set[str]:
    synonyms = {
        "laughter": "laugh",
        "man": "male",
        "speak": "speech",
        "talk": "speech",
        "voice": "speech",
        "woman": "female",
        "kid": "child",
    }
    ignored = {"a", "an", "and", "instance", "of", "only", "s", "the"}
    result: set[str] = set()
    for raw_token in normalized_text(text).split():
        token = raw_token
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("ing") and len(token) > 5:
            token = token[:-3]
            if len(token) > 2 and token[-1] == token[-2]:
                token = token[:-1]
        elif token.endswith("ed") and len(token) > 4:
            token = token[:-2]
        elif token.endswith("s") and len(token) > 3:
            token = token[:-1]
        token = synonyms.get(token, token)
        if token not in ignored:
            result.add(token)
    return result


def _event_similarity(text: str, event: Any) -> float:
    text_tokens = _canonical_event_tokens(text)
    if not text_tokens:
        return 0.0
    best = 0.0
    for alias in _event_aliases(event):
        alias_tokens = _canonical_event_tokens(alias)
        if not alias_tokens:
            continue
        overlap = len(text_tokens & alias_tokens)
        best = max(best, 2 * overlap / (len(text_tokens) + len(alias_tokens)))
    return best


def _mentions_event(answer: str, event: Any) -> bool:
    answer_tokens = _canonical_event_tokens(answer)
    return any(
        bool(alias_tokens := _canonical_event_tokens(alias))
        and alias_tokens.issubset(answer_tokens)
        for alias in _event_aliases(event)
    )


def _claimed_first_index(answer: str, events: Sequence[str]) -> int | None:
    normalized = normalized_text(answer)
    subject = ""
    patterns = (
        r"^(.*?)\b(?:comes?|came|occurs?|occurred|happens?|happened|starts?|started|begins?|began|is|was)\s+(?:the\s+)?(?:first|earlier)\b",
        r"^(.*?)\b(?:before|precedes?)\b",
        r"^(.*?)\bthen\b",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            subject = match.group(1)
            break
    if not subject:
        return None
    scores = [_event_similarity(subject, event) for event in events]
    winner = max(range(len(scores)), key=scores.__getitem__)
    runner_up = max(
        (score for index, score in enumerate(scores) if index != winner),
        default=0.0,
    )
    if scores[winner] < 0.3 or scores[winner] - runner_up < 0.1:
        return None
    return winner


def _claims_event_first(answer: str, event: Any) -> bool:
    normalized = normalized_text(answer)
    for alias in _event_aliases(event):
        escaped = re.escape(alias)
        if re.search(
            rf"\b{escaped}\b(?:\s+\w+){{0,8}}\s+"
            rf"(?:comes?|came|occurs?|occurred|happens?|happened|starts?|started|begins?|began|is|was)\s+"
            rf"(?:the\s+)?(?:first|earlier)\b",
            normalized,
        ):
            return True
        if re.search(rf"\b{escaped}\b(?:\s+\w+){{0,8}}\s+(?:before|precedes?)\b", normalized):
            return True
    return False


def _order_answer_reward(answer: str, struct: Mapping[str, Any]) -> float:
    events = [str(event) for event in struct.get("events", []) if str(event).strip()]
    if len(events) < 2:
        return 0.0

    onsets = struct.get("onsets", [])
    is_tie = False
    first_index = 0
    if isinstance(onsets, list) and len(onsets) >= len(events):
        numeric_onsets = [float(value) for value in onsets[: len(events)]]
        first_value = min(numeric_onsets)
        winners = [index for index, value in enumerate(numeric_onsets) if math.isclose(value, first_value, abs_tol=0.05)]
        is_tie = len(winners) > 1
        first_index = winners[0]
    elif struct.get("first"):
        first_label = normalized_text(str(struct["first"]))
        for index, event in enumerate(events):
            if first_label in _event_aliases(event):
                first_index = index
                break

    if is_tie:
        tie_claim = bool(re.search(r"\b(?:same time|simultaneous(?:ly)?|tie|tied|together|neither .* first)\b", answer, re.I))
        mentions = sum(_mentions_event(answer, event) for event in events)
        return float(tie_claim) * (0.8 + 0.2 * mentions / len(events))

    claimed_index = _claimed_first_index(answer, events)
    if claimed_index is not None:
        return float(claimed_index == first_index)

    claims = [_claims_event_first(answer, event) for event in events]
    if claims[first_index] and sum(claims) == 1:
        return 1.0
    if any(claims):
        return 0.0

    # Give partial credit only when the correct event and an ordering cue are
    # both present; event-name overlap alone is not evidence of the relation.
    has_order_cue = bool(re.search(r"\b(?:first|earlier|before|precedes?)\b", answer, re.I))
    if has_order_cue and _mentions_event(answer, events[first_index]):
        return 0.5
    return 0.0


def _gold_answer(solution: str) -> str:
    return extract_answer(solution)


def answer_reward(
    completion: str,
    task_type: str,
    answer_struct: Any,
    solution: str,
    audio_lengths: Any = None,
    *,
    allow_exact_match: bool = True,
) -> float:
    parsed = scored_completion(completion, audio_lengths)
    if not parsed.valid:
        return 0.0
    answer = answer_head(parsed.answer)
    gold_answer = _gold_answer(solution)
    struct = parse_jsonish(answer_struct, {})
    if not isinstance(struct, dict):
        struct = {}

    if task_type in {"A6-yes", "A6-no", "A11", "A16"}:
        predicted, gold = answer_polarity(answer), answer_polarity(gold_answer)
        return float(predicted is not None and predicted == gold)

    if task_type == "A1":
        predicted_ranges = answer_ranges(answer)
        gold_ranges = sorted(
            tuple(map(float, item)) for item in struct.get("spans", []) if len(item) == 2
        )
        if not gold_ranges:
            gold_ranges = answer_ranges(answer_head(gold_answer))
        ordinal = struct.get("ordinal")
        if ordinal is not None and gold_ranges:
            index = int(ordinal) - 1
            if len(gold_ranges) == 1:
                gold_ranges = gold_ranges[:1]
            elif 0 <= index < len(gold_ranges):
                gold_ranges = [gold_ranges[index]]
            else:
                return 0.0
        return matched_span_score(predicted_ranges, gold_ranges)

    if task_type in {"A5", "A5-gap", "A17", "A18"}:
        gold_value = struct.get("duration", struct.get("gap_duration"))
        if gold_value is None:
            gold_value = first_number(answer_head(gold_answer), require_seconds=True)
        if gold_value is None:
            return token_f1(answer, gold_answer)
        tolerance = max(0.5, min(2.0, float(gold_value) * 0.5))
        return dense_number_reward(first_number(answer, require_seconds=True), float(gold_value), tolerance)

    if task_type == "A3":
        gold_count = int(struct.get("count", struct.get("value", 0)))
        predicted = _answer_count(answer)
        if predicted is None:
            return 0.0
        error = abs(predicted - gold_count)
        return max(0.0, 1.0 - error / max(gold_count, 1))

    if task_type == "A14":
        counts = struct.get("counts", {})
        gold_values = [int(value) for _, value in sorted(counts.items(), key=lambda pair: int(re.sub(r"\D", "", pair[0]) or 0))]
        predicted_values = _answer_integer_list(answer)
        if not gold_values or len(predicted_values) != len(gold_values):
            return 0.0
        return sum(p == g for p, g in zip(predicted_values, gold_values)) / len(gold_values)

    if task_type == "A4":
        return _order_answer_reward(answer, struct)

    if task_type in {"A7", "A8", "A9", "A10", "A13"}:
        return set_f1(clip_set(answer), gold_clip_set(task_type, struct))

    if task_type == "A2":
        raw_events = struct.get("events", [])
        if not raw_events and struct.get("event"):
            raw_events = [struct["event"]]
        events = [
            str(item.get("event", "")) if isinstance(item, dict) else str(item)
            for item in raw_events
        ]
        events = [event for event in events if event]
        if not events:
            return token_f1(answer, gold_answer)
        recall = sum(_mentions_event(answer, event) for event in events) / len(events)
        precision_proxy = canonical_token_precision(answer, answer_head(gold_answer))
        if not recall or not precision_proxy:
            return 0.0
        return 2 * recall * precision_proxy / (recall + precision_proxy)

    if allow_exact_match and normalized_text(parsed.answer) == normalized_text(gold_answer):
        return 1.0
    return token_f1(answer, gold_answer)


def _span_duration(span: Span) -> float:
    return max(0.0, span[1] - span[0])


def _union_duration(spans: Sequence[Span]) -> float:
    if not spans:
        return 0.0
    merged_total = 0.0
    current_start, current_end = sorted(spans)[0]
    for start, end in sorted(spans)[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged_total += _span_duration((current_start, current_end))
            current_start, current_end = start, end
    return merged_total + _span_duration((current_start, current_end))


def _answer_count(answer: str) -> int | None:
    answer = answer_head(answer)
    if re.search(r"\bnot\b", answer, re.IGNORECASE):
        return None
    match = re.search(r"(?<![.\d-])(\d+)(?![.\d])\s*(?:times?|occurrences?)\b", answer, re.IGNORECASE)
    if match:
        return int(match.group(1))
    normalized = normalized_text(answer)
    word_counts = {
        "zero": 0,
        "once": 1,
        "one": 1,
        "twice": 2,
        "two": 2,
        "thrice": 3,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    for token in normalized.split():
        if token in word_counts:
            return word_counts[token]
    return None


def _answer_integer_list(answer: str) -> List[int]:
    source = answer_head(answer)
    cleaned = re.sub(r"\b(?:clip|audio)s?\s*\d+\b", "", source, flags=re.IGNORECASE)
    raw_values = NUMBER_RE.findall(cleaned)
    values = [float(value) for value in raw_values]
    if any(value < 0 or not value.is_integer() for value in values):
        return []
    return [int(value) for value in values]


def consistency_reward(
    completion: str,
    task_type: str,
    answer_struct: Any,
    _gold_route: Any = None,
    audio_lengths: Any = None,
) -> Tuple[float, bool]:
    """Return (score, applicable).

    The check is intentionally disabled when the current span block is not a
    sufficient statistic for the answer (notably A9 cross-turn comparisons).
    """
    parsed = scored_completion(completion, audio_lengths)
    if not parsed.valid or parsed.spans is None:
        return 0.0, False
    struct = parse_jsonish(answer_struct, {})
    if not isinstance(struct, dict):
        return 0.0, False
    route = list(parsed.route)
    answer = answer_head(parsed.answer)

    if task_type in {"A6-yes", "A6-no", "A16"}:
        polarity = answer_polarity(answer)
        if polarity is None:
            return 0.0, True
        present = any(parsed.spans.get(audio_id, ()) for audio_id in route)
        return float(polarity == present), True

    if task_type == "A3":
        predicted_count = _answer_count(answer)
        span_count = sum(len(parsed.spans.get(audio_id, ())) for audio_id in route)
        return float(predicted_count == span_count), True

    if task_type == "A14":
        answer_counts = _answer_integer_list(answer)
        span_counts = [len(parsed.spans.get(audio_id, ())) for audio_id in sorted(route)]
        if not span_counts:
            return 0.0, True
        if len(answer_counts) < len(span_counts):
            return 0.0, True
        return sum(a == b for a, b in zip(answer_counts[: len(span_counts)], span_counts)) / len(span_counts), True

    if task_type == "A1":
        ranges = answer_ranges(answer)
        if not ranges or not route:
            return 0.0, True
        spans = sorted(parsed.spans.get(route[0], ()))
        indices: List[int] = []
        if struct.get("ordinal") is not None:
            indices = [int(struct["ordinal"]) - 1]
        else:
            cluster_ids = struct.get("target_cluster_id", [])
            if not isinstance(cluster_ids, list):
                cluster_ids = [cluster_ids]
            indices = [
                int(match.group(1))
                for cluster_id in cluster_ids
                if (match := re.search(r"occ_(\d+)", str(cluster_id)))
            ]
        if not indices and len(ranges) == len(spans):
            indices = list(range(len(spans)))
        if not indices:
            indices = [0]
        if any(index < 0 or index >= len(spans) for index in indices):
            return 0.0, True
        return matched_span_score(ranges, [spans[index] for index in indices]), True

    if task_type in {"A5", "A5-gap", "A17", "A18"}:
        if not route:
            return 0.0, False
        spans = sorted(parsed.spans.get(route[0], ()))
        if not spans:
            return 0.0, True
        ordinal = struct.get("ordinal")
        cluster_match = re.search(r"occ_(\d+)", str(struct.get("target_cluster_id", "")))
        if cluster_match:
            ordinal = int(cluster_match.group(1)) + 1
        semantics = str(struct.get("duration_semantics", ""))
        if semantics == "total_union":
            predicted = first_number(answer, require_seconds=True)
            return dense_number_reward(
                predicted, _union_duration(spans), 0.2, exact_tolerance=0.100001
            ), True
        if semantics == "total":
            predicted = first_number(answer, require_seconds=True)
            total = sum(_span_duration(span) for span in spans)
            return dense_number_reward(
                predicted, total, 0.2, exact_tolerance=0.100001
            ), True
        if ordinal is not None and 1 <= int(ordinal) <= len(spans):
            target = spans[int(ordinal) - 1]
        elif semantics == "first" or len(spans) == 1:
            target = spans[0]
        elif semantics == "last":
            target = spans[-1]
        else:
            return 0.0, False
        predicted = first_number(answer, require_seconds=True)
        return dense_number_reward(
            predicted, _span_duration(target), 0.2, exact_tolerance=0.100001
        ), True

    if task_type == "A7":
        positive_audios = {
            audio_id for audio_id in route if parsed.spans.get(audio_id, ())
        }
        return set_f1(clip_set(answer), positive_audios), True

    if task_type == "A10":
        absent_audios = {
            audio_id for audio_id in route if not parsed.spans.get(audio_id, ())
        }
        return set_f1(clip_set(answer), absent_audios), True

    if task_type in {"A8", "A13"}:
        candidate_ids = list(route)
        if not candidate_ids:
            return 0.0, True
        if task_type == "A8":
            values = {audio_id: len(parsed.spans.get(audio_id, ())) for audio_id in candidate_ids}
        else:
            semantics = str(struct.get("duration_semantics", "max_single"))
            values = {}
            for audio_id in candidate_ids:
                spans = sorted(parsed.spans.get(audio_id, ()))
                if semantics == "max_single":
                    values[audio_id] = max((_span_duration(span) for span in spans), default=0.0)
                elif semantics == "total_union":
                    values[audio_id] = _union_duration(spans)
                elif semantics == "first":
                    values[audio_id] = _span_duration(spans[0]) if spans else 0.0
                elif semantics == "last":
                    values[audio_id] = _span_duration(spans[-1]) if spans else 0.0
                else:
                    values[audio_id] = sum(_span_duration(span) for span in spans)
        best = max(values.values())
        winners = {audio_id for audio_id, value in values.items() if math.isclose(value, best, abs_tol=0.05)}
        return set_f1(clip_set(answer), winners), True

    # A9 can omit historical winner spans by construction. A2/A4/A11 also
    # require event identity or comparison semantics absent from the span list.
    return 0.0, False


def consistency_component_reward(
    completion: str,
    task_type: str,
    answer_struct: Any,
    gold_route: Any = None,
    audio_lengths: Any = None,
) -> float:
    """Use a neutral score for tasks without a derivable consistency check."""
    if task_type not in CONSISTENCY_TASKS:
        return 1.0
    score, _ = consistency_reward(
        completion,
        task_type,
        answer_struct,
        gold_route,
        audio_lengths,
    )
    return score

