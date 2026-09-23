"""Anchor/offset time-token surgery + timestamp (de)serialization for Qwen2.5-Omni.

Adds 20 special tokens (10 anchor ``<a0>..<a9>``, 10 offset ``<f0>..<f9>``), resizes the
embedding/LM head, and initializes the new rows per the TimeAudio rules (v4.2 plan 2.2):

    anchor  <aN> embedding  <- clone of digit token "N"
    offset  <fN> embedding  <- ( emb("N") + emb(".") ) / 2
    LM head rows are synchronized the same way (skipped automatically if the head is tied).

Variable-length timestamp format (v4.2 plan 4.3):
    0.0-9.9 s   -> 2 tokens   2.5  -> <a2><f5>
    10-99.9 s   -> 3 tokens   12.5 -> <a1><a2><f5>
    100-999.9 s -> 4 tokens   120.5-> <a1><a2><a0><f5>

NOTE: this project uses the *no-underscore* convention ``<a0>`` / ``<f0>`` (NOT TimeAudio's
``<a_0>`` / ``<f_0>``). Every part of the pipeline imports ``TIME_TOKENS`` from here so the
naming can never diverge.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

import torch

ANCHOR_TOKENS: List[str] = [f"<a{i}>" for i in range(10)]
OFFSET_TOKENS: List[str] = [f"<f{i}>" for i in range(10)]
TIME_TOKENS: List[str] = ANCHOR_TOKENS + OFFSET_TOKENS  # anchors first (contiguity check)

MAX_TIME: float = 120.0

# ----------------------------------------------------------------------------- #
# Serialization: float seconds  <->  <aX><fY> token string
# ----------------------------------------------------------------------------- #


def format_time(t: float) -> str:
    """Seconds -> anchor/offset token string. Rounds to 0.1 s, clamps to [0, MAX_TIME].

    2.5 -> '<a2><f5>'; 12.5 -> '<a1><a2><f5>'; 25.0 -> '<a2><a5><f0>'; 120.5 -> clamp.
    """
    t = max(0.0, min(MAX_TIME, round(float(t), 1)))
    whole = int(t)
    frac = int(round((t - whole) * 10))
    if frac == 10:  # e.g. 9.95 rounding edge
        whole += 1
        frac = 0
    anchors = "".join(f"<a{d}>" for d in str(whole))  # variable-length integer part
    return anchors + f"<f{frac}>"


def format_span(t_s: float, t_e: float) -> str:
    """One interval -> '<a2><f5>-<a5><f0>'."""
    return f"{format_time(t_s)}-{format_time(t_e)}"


def format_audio_spans(audio_id: int, spans: List[Tuple[float, float]]) -> str:
    """A single audio's spans -> 'Audios{1}[<a2><f5>-<a5><f0>; <a8><f0>-<a9><f0>]'."""
    body = "; ".join(format_span(s, e) for s, e in spans)
    return f"Audios{{{audio_id}}}[{body}]"


def render_span_block(audio_to_spans: Dict[int, List[Tuple[float, float]]]) -> str:
    """Full <span>...</span> content. Different audios joined by '; '."""
    parts = [
        format_audio_spans(aid, audio_to_spans[aid]) for aid in sorted(audio_to_spans)
    ]
    return f"<span>{'; '.join(parts)}</span>"


# ----------------------------------------------------------------------------- #
# Parsing: token string -> floats (detokenizer). Ported from TimeAudio
# models/utils.py:decode_time_answer_v3, adapted to the no-underscore convention.
# ----------------------------------------------------------------------------- #

_TOKEN = r"<(?:a|f)\d>"
_TOKEN_SEQ = re.compile(rf"({_TOKEN}(?:\s*{_TOKEN})*)")
_AUDIO_BLOCK = re.compile(r"Audios\{(\d+)\}\s*\[(.*?)\]", flags=re.S)
_DECIMAL_BOUNDARY = re.compile(r"^\d+(?:\.\d+)?$")


def _decode_token_run(run: str) -> float:
    ints = "".join(re.findall(r"<a(\d)>", run)) or "0"
    fracs = "".join(re.findall(r"<f(\d)>", run)) or "0"
    return float(f"{ints}.{fracs}")


def detokenize_time(text: str) -> str:
    """Replace every ``<aX><fY>`` run with a decimal string (human-readable display)."""
    return _TOKEN_SEQ.sub(lambda m: f"{_decode_token_run(m.group(0)):.1f}", text)


def decode_time_boundary(text: str) -> float:
    """Decode either the project time-token format or an ordinary decimal.

    Supporting both forms here keeps evaluation semantics identical for the
    no-special-time-token ablation without relaxing the normal data builder,
    which still rejects decimal boundaries in its standard mode.
    """
    value = text.strip()
    token_runs = _TOKEN_SEQ.findall(value)
    if token_runs and _TOKEN_SEQ.fullmatch(value):
        return _decode_token_run(value)
    if _DECIMAL_BOUNDARY.fullmatch(value):
        return float(value)
    raise ValueError(f"invalid time boundary: {text!r}")


def parse_span_block(text: str) -> Dict[int, List[Tuple[float, float]]]:
    """Parse '<span> Audios{1}[<a2><f1>-<a2><f5>; <a4><f0>-<a4><f3>] </span>' ->
    {1: [(2.1, 2.5), (4.0, 4.3)]}. Robust to whitespace and to junk between blocks.
    Intervals that fail to parse are skipped (used by lenient eval)."""
    out: Dict[int, List[Tuple[float, float]]] = {}
    for m in _AUDIO_BLOCK.finditer(text):
        aid = int(m.group(1))
        body = m.group(2)
        spans: List[Tuple[float, float]] = []
        for part in body.split(";"):
            part = part.strip()
            if "-" not in part:
                continue
            lo, hi = part.split("-", 1)
            try:
                spans.append((decode_time_boundary(lo), decode_time_boundary(hi)))
            except ValueError:
                continue
        if spans:
            out.setdefault(aid, []).extend(spans)
    return out


# ----------------------------------------------------------------------------- #
# Tokenizer / embedding surgery
# ----------------------------------------------------------------------------- #


def _embed_and_head(model):
    """Return (tokenizer-agnostic) the thinker's input embedding and lm_head modules.

    Works on the top-level Qwen2_5OmniForConditionalGeneration (before LoRA wrapping)."""
    thinker = model.thinker
    embed = thinker.get_input_embeddings()  # == thinker.model.embed_tokens
    # NOTE: Qwen2_5OmniThinkerForConditionalGeneration.get_output_embeddings() returns
    # None, so reach the head via the attribute directly.
    lm_head = getattr(thinker, "lm_head", None)
    if lm_head is None:
        lm_head = thinker.get_output_embeddings()
    return thinker, embed, lm_head


def add_time_tokens(model, tokenizer) -> dict:
    """Add the 20 time tokens and initialize their embed/lm_head rows.

    Qwen2.5-Omni's vocabulary is *padded* (embedding has 152064 rows while only ~151665
    are real tokens), so the 20 new ids fit inside the existing matrices and NO resize is
    needed in the common case. We only resize if the new vocab would exceed the embedding
    capacity. Must be called BEFORE LoRA wrapping. Returns bookkeeping for verification.
    """
    n_before = len(tokenizer)
    added = tokenizer.add_special_tokens({"additional_special_tokens": TIME_TOKENS})
    assert added == 20, f"expected to add 20 tokens, added {added}"

    thinker, embed, lm_head = _embed_and_head(model)
    capacity = embed.weight.shape[0]
    if len(tokenizer) > capacity:
        # genuinely need more rows; mean_resizing=False keeps new rows clean for the
        # exact-equality checks below.
        thinker.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        thinker, embed, lm_head = _embed_and_head(model)
    # else: ids fit within the padded matrices; initialize the existing rows in place.

    tied = (lm_head is not None) and (
        lm_head.weight.data_ptr() == embed.weight.data_ptr()
    )

    def tid(s: str) -> int:
        return tokenizer.convert_tokens_to_ids(s)

    dot_id = tid(".")
    assert dot_id is not None and dot_id >= 0, "'.' not a single token?"

    with torch.no_grad():
        for i in range(10):
            digit_id = tid(str(i))
            a_id = tid(f"<a{i}>")
            f_id = tid(f"<f{i}>")
            anchor_vec = embed.weight.data[digit_id].clone()  # rule 1
            offset_vec = (
                embed.weight.data[digit_id] + embed.weight.data[dot_id]
            ) / 2.0  # rule 2
            embed.weight.data[a_id] = anchor_vec
            embed.weight.data[f_id] = offset_vec
            if not tied and lm_head is not None:  # rule 3 (only if untied)
                lm_head.weight.data[a_id] = lm_head.weight.data[digit_id].clone()
                lm_head.weight.data[f_id] = (
                    lm_head.weight.data[digit_id] + lm_head.weight.data[dot_id]
                ) / 2.0

    new_ids = sorted(tid(t) for t in TIME_TOKENS)
    return {
        "new_ids": new_ids,
        "n_before": n_before,
        "n_after": len(tokenizer),
        "embed_capacity": int(capacity),
        "resized": len(tokenizer) > capacity,
        "tied_lm_head": bool(tied),
        "lm_head_found": lm_head is not None,
        "dot_id": int(dot_id),
        "anchor_ids": {i: tid(f"<a{i}>") for i in range(10)},
        "offset_ids": {i: tid(f"<f{i}>") for i in range(10)},
    }


@torch.no_grad()
def verify_base_model_tokens(model, tokenizer) -> None:
    """Assert the embedding/LM-head init rules and token-id contiguity (v4.2 plan 5.1)."""
    thinker, embed, lm_head = _embed_and_head(model)

    def tid(s: str) -> int:
        return tokenizer.convert_tokens_to_ids(s)

    ids = [tid(t) for t in TIME_TOKENS]
    assert ids == list(range(min(ids), min(ids) + 20)), f"ids not contiguous: {ids}"
    assert max(ids) == len(tokenizer) - 1, "new tokens not at the vocab end"

    assert torch.equal(
        embed.weight.data[tid("<a3>")], embed.weight.data[tid("3")]
    ), "emb(<a3>) != emb('3')"
    expected_f5 = (embed.weight.data[tid("5")] + embed.weight.data[tid(".")]) / 2.0
    assert torch.equal(
        embed.weight.data[tid("<f5>")], expected_f5
    ), "emb(<f5>) != (emb('5')+emb('.'))/2"

    # LM head rows must be synchronized too (untied head); otherwise generation can't
    # learn the new-token semantics.
    tied = lm_head.weight.data_ptr() == embed.weight.data_ptr()
    if not tied:
        assert torch.equal(
            lm_head.weight.data[tid("<a3>")], lm_head.weight.data[tid("3")]
        ), "lm_head(<a3>) != lm_head('3')"
        exp_h_f5 = (
            lm_head.weight.data[tid("5")] + lm_head.weight.data[tid(".")]
        ) / 2.0
        assert torch.equal(
            lm_head.weight.data[tid("<f5>")], exp_h_f5
        ), "lm_head(<f5>) != (lm_head('5')+lm_head('.'))/2"
