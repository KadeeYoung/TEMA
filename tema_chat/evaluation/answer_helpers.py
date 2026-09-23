"""Shared answer parsing constants; API judging is implemented by tema_chat.judge."""

from __future__ import annotations

import re

from typing import Any

TIME_TASKS = {"A1", "A5", "A5-gap", "A17", "A18"}


ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.I | re.S)


ANSWER_OPEN_RE = re.compile(r"<answer>\s*(.*)$", re.I | re.S)


THINK_RE = re.compile(r"<think>.*?</think>", re.I | re.S)


def extract_final_answer(text: Any) -> str:
    """Extract only the last answer payload, tolerating a truncated closing tag."""
    if not isinstance(text, str):
        return ""
    matches = ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip()
    opened = ANSWER_OPEN_RE.findall(text)
    if opened:
        return opened[-1].strip()
    without_think = THINK_RE.sub("", text).strip()
    return without_think or text.strip()

