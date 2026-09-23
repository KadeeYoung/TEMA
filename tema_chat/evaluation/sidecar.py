"""Benchmark sidecar loading and interval decoding."""

from __future__ import annotations

import collections

import json

import re

from pathlib import Path

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

TIME_RUN_RE = re.compile(r"^(?:<a[0-9]>)+<f[0-9]>$")


DECIMAL_BOUNDARY_RE = re.compile(r"^\d+(?:\.\d+)?$")


AUDIO_LABEL_RE = re.compile(r"Audios\{(\d+)\}")


SPAN_BLOCK_RE = re.compile(r"Audios\{(\d+)\}\s*\[(.*?)\]", flags=re.S)


class DialogueError(ValueError):
    """A validation error that should quarantine the current dialogue."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def read_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DialogueError("bad_json", f"{path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise DialogueError("bad_json_type", f"{path}:{line_no} is not an object")
            yield line_no, obj


def decode_time_run(run: str) -> float:
    value = run.strip()
    if TIME_RUN_RE.fullmatch(value):
        anchors = "".join(re.findall(r"<a([0-9])>", value))
        frac = "".join(re.findall(r"<f([0-9])>", value))
        return float(f"{int(anchors)}.{frac[0]}")
    if DECIMAL_BOUNDARY_RE.fullmatch(value):
        return float(value)
    raise DialogueError("bad_time_boundary", f"invalid time boundary: {run!r}")


def split_interval_parts(body: str) -> List[str]:
    return [part.strip() for part in re.split(r"\s*[,;]\s*", body.strip()) if part.strip()]


def sidecar_key(obj: Mapping[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    return obj.get("group_id"), obj.get("ticket_id")


def load_sidecar(path: Optional[Path]) -> Dict[Tuple[Optional[str], Optional[str]], List[Dict[str, Any]]]:
    if path is None:
        return {}
    out: Dict[Tuple[Optional[str], Optional[str]], List[Dict[str, Any]]] = collections.defaultdict(list)
    for _line_no, row in read_jsonl(path):
        out[sidecar_key(row)].append(row)
    for rows in out.values():
        rows.sort(key=lambda x: int(x.get("turn_index", len(rows))))
    return dict(out)

