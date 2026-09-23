"""Optimizer grouping and vocabulary-row guards for dialogue SFT."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

TIME_TOKENS = [f"<a{i}>" for i in range(10)] + [f"<f{i}>" for i in range(10)]


def is_any_ate_name(name: str) -> bool:
    return name == "ate" or name.startswith("ate.") or ".ate." in name or name.endswith(".ate")


def is_ate_name(name: str) -> bool:
    """Return true for the active trainable ATE copy.

    ATE is also wrapped by PEFT ModulesToSaveWrapper, so the ``original_module``
    copy is kept frozen and the active ``modules_to_save`` copy receives grads.
    """
    if not is_any_ate_name(name):
        return False
    return ".original_module." not in name


def is_lora_name(name: str) -> bool:
    return "lora_" in name.lower()


def is_projector_name(name: str) -> bool:
    return ".audio_tower.proj" in name or "audio_projector" in name or "audio_tower.proj" in name


def is_audio_encoder_or_projector_base_name(name: str) -> bool:
    """Return true for frozen audio tower/projector base parameters.

    Projector LoRA is trainable and excluded; the base audio encoder and the
    projector base_layer must stay bitwise unchanged during SFT.
    """
    if "lora_" in name.lower():
        return False
    return ".audio_tower." in name or "audio_tower." in name


def is_vocab_weight_name(name: str) -> bool:
    if "lora_" in name.lower():
        return False
    return ("embed_tokens" in name or "lm_head" in name) and name.endswith(".weight")


def is_embedding_or_head_name(name: str) -> bool:
    """Return true for the active trainable vocab copy.

    PEFT ModulesToSaveWrapper exposes both ``original_module`` and the active
    ``modules_to_save.<adapter>`` copy. Forward uses the active copy, so SFT
    only optimizes that tensor. Non-PEFT direct modules remain trainable.
    """
    if not is_vocab_weight_name(name):
        return False
    return ".original_module." not in name


def unwrap_model(model):
    """Peel DDP/PEFT wrappers to the underlying Qwen2.5-Omni model."""
    m = model
    if hasattr(m, "module"):
        m = m.module
    if m.__class__.__name__ == "PeftModel" or hasattr(m, "base_model"):
        inner = getattr(m, "base_model", None)
        if inner is not None and hasattr(inner, "model"):
            m = inner.model
    return m


def active_modules_to_save(module):
    """Return the active module behind a PEFT ModulesToSaveWrapper."""
    if hasattr(module, "modules_to_save") and hasattr(module, "active_adapter"):
        return module.modules_to_save[module.active_adapter]
    return module


def resolve_time_token_ids(tokenizer, expected_ids: Optional[Sequence[int]] = None) -> List[int]:
    ids = []
    for token in TIME_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id < 0 or token_id == getattr(tokenizer, "unk_token_id", None):
            raise RuntimeError(f"time token missing from tokenizer: {token}")
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if encoded != [token_id]:
            raise RuntimeError(f"{token} must encode to exactly one token, got {encoded}")
        decoded = tokenizer.decode([token_id])
        if token not in decoded:
            raise RuntimeError(f"{token} failed decode round-trip: {decoded!r}")
        ids.append(int(token_id))
    if expected_ids is not None and list(map(int, expected_ids)) != ids:
        raise RuntimeError(f"time token id mismatch: expected={list(expected_ids)} actual={ids}")
    return ids


def freeze_for_sft(model, train_time_rows: bool = True) -> Dict[str, int]:
    """Apply the SFT train/freeze policy by parameter name."""
    counts = {"trainable": 0, "frozen": 0}
    for name, param in model.named_parameters():
        trainable = (
            is_lora_name(name)
            or is_ate_name(name)
            or (train_time_rows and is_embedding_or_head_name(name))
        )
        # Projector base weights stay frozen; projector LoRA is caught by is_lora_name.
        param.requires_grad = bool(trainable)
        counts["trainable" if trainable else "frozen"] += int(param.numel())
    return counts


@dataclass
class TimeRowMaskHandle:
    handles: List[Any]
    time_token_ids: List[int]

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def register_time_row_gradient_mask(model, time_token_ids: Sequence[int]) -> TimeRowMaskHandle:
    """Mask embed/lm_head gradients so only the 20 time-token rows can update."""
    ids = torch.tensor(list(map(int, time_token_ids)), dtype=torch.long)
    handles = []
    for name, param in model.named_parameters():
        if not param.requires_grad or not is_embedding_or_head_name(name):
            continue

        def make_hook(time_ids: torch.Tensor):
            def hook(grad: torch.Tensor) -> torch.Tensor:
                if grad is None:
                    return grad
                masked = torch.zeros_like(grad)
                local_ids = time_ids.to(device=grad.device)
                masked.index_copy_(0, local_ids, grad.index_select(0, local_ids))
                return masked

            return hook

        handles.append(param.register_hook(make_hook(ids)))
    return TimeRowMaskHandle(handles=handles, time_token_ids=list(map(int, time_token_ids)))



class FrozenParameterGuard:
    """Hash selected frozen parameters and assert bitwise invariance."""

    def __init__(self, model, predicate, label: str = "frozen_parameters") -> None:
        self.label = label
        self.snapshots: Dict[str, Dict[str, Any]] = {}
        for name, param in model.named_parameters():
            if predicate(name):
                self.snapshots[name] = self._fingerprint(param)

    def _hash_tensor(self, tensor: torch.Tensor) -> str:
        h = hashlib.sha256()
        h.update(str(tuple(tensor.shape)).encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        chunk = tensor.detach().contiguous()
        if chunk.device.type != "cpu":
            chunk = chunk.cpu()
        try:
            data = chunk.view(torch.uint8).numpy().tobytes()
        except Exception:
            import io

            buf = io.BytesIO()
            torch.save(chunk, buf)
            data = buf.getvalue()
        h.update(data)
        return h.hexdigest()

    def _fingerprint(self, param: torch.nn.Parameter) -> Dict[str, Any]:
        return {
            "shape": list(param.shape),
            "dtype": str(param.dtype),
            "numel": int(param.numel()),
            "requires_grad": bool(param.requires_grad),
            "sha256": self._hash_tensor(param),
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "checked_tensors": len(self.snapshots),
            "tensors": {
                name: {k: v for k, v in meta.items() if k != "sha256"}
                for name, meta in self.snapshots.items()
            },
        }

    @torch.no_grad()
    def check(self, model) -> Dict[str, Any]:
        current_params = dict(model.named_parameters())
        checks: Dict[str, Dict[str, Any]] = {}
        changed: List[str] = []
        missing: List[str] = []
        trainable: List[str] = []
        for name, expected in self.snapshots.items():
            param = current_params.get(name)
            if param is None:
                missing.append(name)
                continue
            current = self._fingerprint(param)
            ok = current["sha256"] == expected["sha256"]
            if not ok:
                changed.append(name)
            if current["requires_grad"]:
                trainable.append(name)
            checks[name] = {
                "ok": ok,
                "shape": current["shape"],
                "dtype": current["dtype"],
                "numel": current["numel"],
                "requires_grad": current["requires_grad"],
                "before_sha256": expected["sha256"],
                "after_sha256": current["sha256"],
            }
        if changed or missing or trainable:
            raise AssertionError(
                f"{self.label} changed={changed[:8]} missing={missing[:8]} trainable={trainable[:8]}"
            )
        return {"status": "pass", "label": self.label, "checked_tensors": len(checks), "checks": checks}

    @torch.no_grad()
    def assert_unchanged(self, model) -> None:
        self.check(model)


class VocabularyRowGuard:
    """Hash non-time vocabulary rows and assert bitwise invariance after a step."""

    def __init__(self, model, time_token_ids: Sequence[int]) -> None:
        self.time_token_ids = set(map(int, time_token_ids))
        self.snapshots: Dict[str, Dict[str, Any]] = {}
        for name, param in model.named_parameters():
            if is_vocab_weight_name(name):
                self.snapshots[name] = self._fingerprint(param)

    def _non_time_spans(self, rows: int) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        start = 0
        for token_id in sorted(i for i in self.time_token_ids if 0 <= i < rows):
            if start < token_id:
                spans.append((start, token_id))
            start = token_id + 1
        if start < rows:
            spans.append((start, rows))
        return spans

    def _hash_non_time_rows(self, tensor: torch.Tensor, spans: Sequence[Tuple[int, int]]) -> str:
        h = hashlib.sha256()
        h.update(str(tuple(tensor.shape)).encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        for start, end in spans:
            if start >= end:
                continue
            chunk = tensor.detach()[start:end].contiguous()
            if chunk.device.type != "cpu":
                chunk = chunk.cpu()
            try:
                data = chunk.view(torch.uint8).numpy().tobytes()
            except Exception:
                import io

                buf = io.BytesIO()
                torch.save(chunk, buf)
                data = buf.getvalue()
            h.update(start.to_bytes(8, "little", signed=False))
            h.update(end.to_bytes(8, "little", signed=False))
            h.update(data)
        return h.hexdigest()

    def _fingerprint(self, param: torch.nn.Parameter) -> Dict[str, Any]:
        rows = int(param.shape[0])
        spans = self._non_time_spans(rows)
        return {
            "shape": list(param.shape),
            "dtype": str(param.dtype),
            "non_time_rows": int(sum(end - start for start, end in spans)),
            "sha256": self._hash_non_time_rows(param, spans),
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "checked_tensors": len(self.snapshots),
            "tensors": {
                name: {k: v for k, v in meta.items() if k != "sha256"}
                for name, meta in self.snapshots.items()
            },
        }

    @torch.no_grad()
    def check(self, model) -> Dict[str, Any]:
        current_params = dict(model.named_parameters())
        checks: Dict[str, Dict[str, Any]] = {}
        changed: List[str] = []
        missing: List[str] = []
        for name, expected in self.snapshots.items():
            param = current_params.get(name)
            if param is None:
                missing.append(name)
                continue
            current = self._fingerprint(param)
            ok = current["sha256"] == expected["sha256"]
            if not ok:
                changed.append(name)
            checks[name] = {
                "ok": ok,
                "shape": current["shape"],
                "dtype": current["dtype"],
                "non_time_rows": current["non_time_rows"],
                "before_sha256": expected["sha256"],
                "after_sha256": current["sha256"],
            }
        if changed or missing:
            raise AssertionError(
                f"non-time vocabulary rows changed={changed[:8]} missing={missing[:8]}"
            )
        return {"status": "pass", "checked_tensors": len(checks), "checks": checks}

    @torch.no_grad()
    def assert_unchanged(self, model) -> None:
        self.check(model)


class TrainableUpdateGuard:
    """Hash each SFT optimizer group's live values and detect post-step change.

    DeepSpeed ZeRO does not reliably populate ``param.grad`` at the point HF
    Trainer callbacks fire (gradients live in internal, possibly partitioned,
    buffers), so a gradient-norm audit taken from ``param.grad`` silently reads
    zero under multi-GPU ZeRO training even though the optimizer step is real.
    This guard instead hashes actual parameter values before and after the
    optimizer step, which is correct regardless of how DeepSpeed manages
    gradients internally.
    """

    def __init__(self, model, time_token_ids: Sequence[int]) -> None:
        self.time_token_ids = list(map(int, time_token_ids))
        self._groups: Dict[str, List[str]] = {"lora": [], "projector_lora": [], "ate": [], "time_rows": []}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if is_embedding_or_head_name(name):
                self._groups["time_rows"].append(name)
            elif is_ate_name(name):
                self._groups["ate"].append(name)
            elif is_lora_name(name) and is_projector_name(name):
                self._groups["projector_lora"].append(name)
            elif is_lora_name(name):
                self._groups["lora"].append(name)
        self.before = self._hash_groups(model)

    def _tensor_bytes(self, name: str, param: torch.nn.Parameter) -> bytes:
        tensor = param.detach()
        if is_embedding_or_head_name(name):
            ids = torch.tensor(self.time_token_ids, device=tensor.device, dtype=torch.long)
            tensor = tensor.index_select(0, ids)
        tensor = tensor.contiguous()
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        try:
            return tensor.view(torch.uint8).numpy().tobytes()
        except Exception:
            import io

            buf = io.BytesIO()
            torch.save(tensor, buf)
            return buf.getvalue()

    def _hash_groups(self, model) -> Dict[str, str]:
        params = dict(model.named_parameters())
        out: Dict[str, str] = {}
        for group, names in self._groups.items():
            h = hashlib.sha256()
            for name in sorted(names):
                param = params.get(name)
                if param is None:
                    continue
                h.update(name.encode("utf-8"))
                h.update(self._tensor_bytes(name, param))
            out[group] = h.hexdigest()
        return out

    def summary(self) -> Dict[str, Any]:
        return {
            "method": "weight_diff",
            "group_tensor_counts": {g: len(names) for g, names in self._groups.items()},
        }

    @torch.no_grad()
    def check(self, model) -> Dict[str, Any]:
        after = self._hash_groups(model)
        result: Dict[str, Any] = {
            "method": "weight_diff",
            "group_tensor_counts": {g: len(names) for g, names in self._groups.items()},
        }
        for group, names in self._groups.items():
            result[f"{group}_nonzero"] = bool(names) and after[group] != self.before[group]
        return result


def _add_group(
    groups: List[Dict[str, Any]],
    names: List[str],
    params: List[torch.nn.Parameter],
    lr: float,
    weight_decay: float,
    tag: str,
) -> None:
    if not params:
        return
    groups.append(
        {
            "params": params,
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "sft_group": tag,
            "param_names": names,
        }
    )


def build_sft_optimizer_groups(
    model,
    lr_lora: float = 1e-5,
    lr_projector_lora: float = 5e-6,
    lr_ate: float = 2e-5,
    lr_time_rows: float = 2e-5,
    weight_decay: float = 0.0,
    allow_unclassified_full_params: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return optimizer groups and a JSON-serializable audit.

    The embed/lm_head parameters are full matrices, but gradient hooks restrict
    updates to time-token rows. Their optimizer group always uses weight_decay=0.
    """
    buckets: Dict[str, List[Tuple[str, torch.nn.Parameter]]] = {
        "lora": [],
        "projector_lora": [],
        "ate": [],
        "time_rows": [],
        "unclassified_full": [],
    }
    seen: Dict[int, str] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        pid = id(param)
        if pid in seen:
            raise RuntimeError(f"parameter appears twice in named_parameters: {name} and {seen[pid]}")
        seen[pid] = name
        if is_embedding_or_head_name(name):
            buckets["time_rows"].append((name, param))
        elif is_ate_name(name):
            buckets["ate"].append((name, param))
        elif is_lora_name(name) and is_projector_name(name):
            buckets["projector_lora"].append((name, param))
        elif is_lora_name(name):
            buckets["lora"].append((name, param))
        else:
            buckets["unclassified_full"].append((name, param))

    if buckets["unclassified_full"] and not allow_unclassified_full_params:
        names = [name for name, _ in buckets["unclassified_full"][:20]]
        raise RuntimeError(f"unexpected full trainable parameters: {names}")

    groups: List[Dict[str, Any]] = []
    _add_group(
        groups,
        [x[0] for x in buckets["lora"]],
        [x[1] for x in buckets["lora"]],
        lr_lora,
        weight_decay,
        "lora",
    )
    _add_group(
        groups,
        [x[0] for x in buckets["projector_lora"]],
        [x[1] for x in buckets["projector_lora"]],
        lr_projector_lora,
        weight_decay,
        "projector_lora",
    )
    _add_group(
        groups,
        [x[0] for x in buckets["ate"]],
        [x[1] for x in buckets["ate"]],
        lr_ate,
        0.0,
        "ate",
    )
    _add_group(
        groups,
        [x[0] for x in buckets["time_rows"]],
        [x[1] for x in buckets["time_rows"]],
        lr_time_rows,
        0.0,
        "time_rows",
    )
    _add_group(
        groups,
        [x[0] for x in buckets["unclassified_full"]],
        [x[1] for x in buckets["unclassified_full"]],
        lr_lora,
        weight_decay,
        "unclassified_full",
    )

    group_param_ids: set[int] = set()
    for group in groups:
        for param in group["params"]:
            pid = id(param)
            if pid in group_param_ids:
                raise RuntimeError("parameter duplicated across optimizer groups")
            group_param_ids.add(pid)

    audit = {
        "total_params": int(sum(p.numel() for p in model.parameters())),
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "groups": [
            {
                "name": group["sft_group"],
                "lr": group["lr"],
                "weight_decay": group["weight_decay"],
                "n_tensors": len(group["params"]),
                "n_params": int(sum(p.numel() for p in group["params"])),
                "effective_trainable_params": int(
                    sum(
                        min(len(TIME_TOKENS), int(p.shape[0])) * int(p.numel() // max(1, int(p.shape[0])))
                        if group["sft_group"] == "time_rows" and p.ndim >= 1
                        else p.numel()
                        for p in group["params"]
                    )
                ),
                "param_names": group["param_names"],
            }
            for group in groups
        ],
    }
    return groups, audit


def write_optimizer_audit(path: Path, audit: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2, sort_keys=True)
