"""State, gradient, and invariance callbacks for temporal initialization."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from temporal_init.ate_module import monitor_ate
from temporal_init.checkpoint_state import save_temporal_init_extra
from sft.model.optimizer_groups import (
    FrozenParameterGuard,
    VocabularyRowGuard,
    is_ate_name,
    is_audio_encoder_or_projector_base_name,
    is_embedding_or_head_name,
    is_lora_name,
    is_projector_name,
    resolve_time_token_ids,
)

try:
    from swift.callbacks import TrainerCallback
except Exception:  # pragma: no cover
    from transformers import TrainerCallback


def _active_ate(model):
    module = model
    if hasattr(module, "module"):
        module = module.module
    if hasattr(module, "base_model") and hasattr(module.base_model, "model"):
        module = module.base_model.model
    ate = getattr(module.thinker, "ate", None)
    if hasattr(ate, "modules_to_save"):
        ate = ate.modules_to_save[ate.active_adapter]
    return ate


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().contiguous().cpu()
    try:
        return value.view(torch.uint8).numpy().tobytes()
    except Exception:
        import io

        buffer = io.BytesIO()
        torch.save(value, buffer)
        return buffer.getvalue()


def _effective_initialization_hash(model, time_token_ids: List[int]) -> Dict[str, object]:
    digest = hashlib.sha256()
    entries = []
    ids = torch.tensor(time_token_ids, dtype=torch.long)
    for name, parameter in sorted(model.named_parameters()):
        if not parameter.requires_grad:
            continue
        value = parameter.detach()
        effective_numel = int(value.numel())
        if is_embedding_or_head_name(name):
            value = value.index_select(0, ids.to(value.device))
            effective_numel = int(value.numel())
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(_tensor_bytes(value))
        entries.append({"name": name, "shape": list(parameter.shape), "effective_numel": effective_numel})
    return {"sha256": digest.hexdigest(), "entries": entries}


def _norm(tensors: Iterable[torch.Tensor]) -> float:
    values = [value.detach().float().norm() for value in tensors if value is not None]
    return float(torch.stack(values).norm().item()) if values else 0.0


class TemporalInitStateCallback(TrainerCallback):
    def __init__(self, args=None, trainer=None):
        try:
            super().__init__(args, trainer)
        except TypeError:
            super().__init__()
            self.args = args
            self.trainer = trainer
        self._row_guard = None
        self._audio_guard = None
        self._guard_checked = False

    @staticmethod
    def _use_time_tokens() -> bool:
        return os.environ.get("TEMA_TEMPORAL_INIT_USE_TIME_TOKENS", "1").lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _use_ate() -> bool:
        return os.environ.get("TEMA_TEMPORAL_INIT_USE_ATE", "1").lower() in {"1", "true", "yes", "on"}

    def _report_dir(self, args) -> Path:
        path = Path(args.output_dir) / "reports"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _write(self, args, name: str, payload) -> None:
        if int(os.environ.get("RANK", "0")) != 0:
            return
        with (self._report_dir(args) / name).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)

    def _tokenizer(self, kwargs):
        template = getattr(self.trainer, "template", None)
        if template is not None:
            return template.tokenizer
        processor = kwargs.get("processing_class") or kwargs.get("tokenizer")
        return getattr(processor, "tokenizer", processor)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        use_time_tokens = self._use_time_tokens()
        use_ate = self._use_ate()
        tokenizer = self._tokenizer(kwargs)
        time_ids = resolve_time_token_ids(tokenizer) if use_time_tokens else []
        effective = _effective_initialization_hash(model, time_ids)
        ate = _active_ate(model)
        out_weight_zero = (
            bool(torch.count_nonzero(ate.out_proj.weight.detach()).item() == 0)
            if ate is not None
            else None
        )
        out_bias_zero = (
            bool(torch.count_nonzero(ate.out_proj.bias.detach()).item() == 0)
            if ate is not None
            else None
        )
        resume_requested = os.environ.get("TEMA_TEMPORAL_INIT_RESUME_ENABLED") == "1"
        trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        payload = {
            "phase": "resume" if resume_requested else "base model",
            "base_model": os.environ.get("TEMA_TEMPORAL_INIT_BASE_MODEL"),
            "loaded_training_checkpoint": os.environ.get("TEMA_TEMPORAL_INIT_RESUME_CHECKPOINT"),
            "use_time_tokens": use_time_tokens,
            "use_ate": use_ate,
            "tokenizer_size": len(tokenizer),
            "time_token_ids": time_ids,
            "ate_out_proj_weight_zero": out_weight_zero,
            "ate_out_proj_bias_zero": out_bias_zero,
            "effective_trainable_initialization": effective,
            "trainable_tensor_count": len(trainable_names),
            "trainable_parameter_names": trainable_names,
            "status": "pass" if resume_requested or not use_ate or (out_weight_zero and out_bias_zero) else "fail",
        }
        self._write(args, "base_model_initialization.json", payload)
        if payload["status"] != "pass":
            raise AssertionError("ATE is not zero-init at base model")

        self._row_guard = VocabularyRowGuard(model, time_ids) if use_time_tokens else None
        self._audio_guard = FrozenParameterGuard(
            model,
            is_audio_encoder_or_projector_base_name,
            label="frozen_audio_encoder_projector_base",
        )
        if self._row_guard is not None:
            self._write(args, "vocabulary_row_guard.json", {"status": "initialized", **self._row_guard.summary()})
        self._write(args, "frozen_audio_guard.json", {"status": "initialized", **self._audio_guard.summary()})

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        time_ids = resolve_time_token_ids(self._tokenizer(kwargs)) if self._use_time_tokens() else []
        ids = torch.tensor(time_ids, dtype=torch.long)
        buckets = {"llm": [], "projector": [], "ate": [], "time_rows": []}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            grad = parameter.grad
            if is_embedding_or_head_name(name) and time_ids:
                buckets["time_rows"].append(grad.index_select(0, ids.to(grad.device)))
            elif is_ate_name(name):
                buckets["ate"].append(grad)
            elif is_lora_name(name) and is_projector_name(name):
                buckets["projector"].append(grad)
            elif is_lora_name(name):
                buckets["llm"].append(grad)
        for name, tensors in buckets.items():
            self.trainer.custom_metrics["train"][f"grad_norm/{name}"].update(
                torch.tensor(_norm(tensors), device=args.device)
            )
        ate = _active_ate(model)
        if ate is not None:
            ate_norm = monitor_ate(ate)["ate_norm_mean"]
            self.trainer.custom_metrics["train"]["ate/output_norm_mean"].update(
                torch.tensor(ate_norm, device=args.device)
            )

    def on_optimizer_step(self, args, state, control, model=None, **kwargs):
        if model is None or self._guard_checked:
            return
        try:
            row_report = self._row_guard.check(model) if self._row_guard is not None else None
            audio_report = self._audio_guard.check(model)
        except Exception as exc:
            self._write(args, "protection_gate.json", {"status": "fail", "error": str(exc)})
            raise
        self._guard_checked = True
        self._write(
            args,
            "protection_gate.json",
            {"status": "pass", "checked_after_optimizer_step": int(state.global_step) + 1},
        )
        if row_report is not None:
            self._write(args, "vocabulary_row_guard.json", row_report)
        self._write(args, "frozen_audio_guard.json", audio_report)

    def on_save(self, args, state, control, model=None, **kwargs):
        if model is None or int(os.environ.get("RANK", "0")) != 0:
            return
        checkpoint = Path(args.output_dir) / f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        save_temporal_init_extra(model, str(checkpoint), include_vocab=self._use_time_tokens())
        if hasattr(self.trainer, "tc_trace_summary"):
            self._write(args, "positive_batch_trace.json", self.trainer.tc_trace_summary())

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if hasattr(self.trainer, "tc_trace_summary"):
            self._write(args, "positive_batch_trace.json", self.trainer.tc_trace_summary())
