"""Trainer callbacks for SFT extra-state handling."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sft.model.load_temporal_init import adapter_target_modules, load_adapter_config
from sft.model.optimizer_groups import (
    FrozenParameterGuard,
    TrainableUpdateGuard,
    VocabularyRowGuard,
    active_modules_to_save,
    build_sft_optimizer_groups,
    freeze_for_sft,
    is_ate_name,
    is_audio_encoder_or_projector_base_name,
    is_embedding_or_head_name,
    is_lora_name,
    is_projector_name,
    resolve_time_token_ids,
    unwrap_model,
)
from sft.model.save_extra_state import load_sft_extra, save_sft_extra

try:
    from swift.callbacks import TrainerCallback
except Exception:  # pragma: no cover
    from transformers import TrainerCallback


class SftStateCallback(TrainerCallback):
    """Load temporal initialization extra state once, then save SFT extra state per checkpoint."""

    def __init__(
        self,
        args=None,
        trainer=None,
        temporal_init_extra: Optional[str] = None,
        base_model_path: Optional[str] = None,
        temporal_init_checkpoint: Optional[str] = None,
        project_root: Optional[str] = None,
    ) -> None:
        try:
            super().__init__(args, trainer)
        except TypeError:
            super().__init__()
            self.args = args
            self.trainer = trainer
        self.temporal_init_extra = temporal_init_extra or os.environ.get("SFT_TEMPORAL_INIT_EXTRA")
        self.base_model_path = base_model_path or os.environ.get("SFT_BASE_MODEL", "")
        self.sft_checkpoint = os.environ.get("SFT_INIT_FROM_SFT", "")
        self.temporal_init_checkpoint = temporal_init_checkpoint or self.sft_checkpoint or os.environ.get("SFT_INIT_FROM_TEMPORAL_INIT", "")
        self.project_root = project_root or os.environ.get("SFT_PROJECT_ROOT", str(PROJECT_ROOT))
        self._loaded = False
        self._row_guard: Optional[VocabularyRowGuard] = None
        self._frozen_audio_guard: Optional[FrozenParameterGuard] = None
        self._update_guard: Optional[TrainableUpdateGuard] = None
        self._row_guard_steps = int(os.environ.get("SFT_ROW_GUARD_STEPS", "1"))
        self._last_grad_report: Optional[Dict[str, Any]] = None
        self._resume_mode = os.environ.get("SFT_RESUME_MODE") == "1"
        self._use_time_tokens = os.environ.get("SFT_USE_TIME_TOKENS", "1").lower() in {
            "1", "true", "yes", "on"
        }
        self._use_ate = os.environ.get("SFT_USE_ATE", "1").lower() in {
            "1", "true", "yes", "on"
        }

    @staticmethod
    def _tensor_norm(tensors) -> float:
        values = []
        for tensor in tensors:
            if tensor is not None and tensor.numel() > 0:
                values.append(tensor.detach().float().norm())
        if not values:
            return 0.0
        return float(torch.stack(values).norm().item())

    def _grad_based_report(self, model, time_token_ids, step: int) -> Dict[str, Any]:
        """Gradient-norm audit read straight from ``param.grad``.

        Reliable and precise for plain PyTorch/single-GPU runs (no DeepSpeed),
        since it inspects the raw pre-optimizer-step gradient rather than a
        post-step, bf16-rounded weight delta. Under DeepSpeed ZeRO this is
        unreliable (see ``TrainableUpdateGuard``), so treat this as a
        best-effort supplementary signal, not the sole source of truth.
        """
        buckets: Dict[str, list] = {"lora": [], "projector_lora": [], "ate": [], "time_rows": []}
        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            grad = param.grad.detach()
            if is_embedding_or_head_name(name):
                ids = torch.tensor(list(map(int, time_token_ids)), device=grad.device, dtype=torch.long)
                buckets["time_rows"].append(grad.index_select(0, ids))
            elif is_ate_name(name):
                buckets["ate"].append(grad)
            elif is_lora_name(name) and is_projector_name(name):
                buckets["projector_lora"].append(grad)
            elif is_lora_name(name):
                buckets["lora"].append(grad)
        group_norms = {name: self._tensor_norm(values) for name, values in buckets.items()}
        report = {"step": int(step), "group_grad_norms": group_norms}
        for group, norm in group_norms.items():
            report[f"{group}_nonzero"] = norm > 0.0
        return report

    def _processor(self, kwargs):
        if getattr(self, "trainer", None) is not None:
            template = getattr(self.trainer, "template", None)
            if template is not None and getattr(template, "tokenizer", None) is not None:
                return template.tokenizer
        processing_class = kwargs.get("processing_class")
        if processing_class is not None:
            return processing_class
        return kwargs.get("tokenizer")

    def _report_dir(self, args) -> Path:
        report_dir = Path(os.environ.get("SFT_REPORTS_DIR", Path(args.output_dir) / "reports"))
        report_dir.mkdir(parents=True, exist_ok=True)
        return report_dir

    def _write_report(self, args, filename: str, payload) -> None:
        if int(os.environ.get("RANK", "0")) != 0:
            return
        with (self._report_dir(args) / filename).open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)

    def _temporal_init_adapter_targets(self) -> list[str]:
        if not self.temporal_init_checkpoint:
            return []
        try:
            return adapter_target_modules(load_adapter_config(self.temporal_init_checkpoint))
        except Exception:
            return []

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        processor = self._processor(kwargs)
        tokenizer = getattr(processor, "tokenizer", processor)
        time_token_ids = resolve_time_token_ids(tokenizer) if self._use_time_tokens else []

        if self.sft_checkpoint and not self._loaded and not self._resume_mode:
            load_sft_extra(model, self.sft_checkpoint, strict=True)
            if int(os.environ.get("RANK", "0")) == 0:
                from peft import get_peft_model_state_dict
                from safetensors.torch import load_file

                expected = load_file(str(Path(self.sft_checkpoint) / "adapter_model.safetensors"))
                adapter_model = model
                while hasattr(adapter_model, "module"):
                    adapter_model = adapter_model.module
                actual = get_peft_model_state_dict(adapter_model)
                if set(expected) != set(actual):
                    raise AssertionError("SFT continuation adapter keys differ from parent checkpoint")
                mismatches = [name for name, value in expected.items()
                              if not torch.equal(value.to(actual[name].dtype), actual[name].detach().cpu())]
                if mismatches:
                    raise AssertionError(f"SFT continuation adapter differs from parent: {mismatches[:5]}")
                self._write_report(args, "sft_initialization.json", {
                    "status": "pass", "checkpoint": self.sft_checkpoint,
                    "adapter_tensors_exact": len(expected), "sft_extra_loaded": True,
                    "global_step_before_training": int(state.global_step),
                    "resume_only_model": True,
                })
                del actual, expected
            if int(state.global_step) != 0:
                raise AssertionError("New-data SFT continuation must start at step zero")
            self._loaded = True
        elif self.temporal_init_extra and not self._loaded and not self._resume_mode:
            # Resuming an in-progress SFT checkpoint restores the full
            # model -- including the ATE/embed/lm_head modules_to_save copies
            # -- via swift's own resume_from_checkpoint path with
            # resume_only_model=false. Loading temporal initialization's extra state here on
            # top of that would clobber whatever SFT training already did
            # to ATE and the time-token rows.
            from temporal_init.checkpoint_state import load_temporal_init_extra

            extra_dir = Path(self.temporal_init_extra)
            if extra_dir.name == "sta_extra.pt":
                extra_dir = extra_dir.parent
            load_temporal_init_extra(model, str(extra_dir), strict=True)
            self._loaded = True

        if not self.temporal_init_checkpoint and not self._resume_mode:
            thinker = unwrap_model(model).thinker
            ate = getattr(thinker, "ate", None)
            ate = active_modules_to_save(ate) if ate is not None else None
            weight_zero = (
                bool(torch.count_nonzero(ate.out_proj.weight.detach()).item() == 0)
                if ate is not None
                else None
            )
            bias_zero = (
                bool(torch.count_nonzero(ate.out_proj.bias.detach()).item() == 0)
                if ate is not None
                else None
            )
            base_model_report = {
                "status": "pass" if not self._use_ate or (weight_zero and bias_zero) else "fail",
                "initialization": (
                    "raw_base_plus_time_tokens_plus_zero_output_ate"
                    if self._use_time_tokens and self._use_ate
                    else "raw_base_plus_decimal_times_plus_zero_output_ate"
                    if self._use_ate
                    else "raw_base_without_ate"
                ),
                "temporal_init_checkpoint_loaded": False,
                "temporal_init_extra_loaded": False,
                "use_time_tokens": self._use_time_tokens,
                "use_ate": self._use_ate,
                "ate_out_proj_weight_zero": weight_zero,
                "ate_out_proj_bias_zero": bias_zero,
                "time_token_ids": list(map(int, time_token_ids)),
            }
            self._write_report(args, "direct_base_model_initialization.json", base_model_report)
            if base_model_report["status"] != "pass":
                raise AssertionError("direct base model SFT must start with zero-output ATE")

        freeze_for_sft(model, train_time_rows=self._use_time_tokens)
        model._sft_adapter_targets = self._temporal_init_adapter_targets()
        _groups, audit = build_sft_optimizer_groups(
            model,
            lr_lora=float(os.environ.get("SFT_LR_LORA", "1e-5")),
            lr_projector_lora=float(os.environ.get("SFT_LR_PROJECTOR_LORA", "5e-6")),
            lr_ate=float(os.environ.get("SFT_LR_ATE", "2e-5")),
            lr_time_rows=float(os.environ.get("SFT_LR_TIME_ROWS", "2e-5")),
            weight_decay=float(getattr(args, "weight_decay", 0.0)),
        )
        self._write_report(args, "trainable_params.json", audit)
        if self._row_guard_steps > 0:
            if self._use_time_tokens:
                self._row_guard = VocabularyRowGuard(model, time_token_ids)
                self._write_report(
                    args,
                    "vocabulary_row_guard.json",
                    {
                        "status": "initialized",
                        "guard_steps": self._row_guard_steps,
                        "time_token_ids": list(map(int, time_token_ids)),
                        **self._row_guard.summary(),
                    },
                )
            self._frozen_audio_guard = FrozenParameterGuard(
                model,
                is_audio_encoder_or_projector_base_name,
                label="frozen_audio_encoder_projector_base",
            )
            self._write_report(
                args,
                "frozen_audio_projector_guard.json",
                {
                    "status": "initialized",
                    "guard_steps": self._row_guard_steps,
                    **self._frozen_audio_guard.summary(),
                },
            )
            self._update_guard = TrainableUpdateGuard(model, time_token_ids)
            self._write_report(
                args,
                "gradient_audit.json",
                {
                    "status": "initialized",
                    "guard_steps": self._row_guard_steps,
                    **self._update_guard.summary(),
                },
            )

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if model is None or self._row_guard_steps <= 0:
            return
        step = int(state.global_step) + 1
        if step > self._row_guard_steps:
            return
        processor = self._processor(kwargs)
        tokenizer = getattr(processor, "tokenizer", processor)
        time_token_ids = resolve_time_token_ids(tokenizer) if self._use_time_tokens else []
        self._last_grad_report = self._grad_based_report(model, time_token_ids, step)

    def on_optimizer_step(self, args, state, control, model=None, **kwargs):
        if model is None or self._row_guard_steps <= 0:
            return
        step = int(state.global_step) + 1
        if step > self._row_guard_steps:
            return
        if self._row_guard is not None:
            try:
                report = self._row_guard.check(model)
                report.update({
                    "checked_after_step": step,
                    "guard_steps": self._row_guard_steps,
                    "time_token_ids": sorted(self._row_guard.time_token_ids),
                })
            except Exception as exc:
                report = {
                    "status": "fail",
                    "checked_after_step": step,
                    "error": str(exc),
                }
                self._write_report(args, "vocabulary_row_guard.json", report)
                raise
            self._write_report(args, "vocabulary_row_guard.json", report)
        if self._frozen_audio_guard is not None:
            try:
                frozen_report = self._frozen_audio_guard.check(model)
                frozen_report.update({
                    "checked_after_step": step,
                    "guard_steps": self._row_guard_steps,
                })
            except Exception as exc:
                frozen_report = {
                    "status": "fail",
                    "checked_after_step": step,
                    "error": str(exc),
                }
                self._write_report(args, "frozen_audio_projector_guard.json", frozen_report)
                raise
            self._write_report(args, "frozen_audio_projector_guard.json", frozen_report)

    def on_save(self, args, state, control, model=None, **kwargs):
        if model is None or int(os.environ.get("RANK", "0")) != 0:
            return
        processor = self._processor(kwargs)
        tokenizer = getattr(processor, "tokenizer", processor)
        time_token_ids = resolve_time_token_ids(tokenizer) if self._use_time_tokens else []
        ckpt = Path(args.output_dir) / f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        save_sft_extra(
            model=model,
            tokenizer=tokenizer,
            ckpt_dir=ckpt,
            time_token_ids=time_token_ids,
            base_model_path=self.base_model_path,
            temporal_init_checkpoint=self.temporal_init_checkpoint,
            step=int(state.global_step),
            project_root=self.project_root,
            adapter_targets=getattr(model, "_sft_adapter_targets", []),
        )
        if self._update_guard is not None:
            # Authoritative "did this group actually train" signal, checked
            # cumulatively from training start against the checkpoint just
            # written. A single-step grad-based delta can round away to
            # nothing in bf16 storage, so a short run relies on the grad
            # signal below instead; a longer run's cumulative weight delta is
            # what actually matters for the checkpoint being shipped.
            weight_diff = self._update_guard.check(model)
            combined = {
                "checked_at_save_step": int(state.global_step),
                "method": "grad_based (pre-step) OR weight_diff (cumulative since train start)",
                "grad_based": self._last_grad_report,
                "weight_diff": weight_diff,
            }
            for group in ("lora", "projector_lora", "ate", "time_rows"):
                grad_flag = bool(self._last_grad_report and self._last_grad_report.get(f"{group}_nonzero"))
                weight_flag = bool(weight_diff.get(f"{group}_nonzero"))
                combined[f"{group}_nonzero"] = grad_flag or weight_flag
            self._write_report(args, "gradient_audit.json", combined)
