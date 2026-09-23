"""MS-Swift model and optimizer registration for temporal initialization."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from temporal_init.omni_ate_patch import attach_ate
from temporal_init.time_tokens import TIME_TOKENS, add_time_tokens
from sft.model.optimizer_groups import (
    build_sft_optimizer_groups,
    freeze_for_sft,
    register_time_row_gradient_mask,
    resolve_time_token_ids,
)

MODEL_TYPE = "qwen2_5_omni_temporal_init_ate"
NO_TIME_MODEL_TYPE = "qwen2_5_omni_temporal_init_ate_no_time_tokens"
NO_ATE_MODEL_TYPE = "qwen2_5_omni_temporal_init_no_ate"
NO_TIME_NO_ATE_MODEL_TYPE = "qwen2_5_omni_temporal_init_no_time_tokens_no_ate"


def _use_time_tokens() -> bool:
    return os.environ.get("TEMA_TEMPORAL_INIT_USE_TIME_TOKENS", "1").lower() in {"1", "true", "yes", "on"}


def _use_ate() -> bool:
    return os.environ.get("TEMA_TEMPORAL_INIT_USE_ATE", "1").lower() in {"1", "true", "yes", "on"}


def _register_tokenizer_only(tokenizer) -> None:
    ids = [tokenizer.convert_tokens_to_ids(token) for token in TIME_TOKENS]
    if len(set(ids)) == 20 and all(token_id != tokenizer.unk_token_id for token_id in ids):
        resolve_time_token_ids(tokenizer)
        return
    added = tokenizer.add_special_tokens({"additional_special_tokens": TIME_TOKENS})
    if added != 20:
        raise RuntimeError(f"expected 20 base model time tokens, added {added}")
    resolve_time_token_ids(tokenizer)


def _make_loader_class(use_time_tokens: bool = True, use_ate: bool = True):
    from swift.model.models.qwen import Qwen2_5OmniLoader

    class TemporalInitLoader(Qwen2_5OmniLoader):
        def _get_model_processor(self, model_dir, config):
            model, processor = super()._get_model_processor(model_dir, config)
            tokenizer = processor.tokenizer
            if model is None:
                if use_time_tokens:
                    _register_tokenizer_only(tokenizer)
                return model, processor
            token_info = add_time_tokens(model, tokenizer) if use_time_tokens else {
                "enabled": False,
                "n_before": len(tokenizer),
                "n_after": len(tokenizer),
            }
            if use_ate:
                d_model = model.thinker.config.text_config.hidden_size
                attach_ate(
                    model,
                    d_model=d_model,
                    n_freqs=int(os.environ.get("TEMA_TEMPORAL_INIT_ATE_NFREQS", "64")),
                    hidden=int(os.environ.get("TEMA_TEMPORAL_INIT_ATE_HIDDEN", "512")),
                )
            for parameter in model.parameters():
                parameter.requires_grad = False
            model._temporal_init_info = token_info
            model._temporal_init_use_time_tokens = use_time_tokens
            model._temporal_init_use_ate = use_ate
            return model, processor

    return TemporalInitLoader


class TemporalInitOptimizerCallback:
    def __new__(cls, *args, **kwargs):
        from swift.optimizers import OptimizerCallback
        from transformers import Trainer as HfTrainer

        class _Callback(OptimizerCallback):
            def create_optimizer(self, model=None):
                model = model or self.trainer.model
                use_time_tokens = _use_time_tokens()
                freeze_for_sft(model, train_time_rows=use_time_tokens)
                if use_time_tokens:
                    tokenizer = self.trainer.template.tokenizer
                    time_ids = resolve_time_token_ids(tokenizer)
                    register_time_row_gradient_mask(model, time_ids)
                groups, audit = build_sft_optimizer_groups(
                    model,
                    lr_lora=float(os.environ.get("TEMA_TEMPORAL_INIT_LLM_LR", "1e-5")),
                    lr_projector_lora=float(os.environ.get("TEMA_TEMPORAL_INIT_PROJECTOR_LR", "1e-5")),
                    lr_ate=float(os.environ.get("TEMA_TEMPORAL_INIT_ATE_LR", "1e-4")),
                    lr_time_rows=float(os.environ.get("TEMA_TEMPORAL_INIT_EMB_LR", "5e-5")),
                    weight_decay=float(getattr(self.args, "weight_decay", 0.1)),
                )
                for group in groups:
                    group["temporal_init_group"] = group.pop("sft_group")
                    group.pop("param_names", None)
                report_dir = Path(self.args.output_dir) / "reports"
                report_dir.mkdir(parents=True, exist_ok=True)
                if int(os.environ.get("RANK", "0")) == 0:
                    with (report_dir / "trainable_params.json").open(
                        "w", encoding="utf-8"
                    ) as handle:
                        json.dump(audit, handle, ensure_ascii=False, indent=2, sort_keys=True)
                try:
                    optimizer_cls, optimizer_kwargs = HfTrainer.get_optimizer_cls_and_kwargs(self.args, model)
                except TypeError:
                    optimizer_cls, optimizer_kwargs = HfTrainer.get_optimizer_cls_and_kwargs(self.args)
                return optimizer_cls(groups, **optimizer_kwargs)

        return _Callback(*args, **kwargs)


def _register() -> None:
    from swift.callbacks import callbacks_map
    from swift.model import Model, ModelArch, ModelGroup, ModelMeta, register_model
    from swift.optimizers import optimizers_map
    from swift.template import TemplateType

    from temporal_init.callbacks import TemporalInitStateCallback

    def register_variant(model_type: str, use_time_tokens: bool, use_ate: bool) -> None:
        register_model(ModelMeta(
            model_type,
            [
                ModelGroup(
                    [
                        Model(
                            "Qwen/Qwen2.5-Omni-7B",
                            "Qwen/Qwen2.5-Omni-7B",
                            "Qwen/Qwen2.5-Omni-7B",
                        )
                    ],
                    TemplateType.qwen2_5_omni,
                )
            ],
            _make_loader_class(use_time_tokens=use_time_tokens, use_ate=use_ate),
            template=TemplateType.qwen2_5_omni,
            model_arch=ModelArch.qwen2_5_omni,
            architectures=["Qwen2_5OmniForConditionalGeneration"],
            additional_saved_files=["spk_dict.pt"],
            requires=["transformers>=4.50", "qwen_omni_utils", "soundfile"],
            tags=["vision", "video", "audio"],
            is_multimodal=True,
        ), exist_ok=True)

    register_variant(MODEL_TYPE, use_time_tokens=True, use_ate=True)
    register_variant(NO_TIME_MODEL_TYPE, use_time_tokens=False, use_ate=True)
    register_variant(NO_ATE_MODEL_TYPE, use_time_tokens=True, use_ate=False)
    register_variant(NO_TIME_NO_ATE_MODEL_TYPE, use_time_tokens=False, use_ate=False)
    optimizers_map["temporal_init"] = TemporalInitOptimizerCallback
    callbacks_map["temporal_init_state"] = TemporalInitStateCallback


try:
    _register()
except Exception as exc:  # pragma: no cover
    print(f"[temporal_init] registration failed: {exc}", file=sys.stderr)
    raise
