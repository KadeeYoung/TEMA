"""ms-swift 4.5.2 custom register for SFT Qwen2.5-Omni + ATE."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sft.model.load_temporal_init import ensure_temporal_init_time_tokens  # noqa: E402
from sft.model.optimizer_groups import (  # noqa: E402
    TIME_TOKENS,
    build_sft_optimizer_groups,
    freeze_for_sft,
    register_time_row_gradient_mask,
    resolve_time_token_ids,
)

MODEL_TYPE = "qwen2_5_omni_ate_sft"
NO_TIME_MODEL_TYPE = "qwen2_5_omni_ate_sft_no_time_tokens"
NO_ATE_MODEL_TYPE = "qwen2_5_omni_sft_no_ate"
NO_TIME_NO_ATE_MODEL_TYPE = "qwen2_5_omni_sft_no_time_tokens_no_ate"


def _truthy_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _register_tokenizer_only_tokens(tokenizer) -> None:
    if all(tokenizer.convert_tokens_to_ids(t) is not None for t in TIME_TOKENS):
        resolve_time_token_ids(tokenizer)
        return
    if not _truthy_env("SFT_ALLOW_TOKENIZER_ONLY_BOOTSTRAP", True):
        resolve_time_token_ids(tokenizer)
        return
    added = tokenizer.add_special_tokens({"additional_special_tokens": list(TIME_TOKENS)})
    if added != len(TIME_TOKENS):
        raise RuntimeError(f"expected to add 20 tokenizer-only time tokens, got {added}")
    resolve_time_token_ids(tokenizer)


class SftATEOptimizerCallback:
    """Factory wrapper for the installed swift OptimizerCallback interface."""

    def __new__(cls, *args, **kwargs):
        from swift.optimizers import OptimizerCallback
        from transformers import Trainer as HfTrainer

        class _Callback(OptimizerCallback):
            def create_optimizer(self, model=None):
                if model is None:
                    model = self.trainer.model
                use_time_tokens = _truthy_env("SFT_USE_TIME_TOKENS", True)
                freeze_for_sft(model, train_time_rows=use_time_tokens)
                tokenizer = getattr(getattr(self.trainer, "template", None), "tokenizer", None)
                if use_time_tokens and tokenizer is not None:
                    time_ids = resolve_time_token_ids(tokenizer)
                    register_time_row_gradient_mask(model, time_ids)
                groups, audit = build_sft_optimizer_groups(
                    model,
                    lr_lora=float(os.environ.get("SFT_LR_LORA", getattr(self.args, "learning_rate", 1e-5))),
                    lr_projector_lora=float(os.environ.get("SFT_LR_PROJECTOR_LORA", "5e-6")),
                    lr_ate=float(os.environ.get("SFT_LR_ATE", "2e-5")),
                    lr_time_rows=float(os.environ.get("SFT_LR_TIME_ROWS", "2e-5")),
                    weight_decay=float(getattr(self.args, "weight_decay", 0.0)),
                )
                for group in groups:
                    group["temporal_init_group"] = group["sft_group"]
                    group.pop("param_names", None)
                report_dir = Path(os.environ.get("SFT_REPORTS_DIR", Path(self.args.output_dir) / "reports"))
                report_dir.mkdir(parents=True, exist_ok=True)
                if int(os.environ.get("RANK", "0")) == 0:
                    with (report_dir / "trainable_params.json").open("w", encoding="utf-8") as f:
                        json.dump(audit, f, ensure_ascii=False, indent=2, sort_keys=True)
                try:
                    optimizer_cls, optimizer_kwargs = HfTrainer.get_optimizer_cls_and_kwargs(self.args, model)
                except TypeError:
                    optimizer_cls, optimizer_kwargs = HfTrainer.get_optimizer_cls_and_kwargs(self.args)
                return optimizer_cls(groups, **optimizer_kwargs)

        return _Callback(*args, **kwargs)


def _make_sft_loader_class(use_time_tokens: bool = True, use_ate: bool = True):
    from swift.model.models.qwen import Qwen2_5OmniLoader

    class SftQwen2_5OmniATELoader(Qwen2_5OmniLoader):
        def _get_model_processor(self, model_dir, config):
            model, processor = super()._get_model_processor(model_dir, config)
            tokenizer = processor.tokenizer
            if model is None:
                if use_time_tokens:
                    _register_tokenizer_only_tokens(tokenizer)
                return model, processor

            token_info = (
                ensure_temporal_init_time_tokens(
                    model,
                    tokenizer,
                    allow_bootstrap=_truthy_env("SFT_ALLOW_TIME_TOKEN_BOOTSTRAP", True),
                )
                if use_time_tokens
                else {"enabled": False, "n_before": len(tokenizer), "n_after": len(tokenizer)}
            )
            if use_ate:
                d_model = model.thinker.config.text_config.hidden_size
                from temporal_init.omni_ate_patch import attach_ate

                attach_ate(
                    model,
                    d_model=d_model,
                    n_freqs=int(os.environ.get("SFT_ATE_NFREQS", "64")),
                    hidden=int(os.environ.get("SFT_ATE_HIDDEN", "512")),
                )
            freeze_for_sft(model, train_time_rows=use_time_tokens)
            model._sft_token_info = token_info
            model._sft_use_time_tokens = use_time_tokens
            model._sft_use_ate = use_ate
            return model, processor

    return SftQwen2_5OmniATELoader


def get_model_tokenizer_qwen2_5_omni_ate_sft(model_dir, *args, **kwargs):
    """Compatibility helper for direct tests outside the swift launcher."""
    loader_cls = _make_sft_loader_class(use_time_tokens=True, use_ate=True)
    loader = loader_cls(*args, **kwargs)
    config = loader.get_config(model_dir)
    return loader._get_model_processor(model_dir, config)


def _try_register() -> None:
    from swift.callbacks import callbacks_map
    from swift.model import Model, ModelArch, ModelGroup, ModelMeta, register_model
    from swift.optimizers import optimizers_map
    from swift.template import TemplateType

    from sft.train.callbacks import SftStateCallback

    def register_variant(model_type: str, use_time_tokens: bool, use_ate: bool) -> None:
        register_model(ModelMeta(
            model_type,
            [
                ModelGroup(
                    [
                        Model(
                            "Qwen/Qwen2.5-Omni-7B",
                            "Qwen/Qwen2.5-Omni-7B",
                            None,
                        )
                    ],
                    TemplateType.qwen2_5_omni,
                )
            ],
            _make_sft_loader_class(use_time_tokens=use_time_tokens, use_ate=use_ate),
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
    optimizers_map["sft_ate"] = SftATEOptimizerCallback
    callbacks_map["sft_state"] = SftStateCallback


try:
    _try_register()
except Exception as exc:  # pragma: no cover - import-only diagnostics
    print(f"[sft] deferred ms-swift registration failed: {exc}", file=sys.stderr)
