"""Fresh-SFT startup plus actual reference-context and final-weight checks."""
import torch
from peft import get_peft_model_state_dict
from tema_chat.training.audit_base import Audit as FreshAudit, emit, digest


class Audit(FreshAudit):
    def on_train_begin(self,args,state,control,**kw):
        super().on_train_begin(args,state,control,**kw)
        raw=self.t.accelerator.unwrap_model(self.t.model)
        assert self.t.ref_adapter_name is None
        layers=[m for m in raw.modules() if hasattr(m,'lora_A')]
        assert layers
        with self.t.null_ref_context():
            assert all(m.disable_adapters for m in layers)
        assert all(not m.disable_adapters for m in layers)
        emit('reference',dict(reference='new SFT848 via disabled fresh LoRA',context_checked=True,lora_layers=len(layers)))

    def on_train_end(self,args,state,control,**kw):
        weights=get_peft_model_state_dict(self.t.accelerator.unwrap_model(self.t.model))
        assert all(torch.isfinite(v).all() for v in weights.values())
        emit('final_weights',dict(step=state.global_step,tensors=len(weights),all_finite=True))
        super().on_train_end(args,state,control,**kw)
