"""Independent group-GRPO audit using the native device, dtype and reduction shape.

CPU scalar reductions can differ by one float32 ULP from CUDA row reductions.
The group-standard-deviation division amplifies that difference for close rewards.
This module never replaces or alters the trainer's actual advantages.
"""
import torch

class AdvantageAuditError(AssertionError):
    def __init__(self,diagnostics):
        self.diagnostics=diagnostics
        super().__init__(str(diagnostics))

def validate_groups(groups,device):
    assert groups and all(len(g)==4 for g in groups)
    rewards=torch.tensor([[r['reward'] for r in g] for g in groups],dtype=torch.float32,device=device)
    actual=torch.tensor([[r['advantage'] for r in g] for g in groups],dtype=torch.float32,device=device)
    expected=(rewards-rewards.mean(dim=1,keepdim=True))/(rewards.std(dim=1,keepdim=True)+1e-4)
    finite=bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    error=float((actual-expected).abs().max()) if finite else None
    diagnostics=dict(groups=len(groups),device=str(device),dtype=str(rewards.dtype),max_absolute_error=error,finite=finite,formula='(r-row_mean)/(sample_row_std+1e-4)',same_device_and_row_shape=True)
    if not finite or not torch.allclose(actual,expected,atol=1e-5,rtol=1e-5):
        diagnostics.update(rewards_repr=str(rewards.cpu().tolist()),actual_repr=str(actual.cpu().tolist()),expected_repr=str(expected.cpu().tolist()))
        raise AdvantageAuditError(diagnostics)
    return diagnostics
