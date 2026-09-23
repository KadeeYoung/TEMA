"""Stable native-token and valid-audio-feature identity for SFT coverage audits."""
import hashlib
import json

VERSION = 'tokens_labels_ordered_valid_audio_bf16_v1'


def native_hash(ids, labels):
    return hashlib.sha256(json.dumps({'ids': ids, 'labels': labels}, separators=(',', ':')).encode()).hexdigest()


def audio_fingerprints(features, attention_mask):
    """Ignore padded audio frames; preserve valid bf16 values and audio order.

    Native Qwen2.5-Omni encode already emits bf16 features here. Explicit bf16
    canonicalization also tolerates a collator/DeepSpeed float32 container. Only
    valid frames are copied from the accelerator, not 300-second padded arrays.
    """
    import torch
    assert isinstance(features, torch.Tensor) and isinstance(attention_mask, torch.Tensor)
    assert features.ndim == 3 and attention_mask.ndim == 2
    assert features.shape[0] == attention_mask.shape[0] and features.shape[-1] == attention_mask.shape[-1]
    masks = attention_mask.detach().to(device='cpu', dtype=torch.int32)
    result = []
    for index, mask in enumerate(masks):
        assert bool(((mask == 0) | (mask == 1)).all())
        length = int(mask.sum())
        assert length > 0 and bool((mask[:length] == 1).all()) and bool((mask[length:] == 0).all())
        valid = features[index, :, :length].detach().to(device='cpu', dtype=torch.bfloat16).contiguous()
        h = hashlib.sha256()
        h.update(json.dumps({'shape': list(valid.shape), 'dtype': 'bfloat16'}, separators=(',', ':')).encode())
        h.update(valid.view(torch.int16).numpy().tobytes())
        result.append(h.hexdigest())
    return result


def example_hash(text_hash, ordered_audio_hashes):
    return hashlib.sha256(json.dumps({'version': VERSION, 'text': text_hash,
        'audio': list(ordered_audio_hashes)}, separators=(',', ':')).encode()).hexdigest()
