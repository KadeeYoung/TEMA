"""Standard PEFT merge for a locally trained adapter (requires sufficient CPU RAM)."""
import argparse
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',required=True);p.add_argument('--adapter',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error('Output must be a new directory')
 import torch
 from transformers import Qwen2_5OmniForConditionalGeneration,AutoProcessor
 from peft import PeftModel
 def contains_weights(value):
  if isinstance(value,torch.Tensor):return value.numel()>0
  if isinstance(value,dict):return any(contains_weights(v) for v in value.values())
  if isinstance(value,(list,tuple)):return any(contains_weights(v) for v in value)
  return False
 for filename in ['sft_extra.pt','sta_extra.pt']:
  extra=Path(a.adapter)/filename
  if extra.exists() and contains_weights(torch.load(extra,map_location='cpu',weights_only=True)):
   raise ValueError('Nonempty extra tensors require a model-specific export; this tool supports the released no-time-token/no-ATE recipe only')
 base=Qwen2_5OmniForConditionalGeneration.from_pretrained(a.base,torch_dtype=torch.bfloat16,device_map='cpu',low_cpu_mem_usage=True,enable_audio_output=False)
 model=PeftModel.from_pretrained(base,a.adapter).merge_and_unload(safe_merge=True)
 model.save_pretrained(a.output,safe_serialization=True,max_shard_size='4GB')
 AutoProcessor.from_pretrained(a.base).save_pretrained(a.output)
if __name__=='__main__':main()
