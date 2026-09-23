import hashlib,json,os,time
from pathlib import Path
import torch
from peft import get_peft_model_state_dict
from transformers import TrainerCallback
AUD=Path(os.environ['TEMA_SPAN_AUDIT']);AUD.mkdir(parents=True,exist_ok=True)
ARM=os.environ['TEMA_SPAN_ARM']
def emit(kind,row):
 with (AUD/f'{kind}.rank{os.environ.get("RANK","0")}.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
class Audit(TrainerCallback):
 def __init__(self,t):self.t=t
 def on_train_begin(self,args,state,control,**kw):
  raw=self.t.accelerator.unwrap_model(self.t.model);w=get_peft_model_state_dict(raw)
  assert len(w)==392 and all(torch.count_nonzero(v)==0 for k,v in w.items() if 'lora_B' in k)
  assert all('lora_' in n and 'thinker.model' in n for n,p in raw.named_parameters() if p.requires_grad)
  assert args.world_size==4
  # Swift enables model GC, then resets args.gradient_checkpointing=False.
  gc_modules=[n for n,m in raw.named_modules() if getattr(m,"gradient_checkpointing",False) is True]
  assert gc_modules, "No model module has gradient checkpointing enabled"
  assert args.per_device_train_batch_size==int(os.environ.get('TEMA_SPAN_MICROBATCH','2'))
  assert args.gradient_accumulation_steps==int(os.environ.get('TEMA_SPAN_ACCUMULATION','2'))
  assert args.per_device_train_batch_size*args.gradient_accumulation_steps==4
  assert state.global_step==0 and not args.resume_from_checkpoint
  if ARM=='rl':
   assert args.beta==.04 and args.num_generations==4 and self.t.ref_model is None
  self.before={k:v.detach().cpu().clone() for k,v in w.items()};self.started=time.monotonic()
  hashes={k:hashlib.sha256(v.float().numpy().tobytes()).hexdigest() for k,v in self.before.items()}
  (AUD/f'initial_adapter.rank{os.environ.get("RANK","0")}.json').write_text(json.dumps(hashes,sort_keys=True))
  emit('startup',dict(gradient_checkpointing=True,checkpointed_modules=gc_modules,arm=ARM,per_device_batch_size=args.per_device_train_batch_size,gradient_accumulation_steps=args.gradient_accumulation_steps,fresh_lora_zero_delta=True,tensors=len(w),trainable_parameters=sum(v.numel() for v in w.values()),threads={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']}))
 def on_step_end(self,args,state,control,**kw):
  if state.global_step==4:
   w=get_peft_model_state_dict(self.t.accelerator.unwrap_model(self.t.model));changed=sum(not torch.equal(v.detach().cpu(),self.before[k]) for k,v in w.items())
   assert changed and all(torch.isfinite(v).all() for v in w.values());self.before={}
   emit('update',dict(step=4,changed_tensors=changed,all_finite=True));control.should_save=True
  if state.global_step in (4,args.max_steps//2,args.max_steps):
   emit('timing',dict(step=state.global_step,elapsed=time.monotonic()-self.started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30));control.should_save=True
 def on_train_end(self,args,state,control,**kw):emit('completion',dict(step=state.global_step,epoch=state.epoch,elapsed=time.monotonic()-self.started))
