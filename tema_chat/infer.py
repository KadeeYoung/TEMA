"""Complete free-running conversations; never insert gold assistant history."""
import argparse, gc, time
from pathlib import Path
from .common import read,sha,dump,write_rows,digest,SYSTEM
from .natural_prompt import SYSTEM as NATURAL_SYSTEM

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--model',required=True);p.add_argument('--adapter');p.add_argument('--adapter-kind',choices=['grpo','sft'],default='grpo')
 p.add_argument('--data-root',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
 p.add_argument('--natural',action='store_true',help='Original Qwen baseline uses natural answers')
 p.add_argument('--batch-size',type=int,default=16);p.add_argument('--max-new-tokens',type=int,default=1024)
 p.add_argument('--num-shards',type=int,default=1);p.add_argument('--shard-index',type=int,default=0)
 p.add_argument('--attn-impl',default='flash_attention_2');a=p.parse_args()
 if a.batch_size<1 or not 0<=a.shard_index<a.num_shards:p.error('invalid batch size/shard')
 from tema_chat.evaluation.rollout import load_sidecar,build_turn_specs,run_rollout
 from tema_chat.evaluation.engine import load_engine,balanced_shard_items,install_audio_decode_cache,install_audio_feature_cache,context_length_key
 data=a.data_root.resolve()/'benchmark';rows=read(data/'test.swift.jsonl');man=read(data/'manifest.jsonl');side=load_sidecar(data/'sidecar.jsonl')
 assert len(rows)==len(man)==253
 indexed=[]
 for i,(r,m) in enumerate(zip(rows,man)):
  m=dict(m,source_dialogue_idx=i)
  assert all(Path(x).is_file() for x in r['audios']), 'Run dataset materialization/rewrite first'
  build_turn_specs(r,m,side[(m['group_id'],m['ticket_id'])]);indexed.append((i,r,m))
 indexed=balanced_shard_items(indexed,a.num_shards,a.shard_index)
 a.output.mkdir(parents=True,exist_ok=True)
 target=a.output/f'predictions.shard{a.shard_index}.jsonl'
 if target.exists():raise FileExistsError(f'{target}: preserve completed outputs or use a new output directory')
 install_audio_decode_cache(4096)
 model_type='qwen2_5_omni'
 if a.adapter and a.adapter_kind=='sft':
  import sft.model.register_qwen25_omni_ate
  model_type='qwen2_5_omni_sft_no_time_tokens_no_ate'
 engine=load_engine(a.model,a.adapter,'bfloat16','cuda:0',a.batch_size,attn_impl=a.attn_impl,model_type=model_type,load_extra=bool(a.adapter and a.adapter_kind=='sft'))
 install_audio_feature_cache(engine,4096)
 prompt=NATURAL_SYSTEM if a.natural else SYSTEM
 class PromptEngine:
  def infer(self,requests,config):
   import torch
   for r in requests:r.messages=[dict(role='system',content=prompt)]+[dict(m) for m in r.messages]
   order=sorted(range(len(requests)),key=lambda i:context_length_key(dict(messages=requests[i].messages,audios=requests[i].audios)))
   outputs=[None]*len(requests)
   def run(batch):
    try:return engine.infer(batch,config)
    except torch.OutOfMemoryError:
     if len(batch)==1:raise
     gc.collect();torch.cuda.empty_cache();mid=len(batch)//2
     return run(batch[:mid])+run(batch[mid:])
   for offset in range(0,len(order),a.batch_size):
    ids=order[offset:offset+a.batch_size];out=run([requests[i] for i in ids])
    assert len(ids)==len(out)
    for i,r in zip(ids,out):outputs[i]=r
   return outputs
 started=time.time();records,seconds=run_rollout(PromptEngine(),indexed,side,'structured',a.max_new_tokens)
 write_rows(target,records)
 dump(a.output/f'receipt.shard{a.shard_index}.json',dict(model=a.model,adapter=a.adapter,natural=a.natural,batch_size=a.batch_size,max_new_tokens=a.max_new_tokens,system=prompt,system_sha256=digest(prompt),groups=len(indexed),turns=len(records),history='model_generated',oracle=False,predictions_sha256=sha(target),dataset_sha256=sha(data/'test.swift.jsonl'),elapsed=time.time()-started))
 print(target)
if __name__=='__main__':main()
