"""Portable exact-config launchers. --dry-run does not load a model or start training."""
import argparse, json, os, subprocess, sys
from pathlib import Path
from .common import sha, dump
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('stage',choices=['sft','grpo','temporal-init'])
 p.add_argument('--model',required=True,help='Local downloaded base model (SFT: original Qwen; GRPO: merged SFT)')
 p.add_argument('--data-root',type=Path,required=True,help='Materialized dataset root with absolute audio paths')
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--init-adapter',type=Path,help='Temporal-init adapter directory, required for SFT')
 p.add_argument('--gpus',default='0,1,2,3');p.add_argument('--dry-run',action='store_true')
 a=p.parse_args();a.output=a.output.resolve();a.data_root=a.data_root.resolve()
 if not Path(a.model).is_dir():p.error('--model must be a locally downloaded model directory')
 devices=[x for x in a.gpus.split(',') if x]
 if len(set(devices))!=len(devices):p.error('GPU IDs must be unique')
 required_gpus=6 if a.stage=='temporal-init' else 4
 if len(devices)!=required_gpus:p.error(f'Exact recipe requires {required_gpus} GPUs; changing world size changes exposure/order')
 env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH',''),CUDA_VISIBLE_DEVICES=a.gpus,NPROC_PER_NODE=str(len(devices)),ENABLE_AUDIO_OUTPUT='0',SWIFT_AUDIO_LOAD_BACKEND='soundfile_pyav',TOKENIZERS_PARALLELISM='false')
 for k,v in {'OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'1'}.items():env[k]=os.environ.get('TEMA_'+k,v)
 for k,v in {'MASTER_PORT':'29647','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}.items():env.setdefault(k,v)
 dataset=a.data_root/('rl/train2048.rl.jsonl' if a.stage=='grpo' else 'sft/train.swift.jsonl')
 if a.stage=='temporal-init':dataset=a.data_root/'temporal_init/train.jsonl'
 if not dataset.is_file():p.error(f'missing dataset: {dataset}')
 a.output.mkdir(parents=True,exist_ok=True)
 if a.stage=='grpo':
  meta=a.data_root/'rl/train2048.metadata.jsonl'
  rows=[json.loads(l) for l in dataset.open() if l.strip()];m=[json.loads(l) for l in meta.open() if l.strip()]
  assert len(rows)==len(m)==2048 and [r['span_state_key'] for r in rows]==[r['span_state_key'] for r in m]
  for row in rows:
   assert len(row['messages'])==1 and row['messages'][0]['role']=='user'
   assert all(Path(x).is_file() for x in row['audios'])
  replacements={'MODEL':str(Path(a.model).resolve()),'DATASET':str(dataset),'OUTPUT_DIR':str(a.output),'EXTERNAL_PLUGINS':str(ROOT/'tema_chat/training/grpo_plugin.py'),'DEEPSPEED':str(ROOT/'configs/zero2.json')}
  cmd=[os.environ.get('SWIFT_BIN','swift'),'rlhf']+[replacements.get(x[2:-1],x) if x.startswith('${') else x for x in json.loads((ROOT/'configs/grpo_args.json').read_text())]
  env.update(TEMA_SPAN_ARM='rl',TEMA_SPAN_MICROBATCH='2',TEMA_SPAN_ACCUMULATION='2',TEMA_SPAN_AUDIT=str(a.output/'audit'),TEMA_CANONICAL_RL_DATA=str(dataset),TEMA_CANONICAL_RL_SHA256=sha(dataset),TEMA_CURRICULUM_METADATA=str(meta),TEMA_CURRICULUM_METADATA_SHA256=sha(meta))

 else:
  import yaml
  temporal=a.stage=='temporal-init'
  cfg=json.loads((ROOT/'configs'/('temporal_init.json' if temporal else 'sft.json')).read_text())
  cfg.update(model=str(Path(a.model).resolve()),dataset=str(dataset),output_dir=str(a.output),custom_register_path=str(ROOT/('temporal_init/register.py' if temporal else 'sft/model/register_qwen25_omni_ate.py')))
  if temporal:
   cfg['val_dataset']=str(a.data_root/'temporal_init/val500.jsonl')
   cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node','6',str(ROOT/'temporal_init/training/train.py')]
  else:
   if a.init_adapter is None:p.error('--init-adapter is required for SFT')
   init=a.init_adapter.resolve()
   for f in ['adapter_config.json','adapter_model.safetensors','sta_extra.pt']:
    if not (init/f).is_file():p.error(f'missing temporal-init checkpoint file {init/f}')
   cfg.update(init_from_temporal_init=str(init),temporal_init_adapter_config=str(init/'adapter_config.json'),temporal_init_extra=str(init/'sta_extra.pt'),deepspeed=str(ROOT/'configs/zero2.json'),reports_dir=str(a.output/'reports'))
   cmd=[sys.executable,str(ROOT/'sft/train/sft_main.py')]
  config=a.output/'input_config.yaml';config.write_text(yaml.safe_dump(cfg,sort_keys=False))
  cmd+=['--config',str(config)]
  if a.dry_run:cmd+=['--dry-run']
 receipt={'stage':a.stage,'command':cmd,'dataset_sha256':sha(dataset),'gpus':devices,'dry_run':a.dry_run,'environment':{k:env[k] for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NPROC_PER_NODE']}}
 dump(a.output/'release_launch.json',receipt);print(json.dumps(receipt,indent=2))
 if not a.dry_run:subprocess.run(cmd,cwd=ROOT,env=env,check=True)
if __name__=='__main__':main()
