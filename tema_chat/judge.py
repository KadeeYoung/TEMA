"""Semantic QA/QA+T with frozen prompt and Decimal temporal checking; uses API text only."""
import argparse,concurrent.futures,json,os,time
from pathlib import Path
from . import dual_protocol as protocol
from .common import read,write_rows,dump,digest,sha
from .evidence_metrics import uid
from .answer_extraction import extract,local_tests

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--predictions',type=Path,nargs='+',required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--model',default='deepseek-v4-flash');p.add_argument('--base-url',default='https://api.deepseek.com');p.add_argument('--workers',type=int,default=8);p.add_argument('--prepare-only',action='store_true',help='Write judge payloads without sending them');a=p.parse_args()
 local_tests();protocol.DATA=a.data_root.resolve()/'benchmark';preds=[r for f in a.predictions for r in read(f)];contracts={uid(r):r for r in read(protocol.DATA/'scoring_contract.jsonl')};assert len(preds)==len(contracts)==1239 and {uid(r) for r in preds}==set(contracts)
 history={}
 for g in read(protocol.DATA/'test.orchestrator.no_time_tokens.jsonl'):
  h=[];t=0
  for m in g['messages']:
   if m['role']=='assistant':
    history[f"{g['group_id']}|{g['ticket_id']}|{t}"]=list(h[:-1]);t+=1;h.append(dict(role='assistant',content=extract(m['content'])))
   elif m['role']=='user':h.append(dict(m))
 system=protocol.system_text();a.output.mkdir(parents=True,exist_ok=True);cache=a.output/'judge_cache';cache.mkdir(exist_ok=True)
 cases=[dict(uid=uid(r),payload=protocol.payload(r,contracts[uid(r)],history[uid(r)])) for r in preds]
 write_rows(a.output/'judge_inputs.jsonl',cases)
 receipt=dict(version=protocol.VERSION,system_sha256=digest(system),model=a.model,base_url=a.base_url,prediction_sha256={str(f):sha(f) for f in a.predictions},contract_sha256=sha(protocol.DATA/'scoring_contract.jsonl'),turns=len(cases),history='gold context for judging only; model rollout remains free running',prepare_only=a.prepare_only)
 dump(a.output/'judge_receipt.json',receipt)
 if a.prepare_only:return
 key=os.environ.get('DEEPSEEK_API_KEY')
 if not key:raise RuntimeError('Set DEEPSEEK_API_KEY in your environment; no key files are read')
 from openai import OpenAI
 client=OpenAI(api_key=key,base_url=a.base_url,timeout=180,max_retries=3)
 def fingerprint(case):return digest(dict(system=system,model=a.model,base_url=a.base_url,payload=case['payload']))
 def work(chunk):
  batch=[dict(c['payload'],id=f'item{i}') for i,c in enumerate(chunk)];last=''
  for attempt in range(3):
   try:
    user='Evaluate both metrics for each case using the dual protocol:\n'+json.dumps(batch,ensure_ascii=False,separators=(',',':'))
    if attempt:user+=f'\nSchema validation feedback: {last}. Retry {attempt}.'
    resp=client.chat.completions.create(model=a.model,messages=[dict(role='system',content=system),dict(role='user',content=user)],temperature=0,max_tokens=12000,response_format={'type':'json_object'})
    raw=json.loads(resp.choices[0].message.content);validated=protocol.validate(raw,batch);out=[]
    for case,row in zip(chunk,validated):
     row.update(request_sha256=digest(user),system_sha256=digest(system),prompt_version=protocol.VERSION,judge_source='deepseek_dual_api',model=a.model)
     dump(cache/(fingerprint(case)+'.json'),row);out.append((fingerprint(case),row))
    return out
   except Exception as exc:
    last=str(exc)[:1800]
    if attempt==2:
     if len(chunk)>1:return [item for c in chunk for item in work([c])]
     raise
    time.sleep(2**attempt)
 unique={fingerprint(c):c for c in cases};done={}
 for k in unique:
  path=cache/(k+'.json')
  if path.exists():done[k]=json.loads(path.read_text())
 pending=[c for k,c in unique.items() if k not in done];chunks=[pending[i:i+6] for i in range(0,len(pending),6)]
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
  for batch in pool.map(work,chunks):done.update(batch)
 result=[dict(done[fingerprint(c)],id=c['uid']) for c in cases]
 write_rows(a.output/'dual_judgments.jsonl',result)
if __name__=='__main__':main()
