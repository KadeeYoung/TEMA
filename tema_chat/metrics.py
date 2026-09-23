"""Offline evidence/QA aggregation from saved complete-dialogue predictions and judgments."""
import argparse,collections
from pathlib import Path
from .common import read,dump,write_rows
from .evidence_metrics import uid,row_metrics,summary
FAMILIES={'F1':['A1','A5','A5-gap'],'F2':['A2','A6-yes','A6-no','A16'],'F3':['A3','A4'],'F4':['A7','A8','A9','A10','A11','A13','A14'],'F5':['A17','A18']}
def aggregate(rows,natural=False):
 n=len(rows);groups=collections.defaultdict(list)
 for r in rows:groups[r['group']].append(r)
 qa=None if any(r.get('answer_required_correct') is None for r in rows) else sum(r['answer_required_correct'] for r in rows)
 qt=None if qa is None else sum(r['answer_plus_time_correct'] for r in rows)
 ev=None if natural else summary([dict(r['evidence'],correct=bool(r.get('answer_required_correct')),verdict='UNJUDGEABLE' if r.get('unjudgeable') else 'CORRECT' if r.get('answer_required_correct') else 'INCORRECT') for r in rows])
 return dict(n=n,qa_correct=qa,qa=None if qa is None else qa/n,qa_plus_time_correct=qt,qa_plus_time=None if qt is None else qt/n,groups=len(groups),whole_qa=None if qa is None else sum(all(r['answer_required_correct'] for r in rs) for rs in groups.values()),whole_qa_plus_time=None if qa is None else sum(all(r['answer_plus_time_correct'] for r in rs) for rs in groups.values()),evidence=ev)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--predictions',type=Path,nargs='+',required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--judgments',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--natural',action='store_true');a=p.parse_args()
 contracts={uid(r):r for r in read(a.data_root/'benchmark/scoring_contract.jsonl')};pp=[r for f in a.predictions for r in read(f)];pred={uid(r):r for r in pp};assert len(pp)==len(pred)==1239 and set(pred)==set(contracts)
 judgments={r['id']:r for r in read(a.judgments)} if a.judgments else {}
 if a.judgments:assert len(judgments)==1239 and set(judgments)==set(pred)
 rows=[]
 for k,c in contracts.items():
  family=next(f for f,tasks in FAMILIES.items() if c['task_type'] in tasks)
  r=dict(judgments.get(k,{}),uid=k,group=c['group_id']+'|'+c['ticket_id'],task=c['task_type'],family=family,subset=c.get('evaluation_subset','r2_200'),route_size=len(c['gold_route']))
  if not a.natural:r['evidence']=row_metrics(pred[k],c)
  rows.append(r)
 result={'all':aggregate(rows,a.natural),'families':{f:aggregate([r for r in rows if r['family']==f],a.natural) for f in FAMILIES}}
 result['macro_qa']=None if not judgments else sum(v['qa'] for v in result['families'].values())/5
 result['macro_qa_plus_time']=None if not judgments else sum(v['qa_plus_time'] for v in result['families'].values())/5
 result['route_slices']={tag:aggregate([r for r in rows if (r['route_size']>1)==multi],a.natural) for tag,multi in [('single',False),('multiple',True)]}
 a.output.mkdir(parents=True,exist_ok=True);dump(a.output/'metrics.json',result);write_rows(a.output/'per_turn_metrics.jsonl',rows);print(result['all'])
if __name__=='__main__':main()
