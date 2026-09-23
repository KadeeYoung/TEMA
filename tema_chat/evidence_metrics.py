import collections
from decimal import Decimal
import numpy as np
from scipy.optimize import linear_sum_assignment
from tema_chat.rewards.evidence import parse,evidence,score
from tema_chat.rewards.metrics import metrics as old_metrics

def uid(r):return f"{r['group_id']}|{r['ticket_id']}|{r['turn_index']}"

def gold_row(c):
 return dict(gold_route=[int(a[1:]) for a in c['gold_route']],gold_spans={str(int(a[1:])):v for a,v in c['gold_dense_evidence'].items()},audio_lengths={int(a[1:]):v for a,v in c['visible_audio_lengths'].items()},task_type=c['task_type'])

def row_metrics(pred,c):
 gold=gold_row(c)
 try:prefix=evidence(pred['pred_text'])
 except ValueError:prefix=pred['pred_text']
 parsed=parse(prefix,gold['audio_lengths']);base=old_metrics(pred['pred_text'],gold)
 out=dict(uid=uid(c),group=c['group_id']+'|'+c['ticket_id'],task=c['task_type'],turn=c['turn_index'],audio_count=pred['audio_count'],route_audio_count=len(c['gold_route']),evidence_type=c['evidence_type'],question_origin=c['question_origin'],valid=parsed.valid,D=base['D'],route_exact=base['route_exact'],count_exact=base['count_exact'],gold_intervals=base['gold_occurrences'],pred_intervals=base['pred_occurrences'],format_ok=bool(pred.get('pred_format',{}).get('format_ok',pred.get('format_ok',False))),H={})
 gs={int(a[1:]):v for a,v in c['gold_dense_evidence'].items()}
 out['group_type']='all_none' if not any(gs.values()) else ('mixed_positive_none' if any(not v for v in gs.values()) else 'positive_only')
 out['positive_task']=any(gs.values()) and c['evidence_type']!='derived_gap'
 for tolerance in ['0.1','0.2','0.3','0.5']:
  full=bool(base['count_exact'])
  if full:
   for a,spans in gs.items():
    if not spans:continue
    predicted=parsed.spans.get(a,[])
    matrix=np.array([[abs(Decimal(str(x[0]))-Decimal(str(y[0])))<=Decimal(tolerance) and abs(Decimal(str(x[1]))-Decimal(str(y[1])))<=Decimal(tolerance) for y in spans] for x in predicted],dtype=int)
    ii,jj=linear_sum_assignment(-matrix);full=full and int(matrix[ii,jj].sum())==len(spans)
  out['H'][tolerance]=bool(full)
 assert out['H']['0.1']==base['complete_set_01']
 # Preserve existing published one-to-one IoU counts, separately from D and H.
 out['tp05']=pred['span_occurrence_tp_at_0_5'];out['native_gold_intervals']=pred['span_gold_occurrences'];out['native_pred_intervals']=pred['span_pred_occurrences']
 assert out['native_gold_intervals']==out['gold_intervals'],(out['uid'],'gold counts mismatch')
 return out

def summary(rows):
 n=len(rows)
 if not n:return dict(n=0)
 tp=sum(r['tp05'] for r in rows);g=sum(r['native_gold_intervals'] for r in rows);p=sum(r['native_pred_intervals'] for r in rows)
 groups=collections.defaultdict(list)
 for r in rows:groups[r['group']].append(r)
 return dict(n=n,correct=sum(r['correct'] for r in rows),answer_success=sum(r['correct'] for r in rows)/n,unjudgeable=sum(r['verdict']=='UNJUDGEABLE' for r in rows),span_micro_f1_05=2*tp/(p+g) if p+g else None,precision=tp/p if p else None,recall=tp/g if g else None,tp=tp,fp=p-tp,fn=g-tp,gold_intervals=g,pred_intervals=p,mean_D=sum(r['D'] for r in rows)/n,H={t:sum(r['H'][t] for r in rows)/n for t in ['0.1','0.2','0.3','0.5']},route_exact=sum(r['route_exact'] for r in rows)/n,evidence_valid=sum(r['valid'] for r in rows)/n,format_ok=sum(r['format_ok'] for r in rows)/n,joint_H01_answer=sum(r['H']['0.1'] and r['correct'] for r in rows)/n,whole_dialogues=dict(n=len(groups),all_answers_correct=sum(all(r['correct'] for r in rs) for rs in groups.values()),all_evidence_correct=sum(all(r['H']['0.1'] for r in rs) for rs in groups.values())),answer_evidence_cells=dict(H1_A1=sum(r['H']['0.1'] and r['correct'] for r in rows),H1_A0=sum(r['H']['0.1'] and not r['correct'] for r in rows),H0_A1=sum(not r['H']['0.1'] and r['correct'] for r in rows),H0_A0=sum(not r['H']['0.1'] and not r['correct'] for r in rows)))
