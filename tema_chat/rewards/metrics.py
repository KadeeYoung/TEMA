"""Complete set, exact boundary/duration diagnostics, independent of training D."""
import collections,json
from decimal import Decimal
from scipy.optimize import linear_sum_assignment
import numpy as np
from tema_chat.rewards.parsing import normalize_gold_spans,normalize_gold_route
from .evidence import parse,evidence,score

def close(a,b):return abs(Decimal(str(a))-Decimal(str(b)))<=Decimal('.1')
def metrics(text,row):
 try:prefix=evidence(text)
 except ValueError:prefix=text
 p=parse(prefix,row.get('audio_lengths'));gold=normalize_gold_spans(row['gold_spans']);route=normalize_gold_route(row['gold_route']);gn=sum(map(len,gold.values()))
 out=dict(valid=p.valid,gold_occurrences=gn,pred_occurrences=sum(map(len,p.spans.values())) if p.valid else 0,boundary_correct_matches=0,duration_correct_matches=0,count_exact=False,route_exact=False,complete_set_01=False,D=score(prefix,row)['total'])
 if not p.valid:return out
 out['route_exact']=set(p.route)==set(route)
 out['count_exact']=out['route_exact'] and all(len(p.spans.get(a,()))==len(gold.get(a,())) for a in route)
 exact=out['count_exact'];duration_exact=out['count_exact']
 for a in route:
  pred=p.spans.get(a,());gs=gold.get(a,())
  if not gs:continue
  if not pred:exact=False;duration_exact=False;continue
  boundary=np.array([[close(x[0],y[0]) and close(x[1],y[1]) for y in gs] for x in pred],dtype=int)
  ii,jj=linear_sum_assignment(-boundary);b=int(boundary[ii,jj].sum());out['boundary_correct_matches']+=b
  # Duration diagnosis uses chronological instance alignment when counts agree;
  # free matching by equal duration would incorrectly accept the wrong instance.
  dc=sum(close(Decimal(str(x[1]))-Decimal(str(x[0])),Decimal(str(y[1]))-Decimal(str(y[0]))) for x,y in zip(sorted(pred),sorted(gs))) if len(pred)==len(gs) else 0
  out['duration_correct_matches']+=dc
  exact=exact and b==len(gs);duration_exact=duration_exact and dc==len(gs)
 out['complete_set_01']=bool(exact);out['complete_duration_set_01']=bool(duration_exact);return out

def summarize(rows):
 n=len(rows);g=sum(r['gold_occurrences'] for r in rows)
 return dict(n=n,format_valid=sum(r['valid'] for r in rows)/n,complete_set_01=sum(r['complete_set_01'] for r in rows)/n,count_exact=sum(r['count_exact'] for r in rows)/n,route_exact=sum(r['route_exact'] for r in rows)/n,boundary_recall_01=sum(r['boundary_correct_matches'] for r in rows)/g if g else None,chronological_duration_recall_01=sum(r['duration_correct_matches'] for r in rows)/g if g else None,mean_D=sum(r['D'] for r in rows)/n)
