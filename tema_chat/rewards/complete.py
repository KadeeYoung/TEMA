"""Complete-evidence-priority R; frozen H metric, original D unchanged."""
from tema_chat.rewards.evidence import score as original_D
from tema_chat.rewards.metrics import metrics
VERSION='complete_evidence_H90_D10_v1'
def score(text,row):
 d=original_D(text,row);h=int(metrics(text,row)['complete_set_01'])
 assert not h or d['valid']
 assert not d['valid'] or 0<=d['total']<=1.000000000001
 r=.9*h+.1*d['total'] if d['valid'] else -.1
 return dict(d,total=r,original_D=d['total'],H_evid=h,reward_version=VERSION)
