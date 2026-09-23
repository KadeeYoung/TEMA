"""Recover authoritative JSON gold after lossy dataset numeric decoding.

Do not round predictions or enlarge the evidence metric tolerance. The frozen
raw training file supplies gold; dataset-loaded fields are checked for identity
up to negligible serialization error, then replaced only for reward scoring.
"""
import copy,hashlib,json,math,os
from pathlib import Path
from tema_chat.rewards.parsing import normalize_gold_route,normalize_gold_spans,normalize_audio_lengths
from tema_chat.rewards.complete import score as original_score
class CanonicalGold:
 def __init__(self,path,expected_sha256):
  self.path=Path(path);data=self.path.read_bytes();self.sha256=hashlib.sha256(data).hexdigest()
  if self.sha256!=expected_sha256:raise ValueError('Canonical reward dataset hash mismatch')
  self.rows={}
  for line in data.decode().splitlines():
   if not line.strip():continue
   row=json.loads(line);key=row['span_state_key']
   if key in self.rows:raise ValueError('Duplicate canonical state key')
   self.rows[key]={k:copy.deepcopy(row[k]) for k in ['solution','gold_route','gold_spans','audio_lengths','task_type']}
  if not self.rows:raise ValueError('Empty canonical reward dataset')
 def row(self,loaded):
  key=loaded['span_state_key']
  if key not in self.rows:raise ValueError('Unknown canonical state key')
  raw=self.rows[key]
  for field in ['solution','task_type']:
   if loaded.get(field)!=raw[field]:raise ValueError('Canonical gold identity mismatch: '+field)
  if normalize_gold_route(loaded['gold_route'])!=normalize_gold_route(raw['gold_route']):raise ValueError('Gold route changed')
  def same_numbers(a,b,atol):
   if isinstance(b,dict):return isinstance(a,dict) and set(a)==set(b) and all(same_numbers(a[k],v,atol) for k,v in b.items())
   if isinstance(b,(tuple,list)):return isinstance(a,(tuple,list)) and len(a)==len(b) and all(same_numbers(x,y,atol) for x,y in zip(a,b))
   return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a)-float(b))<=atol
  for field,normalizer in [('gold_spans',normalize_gold_spans),('audio_lengths',normalize_audio_lengths)]:
   # datasets Json encodes long audio durations at ten decimal places.
   # This is only an identity guard; scoring still restores exact raw values.
   atol=1e-10 if field=='audio_lengths' else 1e-12
   if not same_numbers(normalizer(loaded[field]),normalizer(raw[field]),atol):raise ValueError('Gold content changed beyond serialization noise: '+field)
  # Leave loaded input/history/actions untouched. Only reward targets are restored.
  return dict(loaded,**copy.deepcopy(raw))
 def score(self,text,loaded):
  result=original_score(text,self.row(loaded))
  return dict(result,gold_numeric_source='raw_json_sha256',gold_source_sha256=self.sha256)
_STORE=None
def score(text,row):
 global _STORE
 if _STORE is None:_STORE=CanonicalGold(os.environ['TEMA_CANONICAL_RL_DATA'],os.environ['TEMA_CANONICAL_RL_SHA256'])
 return _STORE.score(text,row)
