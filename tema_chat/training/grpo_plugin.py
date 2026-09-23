"""Native single-step complete-evidence R GRPO, actual evidence token IDs, no synthetic EOS."""
import collections,copy,os,json,hashlib
from pathlib import Path
import torch
from accelerate.utils import gather_object
from swift.rewards import ORM,orms
from swift.rlhf_trainers.grpo_trainer import GRPOTrainer
import swift.rlhf_trainers.grpo_trainer as trainer_module
from swift.infer_engine.protocol import RolloutOutput
from swift.infer_engine.vllm_engine import VllmEngine
from tema_chat.training.grpo_audit import Audit,emit,digest
from tema_chat.training.advantage import validate_groups,AdvantageAuditError
from tema_chat.rewards.evidence import cut_for_reward,CLOSE
from tema_chat.rewards.canonical import score
from tema_chat.training.fingerprint import audio_fingerprints

_plan_path=Path(os.environ['TEMA_CURRICULUM_METADATA'])
assert hashlib.sha256(_plan_path.read_bytes()).hexdigest()==os.environ['TEMA_CURRICULUM_METADATA_SHA256']
_plan=[json.loads(l) for l in _plan_path.read_text().splitlines() if l.strip()]
assert len(_plan)==2048 and len({r['span_state_key'] for r in _plan})==2048

_add=VllmEngine._add_stop_words
def add_stop(self,config,request):
 _add(self,config,request)
 if CLOSE in (request.stop or []):config.include_stop_str_in_output=True
VllmEngine._add_stop_words=add_stop
class EvidenceReward(ORM):
 def __call__(self,completions,rollout_infos=None,**kw):return [r['span_scores']['total'] for r in rollout_infos]
orms['tema_evidence_complete']=EvidenceReward
_oldinit=GRPOTrainer.__init__
def init(self,*a,**kw):
 _oldinit(self,*a,**kw)
 assert not self.shuffle_dataset and not self.args.dataset_shuffle and not self.args.train_dataloader_shuffle
 assert len(self.train_dataset)==2048
 self.add_callback(Audit(self))
GRPOTrainer.__init__=init

def infer(self,samples,request_config,is_global_inputs=False):
 assert not is_global_inputs and not self.dynamic_num_samples and not self.multi_turn_scheduler
 grouped=collections.defaultdict(list)
 for s in samples:
  key=s.extra['span_state_key'];grouped[key].append(s);s.prompt_id=key;s.extra['add_eos']=False
 for key,ss in grouped.items():
  assert len(ss)==4 and all(s.messages==ss[0].messages and s.audios==ss[0].audios for s in ss)
 outputs=self._rollout(samples,request_config);result=[];records=[]
 for s,o in zip(samples,outputs):
  ch=o.response.choices[0];rawids=list(ch.token_ids or []);assert rawids
  ids,text,reward_text,overhang=cut_for_reward(rawids,self.template.tokenizer)
  sc=score(reward_text,s.extra);valid=sc['valid']
  if valid:assert self.template.tokenizer.eos_token_id not in ids and reward_text.rstrip().endswith(CLOSE)
  # No tag is injected or corrected: text is decoded from sampled native IDs.
  ch.message.content=text;ch.token_ids=ids
  logs=self._extract_logprobs_from_choice(ch)
  if logs:logs=logs[:len(ids)];assert len(logs)==len(ids)
  info=dict(span_state_key=s.extra['span_state_key'],span_scores=sc,span_text=text,span_reward_text=reward_text,closing_token_overhang=overhang,audios=list(s.audios),num_turns=1)
  msgs=copy.deepcopy(s.messages)+[dict(role='assistant',content=text)]
  out=RolloutOutput(response=o.response,messages=msgs,response_token_ids=[ids],response_loss_mask=[[1]*len(ids)],rollout_infos=info,rollout_logprobs=[logs] if logs else [])
  z=copy.deepcopy(s);z.apply_rollout_output(rollout_output=out);result.append(z)
  records.append(dict(request_id=z.request_id,state_key=s.extra['span_state_key'],scores=sc,text=text,reward_text=reward_text,closing_token_overhang=overhang,tokens=ids,generated_tokens=len(rawids),finish_reason=ch.finish_reason,history_turns=sum(m['role']=='assistant' for m in s.messages)))
 emit('rollouts',dict(step=self.state.global_step+1,candidates=records));return result
GRPOTrainer._infer_single_or_multi_turn=infer
_native_encode=trainer_module.encode_sample
def encode(sample,template,**kw):
 enc=_native_encode(sample,template,**kw)
 if sample.rollout_infos.get('span_state_key') and not kw.get('encode_prompt_only'):
  flat=lambda x:x.tolist() if hasattr(x,'tolist') else list(x)
  ids,labs=flat(enc['input_ids']),flat(enc['labels']);expected=sample.response_token_ids[0];active=[i for i,x in enumerate(labs) if x!=-100]
  assert [labs[i] for i in active]==expected,('Synthetic suffix or token mismatch',len(active),len(expected),labs[-8:],expected[-8:])
  head=template.tokenizer.encode('<|im_start|>assistant\n',add_special_tokens=False);starts=[i+len(head) for i in range(len(ids)-len(head)+1) if ids[i:i+len(head)]==head]
  assert starts and active[0]==starts[-1] and active==list(range(active[0],active[0]+len(expected)))
  aud=audio_fingerprints(enc['input_features'],enc['feature_attention_mask']);assert len(aud)==len(sample.audios)
  ident=digest(dict(prefix_ids=ids[:active[0]],audio_features=aud));sample.rollout_infos['encoded_identity']=ident
  emit('encoding',dict(request_id=sample.request_id,state_key=sample.rollout_infos['span_state_key'],encoded_identity=ident,current_tokens=len(expected),input_tokens=len(ids),history_supervised_tokens=0,synthetic_eos_supervised=False))
 return enc
trainer_module.encode_sample=encode
_native_post=GRPOTrainer._postprocess_batch
def post(self,samples,batches):
 _native_post(self,samples,batches);rows=gather_object([dict(state_key=s.extra['span_state_key'],identity=s.rollout_infos['encoded_identity'],reward=s.rollout_infos['span_scores']['total'],original_D=s.rollout_infos['span_scores']['original_D'],H_evid=s.rollout_infos['span_scores']['H_evid'],valid=s.rollout_infos['span_scores']['valid'],advantage=float(s.advantages)) for s in samples]);groups=collections.defaultdict(list)
 for r in rows:groups[r['state_key']].append(r)
 expected=_plan[self.state.global_step*4:(self.state.global_step+1)*4]
 assert set(groups)=={r['span_state_key'] for r in expected},('Curriculum order changed',self.state.global_step+1,list(groups),expected)
 if self.accelerator.is_main_process:emit('curriculum_order',dict(step=self.state.global_step+1,phase=expected[0]['phase'],state_keys=list(groups),expected_successes=[r['successes'] for r in expected]))
 for rs in groups.values():
  assert len(rs)==4 and len({r['identity'] for r in rs})==1
  for r in rs:assert abs(r['reward']-(.9*r['H_evid']+.1*r['original_D'] if r['valid'] else -.1))<1e-12
  if 0<sum(r['H_evid'] for r in rs)<4:
   assert all((r['advantage']>0 if r['H_evid'] else r['advantage']<0) for r in rs),('Mixed-group sign conflict',rs)
 try:advantage_check=validate_groups(list(groups.values()),self.accelerator.device)
 except AdvantageAuditError as error:
  emit('advantage_check_failure',dict(step=self.state.global_step+1,diagnostics=error.diagnostics));raise
 self._metrics['train']['evidence/zero_reward_std_groups'].append(sum(len({r['reward'] for r in rs})==1 for rs in groups.values())/len(groups))
 self._metrics['train']['evidence/valid_rate'].append(sum(s.rollout_infos['span_scores']['valid'] for s in samples)/len(samples))
 self._metrics['train']['evidence/original_D'].append(sum(s.rollout_infos['span_scores']['original_D'] for s in samples)/len(samples))
 self._metrics['train']['evidence/H_evid'].append(sum(s.rollout_infos['span_scores']['H_evid'] for s in samples)/len(samples))
 if self.accelerator.is_main_process:emit('advantages',dict(step=self.state.global_step+1,groups=list(groups.values()),check=advantage_check))
GRPOTrainer._postprocess_batch=post
