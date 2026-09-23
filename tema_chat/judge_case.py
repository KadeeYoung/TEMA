from .judge_arithmetic import catalog
from tema_chat.evaluation import answer_helpers as legacy

def case_payload(r,c,hist):
 contract={k:c.get(k) for k in ['task_type','operator','question','gold_answer','gold_answer_struct','minimal_acceptable_answer','answer_contract_correction','gold_route','gold_dense_evidence','visible_clip_ids','visible_audio_lengths','history_override','answer_contract_version','answer_contract_precedence','evidence_type','gap_duration_reference']}
 if 'gap_derivation' in c:contract['gap_support']={k:c['gap_derivation'].get(k) for k in ['support_instances','derived_interval','endpoint_selection','raw_duration','quantized_endpoint_duration']}
 return dict(temporal_reference_seconds=catalog(contract),task_type=c['task_type'],time_required=c['task_type'] in legacy.TIME_TASKS,reference_history=hist,contract=contract,candidate_final_answer=legacy.extract_final_answer(r['pred_text']))
