"""DeepSeek selects semantic references; Decimal applies the frozen time tolerance.

This is not an exact-answer replacement: entity/instance selection, completeness,
paraphrase, counts, polarity and contradiction remain the semantic judge's job.
"""
from decimal import Decimal, InvalidOperation
import re

TOLERANCE = Decimal('0.1')
SEMANTIC_ERRORS = {'NONE','WRONG_POLARITY','WRONG_COUNT','WRONG_AUDIO','WRONG_EVENT','WRONG_ORDER','OMISSION','CONTRADICTION','OTHER','UNJUDGEABLE'}

def catalog(contract):
    out = {}
    def add(k, value):
        if isinstance(value, (int,float)) and not isinstance(value,bool):
            d=Decimal(str(value))
            if d.is_finite():out[k]=str(d)
    def span(k,v):
        if isinstance(v,list) and len(v)==2 and all(isinstance(x,(int,float)) and not isinstance(x,bool) for x in v):
            add(k+'.onset',v[0]);add(k+'.offset',v[1]);out[k+'.duration']=str(Decimal(str(v[1]))-Decimal(str(v[0])))
    def walk(k,v):
        if isinstance(v,dict):
            for key,value in v.items():
                path=k+'.'+key
                if key in {'span','checked_span','window'}:span(path,value)
                elif key=='spans' and isinstance(value,list):
                    for i,item in enumerate(value):span(f'{path}[{i}]',item)
                elif key in {'duration','earliest_onset'}:add(path,value)
                elif key in {'durations','onsets'}:
                    if isinstance(value,dict):
                        for a,t in value.items():add(path+'.'+a,t)
                    elif isinstance(value,list):
                        for i,t in enumerate(value):add(f'{path}[{i}]',t)
                elif isinstance(value,(dict,list)):walk(path,value)
        elif isinstance(v,list):
            for i,x in enumerate(v):walk(f'{k}[{i}]',x)
    walk('answer',contract.get('gold_answer_struct') or {})
    for a,spans in (contract.get('gold_dense_evidence') or {}).items():
        for i,v in enumerate(spans):span(f'evidence.{a}[{i}]',v)
    for a,length in (contract.get('visible_audio_lengths') or {}).items():
        out[f'clip_extent.{a}.onset']='0.0';add(f'clip_extent.{a}.offset',length)
    target=contract.get('gold_answer_struct') or {}
    selected=target.get('span')
    if selected is None and len(target.get('spans') or [])==1:selected=target['spans'][0]
    if isinstance(selected,list) and len(selected)==2:
        out['answer.onset']=str(Decimal(str(selected[0])));out['answer.offset']=str(Decimal(str(selected[1])))
        out.setdefault('answer.duration',str(Decimal(str(selected[1]))-Decimal(str(selected[0]))))
    gap=contract.get('gap_duration_reference') or {}
    if 'canonical_quantized_endpoint_difference' in gap:add('gap.canonical_duration',gap['canonical_quantized_endpoint_difference'])
    return out

def apply_arithmetic(item,case):
    semantic=item.get('semantic_verdict');error=item.get('semantic_error_type');reason=item.get('semantic_reason')
    if semantic not in {'CORRECT','PARTIAL','INCORRECT','UNJUDGEABLE'} or error not in SEMANTIC_ERRORS:raise ValueError('Invalid semantic verdict/error schema')
    if not isinstance(reason,str) or not reason.strip() or len(reason)>200:raise ValueError('Invalid semantic reason')
    if semantic=='CORRECT' and error!='NONE':raise ValueError('Semantic CORRECT requires NONE error')
    if semantic=='UNJUDGEABLE' and error!='UNJUDGEABLE':raise ValueError('UNJUDGEABLE requires matching error')
    checks=item.get('temporal_checks')
    if not isinstance(checks,list):raise ValueError('temporal_checks must be a list, including for non-temporal tasks')
    reference=case['temporal_reference_seconds'];candidate=case['candidate_final_answer'];verified=[];unmatched=[]
    for check in checks:
        if not isinstance(check,dict):raise ValueError('Invalid temporal check')
        ref=check.get('reference_id');quote=check.get('candidate_quote');value=check.get('candidate_seconds')
        if ref not in reference:
            if semantic in {'INCORRECT','UNJUDGEABLE'}:
                unmatched.append(dict(check,unmatched_reason='No such gold temporal reference; independent semantic verdict already rejects or cannot judge this answer.'))
                continue
            raise ValueError(f'Unknown temporal reference: {ref}. For nonexistent events reject the premise semantically; do not invent a time reference. For a correct absence answer do not treat a checking window as a positive event occurrence. Available reference IDs: {list(reference)}')
        if not isinstance(quote,str) or not quote.strip() or quote not in candidate:raise ValueError('Temporal quote must be an exact nonempty candidate substring')
        if not isinstance(value,str):raise ValueError('candidate_seconds must be a decimal string')
        try:d=Decimal(value)
        except InvalidOperation:raise ValueError('Invalid candidate seconds')
        if not d.is_finite():raise ValueError('Nonfinite candidate seconds')
        # A judge must not silently replace the actual candidate decimal with gold.
        nums=[Decimal(x) for x in re.findall(r'(?<![\w.])-?\d+(?:\.\d+)?',quote)]
        if nums:
            supported=set(nums)
            if re.search(r'milliseconds?|\bms\b',quote,re.I):supported.update(x/1000 for x in nums)
            if re.search(r'minutes?|\bmin\b',quote,re.I):supported.update(x*60 for x in nums)
            for mm,ss in re.findall(r'(\d+):(\d+(?:\.\d+)?)',quote):supported.add(Decimal(mm)*60+Decimal(ss))
            compound=re.search(r'(\d+(?:\.\d+)?)\s*(?:minutes?|min)\s*(?:and\s*)?(\d+(?:\.\d+)?)\s*(?:seconds?|s)\b',quote,re.I)
            if compound:supported.add(Decimal(compound[1])*60+Decimal(compound[2]))
            if d not in supported:raise ValueError('Candidate normalized number is not supported by its verbatim quote')
        gold=Decimal(reference[ref]);difference=abs(d-gold)
        verified.append(dict(candidate_quote=quote,candidate_seconds=str(d),reference_id=ref,reference_seconds=str(gold),absolute_error_seconds=str(difference),within_tolerance=difference<=TOLERANCE))
    failed=[v for v in verified if not v['within_tolerance']]
    # Numerical distance must never be hidden in a semantic error instead of a check.
    if semantic in {'PARTIAL','INCORRECT'} and not failed and re.search(r'outside.*toler|exceed.*toler|beyond.*toler',reason,re.I):
        raise ValueError('Semantic reason illegally applies a numerical tolerance that Decimal accepts')
    verdict=semantic;final_error=error;final_reason=reason
    if failed and semantic!='UNJUDGEABLE':
        verdict='INCORRECT';final_error='WRONG_DURATION' if '.duration' in failed[0]['reference_id'] or failed[0]['reference_id'].endswith('canonical_duration') else 'WRONG_TIME'
        f=failed[0];final_reason=f"Temporal claim {f['candidate_seconds']}s vs {f['reference_seconds']}s: exact error {f['absolute_error_seconds']}s > 0.1s."
    time_required=case['time_required'];time_match=('MATCH' if verdict=='CORRECT' else 'MISMATCH') if time_required else 'NOT_APPLICABLE'
    return dict(id=item['id'],verdict=verdict,score={'CORRECT':1.,'PARTIAL':.5,'INCORRECT':0.,'UNJUDGEABLE':None}[verdict],time_required=time_required,time_match=time_match,error_type=final_error,reason=final_reason[:200],semantic_verdict=semantic,semantic_error_type=error,semantic_reason=reason,temporal_checks=verified,unmatched_temporal_claims=unmatched,arithmetic_version='Decimal_inclusive_0p1.v1')

def validate_response(response,batch):
    items=response if isinstance(response,list) else response.get('items') if isinstance(response,dict) else None
    if not isinstance(items,list) or [i.get('id') if isinstance(i,dict) else None for i in items]!=[c['id'] for c in batch]:raise ValueError('Semantic response IDs/order mismatch')
    return [apply_arithmetic(item,case) for item,case in zip(items,batch)]
