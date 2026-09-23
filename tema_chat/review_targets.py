import copy,re
from decimal import Decimal

def prepare_payload(payload):
    p=copy.deepcopy(payload)
    p['temporal_reference_seconds']={k:v for k,v in p['temporal_reference_seconds'].items() if not k.startswith('clip_extent.')}
    refs={};text=p['contract'].get('gold_answer') or ''
    number=r'\d+(?:\.\d+)?'
    unit=r'(?:milliseconds?|ms|seconds?|secs?|s)\b'
    pattern=rf'(?<![\w.])({number})\s*(?:s\s*)?(?:(?:-|–|—|to)\s*({number})\s*)?({unit})'
    for index,m in enumerate(re.finditer(pattern,text,re.I)):
        factor=Decimal('0.001') if m.group(3).lower() in {'ms','millisecond','milliseconds'} else Decimal(1)
        for part in [1,2]:
            if m.group(part) is None:continue
            key=f'gold_text_time[{index}].value{part}'
            p['temporal_reference_seconds'][key]=str(Decimal(m.group(part))*factor);refs[key]=m.group(0)
    p['authoritative_text_time_sources']=refs
    return p
