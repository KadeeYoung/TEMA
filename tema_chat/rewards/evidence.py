"""Evidence-only reward: original D, with a strict prefix-only format adapter."""
import re
from tema_chat.rewards.parsing import scored_completion, ParsedCompletion
from tema_chat.rewards.dense import occurrence_components
PREFIX = re.compile(r'\A\s*<think>\s*<route>.*?</route>\s*<span>.*?</span>\s*\Z', re.S)
CLOSE = '</span>'

def evidence(text):
    end = text.find(CLOSE)
    if end < 0: raise ValueError('missing_span_close')
    return text[:end+len(CLOSE)]

def parse(text, lengths=None):
    if not PREFIX.fullmatch(text): return ParsedCompletion(False,error='evidence_structure')
    # Fixed local sentinels only satisfy the legacy syntax parser. They never
    # enter a model, its labels, a reward component, or a stored generated reply.
    return scored_completion(text+'\n<reason>parser</reason>\n</think>\n<answer>parser</answer>', lengths)

def score(text,row):
    p=parse(text,row.get('audio_lengths'))
    if not p.valid: return dict(total=-.1,valid=False,error=p.error,dense_span=0.)
    c=occurrence_components(p,row)
    return dict(total=c['occurrence'],dense_span=c['occurrence'],valid=True,error=None,**c)

def cut_ids(ids,tokenizer):
    """Keep the shortest native-token prefix covering the first closing tag.

    The final token may also contain a suffix (e.g. `></`). It cannot be split
    or replaced without changing the sampled action and its log probability.
    """
    text=tokenizer.decode(ids,skip_special_tokens=False)
    end=text.find(CLOSE)
    if end<0:return list(ids),text
    end+=len(CLOSE)
    for n in range(1,len(ids)+1):
        prefix=tokenizer.decode(ids[:n],skip_special_tokens=False)
        if len(prefix)>=end and CLOSE in prefix:
            return list(ids[:n]),prefix
    raise AssertionError('closing text not represented by sampled IDs')


def cut_for_reward(ids, tokenizer):
    """Separate native training tokens from the character-level evidence view.

    Only text inside the retained closing-boundary token may overhang. Later
    tokens have already been removed by cut_ids. Strict parse/score is unchanged.
    """
    kept, text = cut_ids(ids, tokenizer)
    end = text.find(CLOSE)
    if end < 0:
        return kept, text, text, ''
    end += len(CLOSE)
    return kept, text, text[:end], text[end:]
