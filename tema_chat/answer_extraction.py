import re
from tema_chat.evaluation import answer_helpers as legacy

EXTRACTION_VERSION='final_answer_excludes_closed_and_unclosed_hidden_blocks.v1'

def extract(text):
 text=text if isinstance(text,str) else ''
 matches=legacy.ANSWER_RE.findall(text)
 if matches:answer=matches[-1]
 else:
  opened=legacy.ANSWER_OPEN_RE.findall(text)
  answer=opened[-1] if opened else text
 # Remove hidden/evidence material even when generation truncates before closing.
 for tag in ['think','route','span','reason']:
  answer=re.sub(r'<'+tag+r'\b[^>]*>.*?</'+tag+r'\s*>','',answer,flags=re.I|re.S)
  answer=re.sub(r'<'+tag+r'\b[^>]*>.*$','',answer,flags=re.I|re.S)
  answer=re.sub(r'<'+tag+r'\b[^>]*$','',answer,flags=re.I|re.S)
 return answer.strip()

def local_tests():
 cases=[('<think>Yes. A shout is present.',''),('<think>reason</think>',''),('<think>reason</think>Yes.','Yes.'),('<think>reason<answer>Yes.','Yes.'),('<think>x</think><answer>Yes.</answer>','Yes.'),('A shout is audible at 5 seconds.','A shout is audible at 5 seconds.'),('<think>x</think><think>unfinished',''),('<route>Audios{1}</route><span>Audios{1}[1-2]</span>','')]
 assert all(extract(a)==b for a,b in cases)
 return dict(passed=True,n=len(cases))
