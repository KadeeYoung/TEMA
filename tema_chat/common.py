import json,hashlib,os
from pathlib import Path

SYSTEM='''You are answering a continuing conversation about audio clips. Each clip has its own time axis starting at zero. Clips are numbered 1, 2, ... in upload order. Only use audio clips already uploaded and the conversation so far.
Return a complete response in this format:
<think><route>Audios{1},Audios{2}</route><span>Audios{1}[start-end,...];Audios{2}[NONE]</span><reason>Brief explanation.</reason></think><answer>Direct answer to the current question.</answer>
The numbers above illustrate the format; replace them with the actual relevant clip numbers and values. Route lists every audio that must be inspected for the question, including inspected clips with no matching event, not only the winning clip of a comparison. Span lists all occurrences of the relevant event(s) within each routed audio using decimal seconds. Use Audios{k}[NONE] for an inspected audio with no matching occurrence. Do not include unneeded audios. For a question about a gap, Span gives the query-defined gap interval(s) between the specified occurrences; a gap need not be globally silent. Answer the specific question, selecting the requested instance or complete winning set as appropriate.'''

read=lambda p:[json.loads(l) for l in Path(p).open() if l.strip()]

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def digest(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');t.replace(p)

def write_rows(p,rows):
 p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));t.replace(p)
