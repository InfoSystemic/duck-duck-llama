import json,sys,urllib.request,hashlib
label,tp=sys.argv[1],sys.argv[2]
# NOVEL prompts only - never asked of this instance before (ngram-mod replay trap)
P=[('novelA','Describe the life cycle of a cicada and why prime-numbered brood years may be adaptive.'),
   ('novelB','Write a Rust function that computes the Levenshtein distance between two strings.')]
def ask(p,mx=512):
    b=dict(model='dsv4-flash',messages=[dict(role='user',content=p)],temperature=0,seed=42,max_tokens=mx,cache_prompt=False,stream=False)
    r=urllib.request.Request('http://127.0.0.1:18132/v1/chat/completions',data=json.dumps(b).encode(),headers={'Content-Type':'application/json'})
    d=json.load(urllib.request.urlopen(r,timeout=1800)); m=d['choices'][0]['message']
    return (m.get('reasoning_content') or '')+'\x00'+(m.get('content') or ''), d.get('timings',{})
c,_=ask('What is 17 * 23? Reply with only the number.',96)
out=[f"{label:9} TP={tp} correctness={'PASS' if '391' in c else 'FAIL'}"]
for k,p in P:
    t,tm=ask(p)
    out.append(f"  {k} {tm.get('predicted_per_second',0):6.2f} tok/s  sha={hashlib.sha256(t.encode()).hexdigest()[:12]}")
print('\n'.join(out),flush=True)
