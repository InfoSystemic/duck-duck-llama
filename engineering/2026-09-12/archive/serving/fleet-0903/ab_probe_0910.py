import json,sys,urllib.request,hashlib,statistics,time
label,tp=sys.argv[1],sys.argv[2]
def gen(n=160):
    b=dict(prompt='The quick brown fox',temperature=0,seed=42,n_predict=n,cache_prompt=False,stream=True)
    r=urllib.request.Request('http://127.0.0.1:18133/completion',data=json.dumps(b).encode(),headers={'Content-Type':'application/json'})
    txt=''; n_tok=0; t0=None
    with urllib.request.urlopen(r,timeout=900) as resp:
        for raw in resp:
            raw=raw.decode('utf-8','replace').strip()
            if not raw.startswith('data:'): continue
            try: d=json.loads(raw[5:].strip())
            except Exception: continue
            c=d.get('content','') or ''
            if c:
                if t0 is None: t0=time.perf_counter()   # start clock at first token (skip prefill)
                else: n_tok+=1
                txt+=c
    return txt, (n_tok/(time.perf_counter()-t0) if t0 and n_tok else 0.0)
gen(16)
rates=[]; txt=None
for _ in range(3):
    c,r=gen(160); rates.append(r); txt=c
print(f"{label:16} TP={tp}  median {statistics.median(rates):6.2f} tok/s  runs={[round(x,2) for x in rates]}  sha={hashlib.sha256((txt or '').encode()).hexdigest()[:12]}",flush=True)
