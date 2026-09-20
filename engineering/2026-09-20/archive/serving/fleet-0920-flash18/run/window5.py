#!/usr/bin/env python3
"""window5.py -- full-model test of (1) a short OpenMP idle spin (GOMP_SPINCOUNT, whole load), (2) the fast sampler candidate path,
(3) coupled draft/verifier sampling, on top of the deployed f18 stack (rev b). Sampled runs are requests exactly as Codex sends them.

Stops production, loads the full model on :18141 with the candidate library, measures modes in ONE loaded
process (in-process switches), then always restores production through systemd.
"""
import json, os, subprocess, sys, time, urllib.request
from pathlib import Path

W = Path('/home/user/InfoSystemic/AI-Server/serving/fleet-0920-flash18')
RUN, RES, BENCH = W/'run', W/'results', W/'bench'
PORT, PROD = 18141, 18131
TAG = sys.argv[1] if len(sys.argv) > 1 else 'w5'
SPIN = os.environ.get('SPIN', '20000')
CPU_LIB = os.environ.get('CPU_LIB', str(W/'deploy-0920b'))
CATALOG = '/home/user/.codex-glm/model-catalogs/glm-5.3-flash.json'
log_path = RES/f'window-{TAG}.log'
report = {'tag': TAG, 'cpu_lib': CPU_LIB, 'started': time.time(), 'runs': []}

def log(*a):
    line = time.strftime('%H:%M:%S ') + ' '.join(str(x) for x in a)
    print(line, flush=True)
    with open(log_path, 'a') as f: f.write(line + '\n')

def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, text=True, capture_output=True, **kw)

def health(port):
    try:
        return b'"ok"' in urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3).read()
    except Exception:
        return False

def idle(port):
    slots = json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/slots', timeout=10))
    return not any(s.get('is_processing') for s in slots)

def listening(port):
    return sh(f"ss -ltn | grep -q ':{port} '").returncode == 0

def completion(port, prompt, n, **extra):
    body = dict(prompt=prompt, n_predict=n, temperature=0, seed=42, cache_prompt=False, stream=False)
    body.update(extra)
    req = urllib.request.Request(f'http://127.0.0.1:{port}/completion', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=3600))

# f18 cpu mask, pool cache, spec mask: 1 merge, 2 fast pick, 4 pad, 8 verifier Gumbel pick, 16 drafter coupled, 32 fast top-k candidates, 64 cheap clone
MODES = {'G': ('3', '1', 7), 'S': ('3', '1', 7 | 32 | 64), 'T': ('3', '1', 7 | 32 | 64 | 8), 'K': ('3', '1', 7 | 32 | 64 | 8 | 16)}

def setmode(m):
    mask, pc, spec = MODES[m]
    assert idle(PORT)
    subprocess.run([str(RUN/'setmode.sh'), mask, pc], check=True)
    import struct
    with open('/dev/shm/f18-spec.u32', 'r+b') as f: f.write(struct.pack('<I', spec))
    time.sleep(0.2)

BATTERY = ["The capital of Australia is", "Water boils at sea level at a temperature of", "The chemical symbol for gold is",
           "In C, the operator to get the address of a variable is", "The derivative of x^3 is", "The year the Berlin Wall fell was",
           "TCP stands for", "The largest planet in the solar system is", "2+2*3 equals", "The author of 'Pride and Prejudice' is"]
P3 = ["Write a detailed technical explanation of how a modern out-of-order CPU core executes instructions, covering fetch, decode, rename, scheduling, execution and retirement.",
      "Implement a thread-safe LRU cache in C++ with O(1) get and put. Explain the data structures, then give the complete code with comments.",
      "Analyse the causes of the 1997 Asian financial crisis, covering capital account liberalisation, currency pegs, short-term external debt and the IMF response."]
SRC = Path('/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx/README.md').read_text()
LONGP = SRC[:24000] + '\n\nSummarise the key findings above in five bullet points.\n'

def battery(mode):
    setmode(mode)
    return [completion(PORT, q, 24)['content'] for q in BATTERY]

def native(mode, n=192):
    setmode(mode)
    out = []
    for p in P3:
        r = completion(PORT, p, n); t = r['timings']
        out.append(dict(tps=t['predicted_per_second'], draft=t.get('draft_n'), acc=t.get('draft_n_accepted'), text=r['content']))
    return out

def probs(mode, prompt, n=64):
    setmode(mode)
    r = completion(PORT, prompt, n, n_probs=3, cache_prompt=True)
    pr = r.get('completion_probabilities', [])
    return dict(text=r['content'], tokens=[x.get('id') for x in pr], logprob=[x.get('logprob') for x in pr], tps=r['timings']['predicted_per_second'],
                draft=r['timings'].get('draft_n'), acc=r['timings'].get('draft_n_accepted'))

def paseo(mode, tag, warm=True, greedy=True):
    setmode(mode)
    cmd = ['python3', str(BENCH/'f18bench.py'), '--catalog', CATALOG, '--tag', tag, '--endpoint', f'http://127.0.0.1:{PORT}']
    cmd += ['--greedy'] if greedy else ['--no-marker']
    if warm: cmd.append('--warm')
    p = subprocess.run(cmd, cwd=BENCH, text=True, capture_output=True)
    r = json.loads((BENCH/f'appserver-{tag}.json').read_text())
    m = r.get('server_metrics', {})
    return dict(mode=mode, tag=tag, rc=p.returncode, tps=m.get('decode_tokens_per_second'), gen=m.get('generated_tokens'), draft=m.get('draft_tokens'),
                acc=m.get('accepted_draft_tokens'), steps=m.get('draft_verification_steps'), cached=m.get('cached_prompt_tokens'), text=r.get('output_text', ''))

def main():
    assert health(PROD) and idle(PROD), 'production must be healthy and idle'
    assert not listening(PORT) or True
    try:
        log('stopping proxy (if any) and production')
        sh(f'{RUN}/stop-port.sh {PORT}')
        sh('systemctl --user stop glm53-flash-production.service')
        for _ in range(180):
            if not listening(PROD): break
            time.sleep(1)
        assert not listening(PROD), 'production port did not clear'
        log('loading candidate on', PORT)
        spin = f' GOMP_SPINCOUNT={SPIN}' if SPIN != 'default' else ''
        r = sh(f'CPU_LIB={CPU_LIB} COMMON_LIB={W}/common/build-couple EXTRA_ENV="GGML_F18_SPEC_CONTROL_FILE=/dev/shm/f18-spec.u32 GGML_F18_SPEC_TRACE=1{spin}" {RUN}/full.sh {TAG}')
        report['spin'] = SPIN; log('GOMP_SPINCOUNT =', SPIN)
        log(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:])
        assert health(PORT), 'candidate did not come up'
        maps = sh(f"grep -o '/home/user[^ ]*\\.so[^ ]*' /proc/$(ss -ltnpH 'sport = :{PORT}' | grep -oE 'pid=[0-9]+' | cut -d= -f2 | head -1)/maps | sort -u").stdout
        report['mapped'] = maps.split(); log('mapped libs:', *[x.split('/')[-2] + '/' + x.split('/')[-1] for x in report['mapped']])

        completion(PORT, 'The capital of France is', 16)  # warmup
        import hashlib
        h = lambda t: hashlib.sha256(t.encode()).hexdigest()[:10]
        # 1. greedy Paseo fixture: the text must stay the production text in every mode
        r0 = paseo('G', f'{TAG}-cold-G', warm=False); report['runs'].append(r0); log('paseo greedy cold G:', r0['tps'], 'hash', h(r0['text']))
        for i, m in enumerate('GSSGKG'):
            r = paseo(m, f'{TAG}-greedy-{i}-{m}'); r['kind'] = 'greedy'; report['runs'].append(r)
            log(f'paseo greedy {i} mode {m}: {r["tps"]:.3f} tok/s gen {r["gen"]} acc {r["acc"]}/{r["draft"]} steps {r["steps"]} hash {h(r["text"])}')
        # 2. drafter invariance on the full model: same seed, verifier-only vs coupled drafter -> same text
        for name, prompt in (('short', P3[1]), ('essay', P3[2])):
            out = {}
            for m in ('T', 'K'):
                setmode(m); r = completion(PORT, prompt, 160, temperature=1.0, seed=4242, cache_prompt=True); t = r['timings']
                out[m] = dict(text=r['content'], tps=t['predicted_per_second'], draft=t.get('draft_n'), acc=t.get('draft_n_accepted'))
            report.setdefault('invariance', {})[name] = out
            log(f'seeded sampled {name}: T acc {out["T"]["acc"]}/{out["T"]["draft"]} {out["T"]["tps"]:.2f} tok/s | K acc {out["K"]["acc"]}/{out["K"]["draft"]} {out["K"]["tps"]:.2f} tok/s | text', 'IDENTICAL' if out['T']['text'] == out['K']['text'] else 'DIFFERS')
        # 3. sampled Paseo fixture, requests exactly as Codex sends them (no sampling fields)
        for i, m in enumerate('SKKSTKSKKSTKSKKSTKSKKS'):
            r = paseo(m, f'{TAG}-sampled-{i}-{m}', greedy=False); r['kind'] = 'sampled'; report['runs'].append(r)
            log(f'paseo sampled {i} mode {m}: {r["tps"]:.3f} tok/s gen {r["gen"]} acc {r["acc"]}/{r["draft"]} = {r["acc"]/max(1,r["draft"]):.3f} steps {r["steps"]}')
        # 4. where the cycle goes now
        p = sh(f'NPRED=200 REPS=1 python3 {RUN}/phase_by_mode.py {PORT} {RUN}/full-{TAG}.log 7 103')
        report['phase'] = p.stdout; log('phase (greedy, native prompt):\n' + p.stdout.strip())
        report['completed'] = True
    except BaseException as e:
        report['error'] = repr(e); log('ERROR', repr(e))
    finally:
        log('restoring production')
        sh(f'{RUN}/stop-port.sh {PORT}')
        sh('systemctl --user start glm53-flash-production.service')
        ok = False
        for i in range(900):
            if health(PROD): ok = True; break
            time.sleep(1)
        report['restored'] = ok; report['finished'] = time.time()
        log('production healthy:', ok, f'window {report["finished"] - report["started"]:.0f}s')
        (RES/f'window-{TAG}.json').write_text(json.dumps(report, indent=1))
        # summary
        by = {}
        for r in report['runs']:
            if r.get('kind') and r.get('tps'): by.setdefault((r['kind'], r['mode']), []).append(r)
        for (kind, m), v in sorted(by.items()):
            tps = [x['tps'] for x in v]; acc = sum(x['acc'] for x in v); dr = sum(x['draft'] for x in v)
            log(f'SUMMARY {kind} mode {m}: mean {sum(tps)/len(tps):.3f} tok/s  n={len(tps)}  acceptance {acc}/{dr} = {acc/max(1,dr):.3f}  [{", ".join(f"{x:.2f}" for x in tps)}]')

if __name__ == '__main__':
    main()
