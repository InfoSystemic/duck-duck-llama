#!/usr/bin/env python3
"""window7.py -- log the verifier's candidates and the coupled drafter's top-10 per position on the FULL model under sampled
requests (revision d configuration + a logging libllama-common), so the drafter's temperature can be fitted offline.
Stops production, loads on :18141, runs sampled traffic, always restores production."""
import json, os, subprocess, sys, time, urllib.request
from pathlib import Path
W = Path('/home/user/InfoSystemic/AI-Server/serving/fleet-0920-flash18')
RUN, RES, BENCH = W/'run', W/'results', W/'bench'
PORT, PROD = 18141, 18131
TAG = sys.argv[1] if len(sys.argv) > 1 else 'w7'
CATALOG = '/home/user/.codex-glm/model-catalogs/glm-5.3-flash.json'
LOGF = f'/dev/shm/f18-couple-{TAG}.log'
log_path = RES/f'window-{TAG}.log'
report = {'tag': TAG, 'started': time.time(), 'runs': []}
def log(*a):
    line = time.strftime('%H:%M:%S ') + ' '.join(str(x) for x in a); print(line, flush=True)
    with open(log_path, 'a') as f: f.write(line + '\n')
def sh(cmd, **kw): return subprocess.run(cmd, shell=True, text=True, capture_output=True, **kw)
def health(port):
    try: return b'"ok"' in urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3).read()
    except Exception: return False
def idle(port):
    return not any(s.get('is_processing') for s in json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/slots', timeout=10)))
def listening(port): return sh(f"ss -ltn | grep -q ':{port} '").returncode == 0
def completion(prompt, n, **extra):
    body = dict(prompt=prompt, n_predict=n, cache_prompt=True, stream=False); body.update(extra)
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/completion', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=3600))
def paseo(tag):
    cmd = ['python3', str(BENCH/'f18bench.py'), '--catalog', CATALOG, '--tag', tag, '--endpoint', f'http://127.0.0.1:{PORT}', '--no-marker', '--warm']
    subprocess.run(cmd, cwd=BENCH, text=True, capture_output=True)
    m = json.loads((BENCH/f'appserver-{tag}.json').read_text()).get('server_metrics', {})
    return dict(tag=tag, tps=m.get('decode_tokens_per_second'), gen=m.get('generated_tokens'), draft=m.get('draft_tokens'), acc=m.get('accepted_draft_tokens'))
P3 = ["Write a detailed technical explanation of how a modern out-of-order CPU core executes instructions, covering fetch, decode, rename, scheduling, execution and retirement.",
      "Implement a thread-safe LRU cache in C++ with O(1) get and put. Explain the data structures, then give the complete code with comments.",
      "Analyse the causes of the 1997 Asian financial crisis, covering capital account liberalisation, currency pegs, short-term external debt and the IMF response.",
      "Write a Python module that parses nginx access logs, aggregates requests per minute per status class, and prints a table. Include type hints and tests."]
def main():
    assert health(PROD) and idle(PROD), 'production must be healthy and idle'
    try:
        log('stopping production'); sh(f'{RUN}/stop-port.sh {PORT}'); sh('systemctl --user stop glm53-flash-production.service')
        for _ in range(180):
            if not listening(PROD): break
            time.sleep(1)
        assert not listening(PROD)
        if os.path.exists(LOGF): os.unlink(LOGF)
        env = (f'GOMP_SPINCOUNT=20000 GGML_CPU_NUMA_SHARED_TEAM=1 LLAMA_F18_MTP_QROWS=1 GGML_F18_MTP_PAD_MAX_CTX=1048576 GGML_F18_FEATURES=7 '
               f'GGML_F18_MTP_MERGE=1 GGML_F18_MTP_FAST_PICK=1 GGML_F18_MTP_PAD=1 GGML_F18_FAST_SAMPLER=1 GGML_F18_COUPLED=1 GGML_F18_COUPLE_LOG={LOGF}')
        # no control files: the environment decides, exactly as in production (full.sh would map the cpu control file; mask comes from the file)
        import struct
        open('/dev/shm/f18-control.u32', 'wb').write(struct.pack('<I', 7)); open('/dev/shm/f18-poolcache.u32', 'wb').write(struct.pack('<I', 1))
        r = sh(f'CPU_LIB={W}/cpu/build-dispatch COMMON_LIB={W}/common/build-couple-log LLAMA_LIB={W}/llama/build-qrows EXTRA_ENV="{env}" {RUN}/full.sh {TAG}')
        log(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:]); assert health(PORT)
        completion('The capital of France is', 16, temperature=0)
        r0 = paseo(f'{TAG}-cold'); log('paseo sampled cold:', r0['tps'])
        for i in range(14):
            r = paseo(f'{TAG}-sampled-{i}'); report['runs'].append(r)
            log(f'paseo sampled {i}: {r["tps"]:.3f} tok/s gen {r["gen"]} acc {r["acc"]}/{r["draft"]} = {r["acc"]/max(1,r["draft"]):.3f}')
        for i, p in enumerate(P3 * 2):
            r = completion(p, 320); t = r['timings']
            report.setdefault('native', []).append(dict(i=i, tps=t['predicted_per_second'], draft=t.get('draft_n'), acc=t.get('draft_n_accepted')))
            log(f'native sampled {i}: {t["predicted_per_second"]:.3f} tok/s acc {t.get("draft_n_accepted")}/{t.get("draft_n")}')
        report['completed'] = True
    except BaseException as e:
        report['error'] = repr(e); log('ERROR', repr(e))
    finally:
        log('restoring production'); sh(f'{RUN}/stop-port.sh {PORT}'); sh('systemctl --user start glm53-flash-production.service')
        ok = False
        for i in range(900):
            if health(PROD): ok = True; break
            time.sleep(1)
        report['restored'] = ok; report['finished'] = time.time()
        if os.path.exists(LOGF):
            sh(f'cp {LOGF} {RES}/couple-{TAG}.log'); log('couple log lines:', sh(f'wc -l < {RES}/couple-{TAG}.log').stdout.strip())
        v = [x['tps'] for x in report['runs'] if x.get('tps')]
        if v: log(f'SUMMARY sampled fixture: mean {sum(v)/len(v):.3f} n={len(v)} acceptance {sum(x["acc"] for x in report["runs"])/max(1,sum(x["draft"] for x in report["runs"])):.3f}')
        log('production healthy:', ok, f'window {report["finished"] - report["started"]:.0f}s')
        (RES/f'window-{TAG}.json').write_text(json.dumps(report, indent=1))
if __name__ == '__main__': main()
