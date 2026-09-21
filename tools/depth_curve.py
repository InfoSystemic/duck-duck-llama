#!/usr/bin/env python3
"""depth_curve.py -- decode and prefill rate of the PRODUCTION server as a session grows, without getting in anyone's way.

An agent session is append-only: every turn adds a tool result or a file and asks for the next step. This probe does the same
against /v1/chat/completions: each turn appends one chunk of open-source C/C++ (the llama.cpp tree) plus a fixed instruction,
so the prompt cache carries everything but the new chunk, exactly as it does for a harness. At every depth it generates twice
from the identical prompt: once with the server's default sampling (what Codex traffic gets) and once greedy.

Reported per depth: tokens already cached, tokens prefilled and their rate, and for the decode ms per verify cycle (the cost that
depends on depth), tokens per cycle (the part that depends on the text) and tok/s.

Politeness: the server has two slots and other sessions use it. A watcher polls /slots; the moment a second slot is processing,
this probe drops its own connection (the server cancels the task at the next batch boundary), waits for a long idle gap and
retries the same turn. It gives up after --max-preempt preemptions. It also stops if the unit's cgroup memory nears its limit.
The assistant turns kept in the history are a fixed string, not the generated text, so a rerun with a larger --max-depth
reproduces the same prompt prefix and continues from the cache.
"""
import argparse, http.client, json, os, sys, threading, time
from pathlib import Path

INSTRUCTION = ("\n\nIn one paragraph of about 120 words, say what the code above does, then list three functions it defines "
               "and what each one is for. Do not call any tools.")
SYSTEM = "You are a senior C++ engineer reviewing a repository one file at a time. Answer concisely and precisely."
CANNED = "Noted. Send the next file."

def get_json(port, path, timeout=10):
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=timeout)
    try:
        c.request('GET', path); r = c.getresponse(); return json.loads(r.read())
    finally:
        c.close()

def post_json(port, path, body, timeout=30):
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=timeout)
    try:
        c.request('POST', path, json.dumps(body), {'Content-Type': 'application/json'}); r = c.getresponse()
        return r.status, json.loads(r.read())
    finally:
        c.close()

def busy_slots(port):
    return sum(1 for s in get_json(port, '/slots') if s.get('is_processing'))

def cgroup_gib(path):
    try: return int(Path(path).read_text()) / 2**30
    except (OSError, ValueError): return 0.0

def wait_idle(port, seconds, log):
    """return once no slot has been processing for `seconds` in a row"""
    t0 = time.time(); quiet_since = None
    while True:
        try: n = busy_slots(port)
        except OSError as e: log(f'  /slots failed: {e}'); n = 1
        now = time.time()
        if n == 0:
            quiet_since = quiet_since or now
            if now - quiet_since >= seconds: return now - t0
        else:
            quiet_since = None
        time.sleep(2.0)

class Turn:
    """one request with a watcher that aborts it when somebody else shows up"""
    def __init__(self, port, body, mem_path, mem_stop_gib, log):
        self.port, self.body, self.mem_path, self.mem_stop, self.log = port, body, mem_path, mem_stop_gib, log
        self.conn = None; self.preempted = False; self.mem_abort = False; self.done = threading.Event(); self.shared = False

    def watch(self):
        while not self.done.wait(1.5):
            try: n = busy_slots(self.port)
            except OSError: continue
            if n >= 2:
                self.shared = True; self.preempted = True
                self.log('  another request arrived: dropping ours')
                try: self.conn.sock and self.conn.sock.shutdown(2)
                except OSError: pass
                return
            if self.mem_path and cgroup_gib(self.mem_path) > self.mem_stop:
                self.mem_abort = True
                self.log(f'  cgroup memory above {self.mem_stop:.0f} GiB: stopping')
                try: self.conn.sock and self.conn.sock.shutdown(2)
                except OSError: pass
                return

    def run(self, timeout):
        self.conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=timeout)
        w = threading.Thread(target=self.watch, daemon=True); w.start()
        t0 = time.time()
        try:
            self.conn.request('POST', '/v1/chat/completions', json.dumps(self.body), {'Content-Type': 'application/json'})
            r = self.conn.getresponse(); raw = r.read()
            return r.status, json.loads(raw), time.time() - t0
        except (OSError, http.client.HTTPException, ValueError) as e:
            return None, {'error': repr(e)}, time.time() - t0
        finally:
            self.done.set(); w.join(timeout=5)
            try: self.conn.close()
            except OSError: pass

def load_chunks(root, chars_per_chunk, n_chunks):
    files = sorted(p for pat in ('src/*.cpp', 'ggml/src/*.c', 'ggml/src/*.cpp', 'ggml/src/ggml-cpu/*.c', 'ggml/src/ggml-cpu/*.cpp',
                                 'common/*.cpp', 'tools/server/*.cpp', 'src/models/*.cpp') for p in Path(root).glob(pat))
    text = []
    for p in files:
        try: text.append(f'// ===== file: {p.relative_to(root)} =====\n' + p.read_text(errors='replace'))
        except OSError: pass
    blob = '\n'.join(text)
    need = chars_per_chunk * n_chunks
    assert len(blob) >= need, f'only {len(blob)} characters of source, need {need}'
    return [blob[i*chars_per_chunk:(i + 1)*chars_per_chunk] for i in range(n_chunks)]

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, default=18131)
    ap.add_argument('--step', type=int, default=8192, help='tokens of new context per turn')
    ap.add_argument('--max-depth', type=int, default=131072)
    ap.add_argument('--gen', type=int, default=192)
    ap.add_argument('--source', default=str(Path.home() / 'InfoSystemic/AI-Server/engines/llama.cpp-glm5n-goal-0904'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--idle-s', type=float, default=30.0, help='idle gap required before every turn')
    ap.add_argument('--idle-after-preempt-s', type=float, default=150.0)
    ap.add_argument('--max-preempt', type=int, default=6)
    ap.add_argument('--no-greedy', action='store_true')
    ap.add_argument('--cgroup-memory-current', default='')
    ap.add_argument('--mem-stop-gib', type=float, default=362.0)
    ap.add_argument('--timeout', type=float, default=3600)
    a = ap.parse_args()

    out = Path(a.out); logf = open(out.with_suffix('.log'), 'a')
    def log(msg):
        line = f'{time.strftime("%H:%M:%S")} {msg}'; print(line, flush=True); logf.write(line + '\n'); logf.flush()

    # calibrate characters per token on this tokenizer with a sample of the same material
    sample = load_chunks(a.source, 40000, 1)[0]
    st, tk = post_json(a.port, '/tokenize', {'content': sample})
    cpt = len(sample) / max(1, len(tk.get('tokens', [])))
    n_turns = a.max_depth // a.step
    chunks = load_chunks(a.source, int(a.step * cpt * 0.985), n_turns)   # a little short of the step: the instruction and template fill the rest
    log(f'{cpt:.3f} characters per token; {n_turns} turns of ~{a.step} tokens; generating {a.gen} per pass')

    result = {'started': time.strftime('%F %T %Z'), 'port': a.port, 'step': a.step, 'gen': a.gen, 'points': [], 'preemptions': 0,
              'note': 'production server, host not penned, other sessions may have run between turns'}
    messages = [{'role': 'system', 'content': SYSTEM}]
    n_preempt = 0
    for i, chunk in enumerate(chunks):
        user = {'role': 'user', 'content': f'File batch {i + 1}:\n```cpp\n{chunk}\n```' + INSTRUCTION}
        point = {'turn': i + 1}
        for mode in (['sampled'] if a.no_greedy else ['sampled', 'greedy']):
            body = {'model': 'GLM-5.3-Flash', 'messages': messages + [user], 'max_tokens': a.gen, 'stream': False, 'cache_prompt': True,
                    'chat_template_kwargs': {'reasoning_effort': 'low'}}
            if mode == 'greedy': body.update(temperature=0, seed=42)
            while True:
                waited = wait_idle(a.port, a.idle_s if not n_preempt or point.get('_clean') else a.idle_after_preempt_s, log)
                turn = Turn(a.port, body, a.cgroup_memory_current, a.mem_stop_gib, log)
                status, resp, wall = turn.run(a.timeout)
                if turn.mem_abort:
                    result['stopped'] = 'memory guard'; out.write_text(json.dumps(result, indent=1)); return 2
                if turn.preempted or turn.shared:
                    n_preempt += 1; result['preemptions'] = n_preempt; point['_clean'] = False
                    log(f'  preempted after {wall:.0f} s ({n_preempt}/{a.max_preempt})')
                    if n_preempt >= a.max_preempt:
                        result['stopped'] = 'too many preemptions'; out.write_text(json.dumps(result, indent=1)); return 3
                    continue
                if status != 200 or 'timings' not in resp:
                    log(f'  request failed: status {status} {str(resp)[:300]}')
                    point.setdefault('errors', []).append(str(resp)[:300])
                    if len(point['errors']) >= 2:
                        result['stopped'] = 'request errors'; result['points'].append(point); out.write_text(json.dumps(result, indent=1)); return 4
                    time.sleep(10); continue
                point['_clean'] = True
                break
            t = resp['timings']; usage = resp.get('usage', {})
            cached, fresh = int(t.get('cache_n', 0)), int(t.get('prompt_n', 0))
            n, ms = int(t.get('predicted_n', 0)), float(t.get('predicted_ms', 0.0))
            acc, drafted = int(t.get('draft_n_accepted', 0)), int(t.get('draft_n', 0))
            cycles = max(1, n - acc)
            rec = {'cached': cached, 'prefilled': fresh, 'prefill_s': round(float(t.get('prompt_ms', 0.0))/1000, 2),
                   'prefill_tok_s': round(fresh/(float(t.get('prompt_ms', 1.0))/1000), 2) if fresh else None,
                   'depth': cached + fresh, 'generated': n, 'decode_s': round(ms/1000, 3), 'tok_s': round(n/(ms/1000), 2) if ms else None,
                   'ms_per_cycle': round(ms/cycles, 2), 'tokens_per_cycle': round(n/cycles, 3), 'drafted': drafted, 'accepted': acc,
                   'finish': resp['choices'][0].get('finish_reason'), 'wall_s': round(wall, 1), 'waited_idle_s': round(waited, 1),
                   'cgroup_gib': round(cgroup_gib(a.cgroup_memory_current), 1) if a.cgroup_memory_current else None}
            point[mode] = rec
            log(f"turn {i + 1:2d} {mode:7s} depth {rec['depth']:7d} (cached {cached:7d}, prefilled {fresh:6d} at {rec['prefill_tok_s']} tok/s, {rec['prefill_s']} s) | "
                f"decode {rec['tok_s']} tok/s = {rec['tokens_per_cycle']} tok/cycle / {rec['ms_per_cycle']} ms per cycle | mem {rec['cgroup_gib']} GiB")
            if mode == 'sampled' and i >= 1 and fresh > 2.5 * a.step:
                log('  the prompt cache is not carrying the prefix: stopping rather than re-prefilling everything each turn')
                result['stopped'] = 'prompt cache miss'; result['points'].append(point); out.write_text(json.dumps(result, indent=1)); return 5
        point.pop('_clean', None)
        result['points'].append(point); out.write_text(json.dumps(result, indent=1))
        messages += [user, {'role': 'assistant', 'content': CANNED}]
    result['finished'] = time.strftime('%F %T %Z'); out.write_text(json.dumps(result, indent=1))
    log('done')
    return 0

if __name__ == '__main__':
    sys.exit(main())
