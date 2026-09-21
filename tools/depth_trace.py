#!/usr/bin/env python3
"""depth_trace.py --turns N --tag TAG -- per-op profile of a production decode at the depth reached by depth_curve.py turn N.

Rebuilds the identical append-only history depth_curve.py sent (same chunks, same canned assistant turns), so the prompt cache
answers it; sends one greedy request and arms the CPU op profiler (GGML_CPU_OP_PROFILE_ARM_FILE) once decode is under way, then
pulls the trace lines out of the journal and summarises the last verify graph with opsum.py.

Refuses to re-prefill from scratch: if the server reports more than --max-prefill new prompt tokens the request is dropped.
"""
import argparse, http.client, json, subprocess, sys, threading, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from depth_curve import INSTRUCTION, SYSTEM, CANNED, load_chunks, post_json, get_json

W = Path('/home/user/InfoSystemic/AI-Server/serving/fleet-0920-flash18')
ARM = '/dev/shm/flash-optrace.arm'

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, default=18131)
    ap.add_argument('--turns', type=int, required=True, help='history length in 8K-token turns (1 = 8K, 16 = 128K)')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--gen', type=int, default=64)
    ap.add_argument('--arm-after', type=float, default=1.5)
    ap.add_argument('--max-prefill', type=int, default=3000)
    ap.add_argument('--step', type=int, default=8192)
    ap.add_argument('--total-turns', type=int, default=16, help='must match the depth_curve run whose cache is reused')
    ap.add_argument('--source', default=str(Path.home() / 'InfoSystemic/AI-Server/engines/llama.cpp-glm5n-goal-0904'))
    ap.add_argument('--final-instruction', default='\n\nNow write about 300 words on how the files seen so far fit together, naming '
                    'the main data structures and the functions that connect them. Do not call any tools.',
                    help='replaces the per-turn instruction on the traced turn so the answer cannot be the canned "Noted." (the prefix '
                         'up to the chunk stays cached; only these tokens are prefilled)')
    a = ap.parse_args()

    sample = load_chunks(a.source, 40000, 1)[0]
    st, tk = post_json(a.port, '/tokenize', {'content': sample})
    cpt = len(sample) / max(1, len(tk.get('tokens', [])))
    chunks = load_chunks(a.source, int(a.step * cpt * 0.985), a.total_turns)[:a.turns]
    messages = [{'role': 'system', 'content': SYSTEM}]
    for i, chunk in enumerate(chunks):
        last = i == len(chunks) - 1
        messages.append({'role': 'user', 'content': f'File batch {i + 1}:\n```cpp\n{chunk}\n```' + (a.final_instruction if last else INSTRUCTION)})
        if not last: messages.append({'role': 'assistant', 'content': CANNED})
    body = {'model': 'GLM-5.3-Flash', 'messages': messages, 'max_tokens': a.gen, 'stream': False, 'cache_prompt': True,
            'temperature': 0, 'seed': 42, 'chat_template_kwargs': {'reasoning_effort': 'low'}}

    if any(s.get('is_processing') for s in get_json(a.port, '/slots')):
        print('server busy; not starting'); return 3
    t_since = int(time.time()) - 2
    conn = http.client.HTTPConnection('127.0.0.1', a.port, timeout=3600)
    state = {'aborted': False, 'armed': False}
    def watch():
        t0 = time.time()
        while not state.get('done'):
            time.sleep(0.5)
            try: slots = get_json(a.port, '/slots', timeout=5)
            except OSError: continue
            mine = [s for s in slots if s.get('is_processing')]
            if any(s.get('n_prompt_tokens_processed', 0) > a.max_prefill for s in mine):
                state['aborted'] = True; print('cache miss: the server is re-prefilling from scratch; dropping the request')
                try: conn.sock and conn.sock.shutdown(2)
                except OSError: pass
                return
            def n_decoded(s):   # /slots reports next_token as a list (one entry per sequence) on this server
                nt = s.get('next_token') or {}
                if isinstance(nt, list): nt = nt[0] if nt else {}
                return int(nt.get('n_decoded', 0) or 0)
            decoding = [s for s in mine if n_decoded(s) >= 1]
            if not state['armed'] and decoding and time.time() - t0 >= a.arm_after:
                # decode has begun (n_decoded counts generated tokens): arm the profiler (skips 4 graphs, records 8)
                Path(ARM).touch(); state['armed'] = True; state['armed_at'] = time.time() - t0
    w = threading.Thread(target=watch, daemon=True); w.start()
    t0 = time.time()
    try:
        conn.request('POST', '/v1/chat/completions', json.dumps(body), {'Content-Type': 'application/json'})
        r = conn.getresponse(); resp = json.loads(r.read())
    except (OSError, http.client.HTTPException, ValueError) as e:
        resp = {'error': repr(e)}
    finally:
        state['done'] = True; w.join(timeout=5); conn.close()
        try: Path(ARM).unlink()
        except FileNotFoundError: pass
    if state['aborted'] or 'timings' not in resp:
        print('no trace:', str(resp)[:300]); return 4
    t = resp['timings']
    print(f"depth {int(t.get('cache_n', 0)) + int(t.get('prompt_n', 0))} (cached {t.get('cache_n')}, prefilled {t.get('prompt_n')}) | "
          f"{t.get('predicted_n')} tokens in {t.get('predicted_ms', 0)/1000:.2f} s = {t.get('predicted_per_second', 0):.2f} tok/s, "
          f"drafts {t.get('draft_n_accepted')}/{t.get('draft_n')}, wall {time.time() - t0:.1f} s, armed {state['armed']} at {state.get('armed_at', -1):.1f} s")
    time.sleep(6.0)
    out = W / 'results' / f'optrace-depth-{a.tag}.log'
    txt = subprocess.run(['journalctl', '--user', '-u', 'glm53-flash-production.service', f'--since=@{t_since}', '-o', 'cat', '--no-pager'],
                         capture_output=True, text=True, errors='replace').stdout
    lines = [l for l in txt.splitlines() if 'CPU_OP_PROFILE' in l]
    out.write_text('\n'.join(lines) + '\n')
    print(f'{len(lines)} trace lines -> {out}')
    subprocess.run([sys.executable, str(W / 'run' / 'opsum.py'), str(out), '500', '45'])
    return 0

if __name__ == '__main__':
    sys.exit(main())
