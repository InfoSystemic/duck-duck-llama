#!/usr/bin/env python3
"""ppl_glm.py [--env K=V]... [--lib-prepend DIR]... [--ctx 2048] [--chunks 12] [-f text] [-- extra llama-perplexity args]

Perplexity of GLM-5.3-Flash under the PRODUCTION runtime: the model path, device list, tensor split, threads and every
GGML_/LLAMA_/GOMP_ variable are read from the running glm53-flash-production.service process (/proc/PID/{cmdline,environ}), so
the only differences between two runs are the overrides given here. Used as the quality gate for changes that are not
bit-identical (load-time requantisation of the dense weights: GGML_CPU_{ATTN,SHEXP,OUTPUT,DENSE_FFN}_REQUANT=q6_K|q5_K|q4_K).

A second full instance needs ~200 GB more RAM and shares the memory bus with production for the duration (10-20 min at
ctx 2048 x 12 chunks): run it when production is idle and never during a measurement.
"""
import argparse, os, sys, subprocess
from pathlib import Path

TOOL = '/home/user/InfoSystemic/AI-Server/engines/llama.cpp-glm5n-goal-0904/build-goal/bin/llama-perplexity'
FLAGS_KEEP = {'--model', '--device', '--split-mode', '--tensor-split', '--threads', '--threads-batch', '--load-mode', '--fit',
              '--flash-attn', '--cache-type-k', '--cache-type-v', '--gpu-layers', '--n-gpu-layers', '-ngl'}

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--env', action='append', default=[])
    ap.add_argument('--lib-prepend', action='append', default=[])
    ap.add_argument('--ctx', type=int, default=2048)
    ap.add_argument('--chunks', type=int, default=12)
    ap.add_argument('-f', default='/tmp/ppl-qwen.txt')
    ap.add_argument('--dry-run', action='store_true')
    a, extra = ap.parse_known_args()
    pid = subprocess.run(['systemctl', '--user', 'show', '-p', 'MainPID', '--value', 'glm53-flash-production.service'],
                         capture_output=True, text=True).stdout.strip()
    if not pid or pid == '0': sys.exit('production is not running; nothing to copy the runtime from')
    argv = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')[:-1]
    argv = [x.decode() for x in argv]
    penv = dict(x.decode().split('=', 1) for x in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0') if b'=' in x)
    keep = []
    i = 0
    while i < len(argv):
        if argv[i] in FLAGS_KEEP and i + 1 < len(argv):
            keep += [argv[i], argv[i + 1]]; i += 2
        else:
            i += 1
    env = {k: v for k, v in os.environ.items() if not k.startswith(('GGML_', 'LLAMA_', 'GOMP_', 'OMP_'))}
    env.update({k: v for k, v in penv.items() if k.startswith(('GGML_', 'LLAMA_', 'GOMP_', 'OMP_', 'LD_LIBRARY_PATH', 'LIB_PREPEND'))})
    for kv in a.env:
        k, v = kv.split('=', 1); env[k] = v
    for d in reversed(a.lib_prepend):
        d = str(Path(d).resolve()); env['LD_LIBRARY_PATH'] = d + ':' + env.get('LD_LIBRARY_PATH', '')
    inv = ['/usr/bin/taskset', '-c', '0-127', TOOL] + keep + ['-c', str(a.ctx), '-b', str(a.ctx), '--chunks', str(a.chunks), '-f', a.f] + extra
    print('env overrides:', a.env, 'lib prepend:', a.lib_prepend, flush=True)
    print(' '.join(inv[3:]), flush=True)
    if a.dry_run: return 0
    os.execvpe(inv[0], inv, env)

if __name__ == '__main__':
    sys.exit(main() or 0)
