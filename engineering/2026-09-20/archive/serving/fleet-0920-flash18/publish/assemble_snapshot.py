#!/usr/bin/env python3
"""assemble_snapshot.py -- curated engineering/2026-09-20 snapshot for duck-duck-llama (file operations only).

Sources and notes are copied byte for byte. Evidence JSON goes through the repository exporter's clean() plus a
response-text guard (long strings become length/hash summaries). Text that names a client is redacted and marked filtered.
No weights, binaries, raw traces, process or environment dumps.
"""
import hashlib, json, re, sys, datetime
from pathlib import Path
REPO = Path('/home/user/InfoSystemic/AI-Server/engines/duck-duck-llama-f18')
sys.path.insert(0, str(REPO/'tools'))
from export_tuning_snapshot import clean
import collections

SNAP = REPO/'engineering/2026-09-20'
F = Path('/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx')
S = Path('/home/user/sr950-strategy')
W = Path('/home/user/InfoSystemic/AI-Server/serving/fleet-0920-flash18')
sha = lambda b: hashlib.sha256(b).hexdigest()

keep = [Path(p) for p in json.load(open('/tmp/f18-keep.json'))]
keep = [p for p in keep if p.stat().st_size <= 150_000]
mine = [W/'cpu/f18-common.inc', W/'cpu/fa-mqa.inc', W/'cpu/topk-fast.inc', W/'cpu/build.sh', W/'cpu/test/test_fa.cpp', W/'cpu/test/t16.cpp',
        W/'run/proxy.sh', W/'run/full.sh', W/'run/setmode.sh', W/'run/stop-port.sh', W/'run/parity.py', W/'run/optrace.sh', W/'run/opsum.py', W/'run/window1.py',
        W/'bench/f18bench.py', W/'bench/summarize_metrics.py', W/'common/make_f18.py', W/'common/build.sh',
        W/'results/window-w1.json', W/'results/window-w1.log', W/'publish/tools/topk_select_check.cpp', W/'publish/assemble_snapshot.py']
# revisions b-d (after the first publication): kernels, draft loop, coupled sampling, worker team, query rows, windows w2-w6
mine += [W/'cpu/gdn-rows.inc', W/'cpu/fa-mqa.v1.inc', W/'cpu/make_dispatch.py', W/'cpu/build-dispatch.sh', W/'cpu/build-verify.sh', W/'cpu/build-timing.sh',
         W/'cpu/test/test_fa_invariance.cpp',
         W/'common/make_f18_pad.py', W/'common/make_f18_couple.py', W/'common/build2.sh', W/'common/f18-coupling.inc', W/'common/f18-topk-scan.inc',
         W/'common/test/coupled_sampling_check.cpp', W/'common/test/topk_scan_check.cpp',
         W/'llama/make_ctx.py', W/'llama/make_mtp_qrows.py', W/'llama/build.py', W/'llama/build_glm5next.py',
         W/'run/window2.py', W/'run/window3.py', W/'run/window4.py', W/'run/window5.py', W/'run/window6.py',
         W/'run/couple_check.py', W/'run/fast_sampler_check.py', W/'run/fast_sampler_timing.py', W/'run/phase_by_mode.py', W/'run/phase_timeline.py',
         W/'publish/make_patches_revd.py']
mine += [W/'results'/f'window-{w}.{e}' for w in ('w2', 'w3', 'w4', 'w5', 'w6') for e in ('json', 'log')]
mine += [W/d/n for d in ('deploy-0920b', 'deploy-0920c', 'deploy-0920d') for n in ('95-f18-0920.conf.proposed', 'deploy.sh', 'SHA256SUMS')]
mine += sorted((W/'publish/headers').glob('*.txt'))
# window w7: both sides of the coupled sampler logged, offline replay; the as-deployed sources next to the logging variant
mine += [W/'run/window7.py', W/'run/couple_fit.py', W/'results/window-w7.json', W/'results/window-w7.log', W/'results/couple-fit-w7.txt',
         W/'common/revd/README.md', W/'common/revd/make_f18_couple.py', W/'common/revd/f18-coupling.inc']
# revision e: the Meta backend patch with the recovered libggml-base recipe, windows w8/w9, host profiling helpers
mine += [W/'base/make_meta.py', W/'base/build.sh', W/'run/window8.py', W/'run/window9.py', W/'run/proxy_revd.sh', W/'run/adaptive_depth_sim.py',
         W/'results/window-w8.json', W/'results/window-w8.log', W/'results/window-w9.json', W/'results/window-w9.log']
mine += [W/'deploy-0920e'/n for n in ('95-f18-0920.conf.proposed', 'deploy.sh', 'SHA256SUMS')]
keep += [p for p in mine if p.exists()]
missing = [str(p) for p in mine if not p.exists()]
if missing: print('MISSING from the mine list:', missing)

EXTRA_OMIT = {'output_text', 'outputs', 'generated_text', 'completion', 'battery', 'texts', 'a', 'b'}
def guard(v, counts):
    if isinstance(v, dict):
        out = {}
        for k, c in v.items():
            if k.lower() in EXTRA_OMIT and not isinstance(c, (int, float, bool)):
                counts['removed_fields'] += 1; continue
            out[k] = guard(c, counts)
        return out
    if isinstance(v, list):
        return [guard(c, counts) for c in v]
    if isinstance(v, str) and len(v) > 400:
        counts['summarised_strings'] += 1
        return {'omitted_string_chars': len(v), 'sha256': sha(v.encode())}
    return v

CLIENT = re.compile(r'<client-name-pattern-redacted>', re.I)
def dest(p):
    if str(p).startswith(str(F)):  return Path('archive/serving/fleet-0912-ctx')/p.relative_to(F)
    if str(p).startswith(str(W)):  return Path('archive/serving/fleet-0920-flash18')/p.relative_to(W)
    return Path('archive/sr950-strategy')/p.relative_to(S)

files, counts, skipped = [], collections.Counter(), []
for p in sorted(set(keep)):
    raw = p.read_bytes(); out = raw; filtered = False
    if p.suffix == '.json':
        try:
            data = json.loads(raw)
        except Exception as e:
            skipped.append((str(p), 'invalid json')); continue
        c = collections.Counter()
        data = guard(clean(data, c), c); counts.update(c)
        out = (json.dumps(data, indent=1) + '\n').encode(); filtered = True
    else:
        text = raw.decode('utf-8', errors='replace')
        if CLIENT.search(text):
            if p.suffix in ('.sh',) or 'model-route' in p.name:
                skipped.append((str(p), 'names a client service')); continue
            text = CLIENT.sub('[client]', text); out = text.encode(); filtered = True; counts['redacted_text_files'] += 1
    d = SNAP/dest(p); d.parent.mkdir(parents=True, exist_ok=True); d.write_bytes(out)
    cat = 'engineering-source-or-notes' if p.suffix != '.json' else 'filtered-evidence'
    files.append({'path': dest(p).as_posix(), 'category': cat, 'original_sha256': sha(raw), 'sha256': sha(out), 'bytes': len(out), 'filtered': filtered})

manifest = {'date': '2026-09-20', 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scope': 'GLM-5.3-Flash decode engineering through Paseo/Codex, September 18-20: fused/wider pooling, flat state copy, bounded KV rollback, cache-only MTP catch-up, pooled-result cache, batch-invariant cell-split MQA attention, selection top-k, MTP draft-loop restructure, coupled draft/verifier sampling, sampler host cost, OpenMP team hand-off (spin count and shared team), row-split gated delta net, MTP query rows, thread-free graph-input uploads and blocking dispatch waits in the tensor-parallel backend, production revisions b-e, and the recorded negative results',
    'policy': 'Curated subset. Source/configuration/notes are preserved byte for byte unless they name a client, in which case the name is redacted and the file is marked filtered. Evidence JSON removes host context, environment dumps, commands, prompts and response text; long strings become length/hash summaries. No weights, compiled artifacts, raw traces or per-request captures.',
    'files': files, 'inherited_files': [], 'excluded': [{'source': a, 'reason': b} for a, b in skipped], 'filter_counts': dict(counts)}
(SNAP/'archive-manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
for name in ('source-bundles.json', 'selected-overlays.json', 'correctness-overlays.json'):
    (SNAP/name).write_text('[]\n')
print(len(files), 'files,', sum(f['bytes'] for f in files)//1024, 'KB; filtered', sum(f['filtered'] for f in files), '; skipped', len(skipped), dict(counts))
for a, b in skipped: print('  skipped', a[-80:], '-', b)
