#!/usr/bin/env python3
"""make_patches_revd.py -- (re)generate the 09-20 patch files of duck-duck-llama from the workspace sources.

Each patch = a prose header (what, state, evidence) + unified diffs against the exact parent. The three ggml-cpu ops patches are
cut from ONE modified file by applying hunk subsets in order, so every patch has the right line numbers for its place in the series.
"""
import subprocess, re, hashlib, sys
from pathlib import Path
W    = Path('/home/user/InfoSystemic/AI-Server/serving/fleet-0920-flash18')
F    = Path('/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx')
ENG  = Path('/home/user/InfoSystemic/AI-Server/engines/llama.cpp-glm5n-goal-0904')
REPO = Path('/home/user/InfoSystemic/AI-Server/engines/duck-duck-llama-f18')
WORK = W/'publish/work'; WORK.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def udiff(a, b, la, lb):
    r = subprocess.run(['diff', '-u', '--label', la, '--label', lb, str(a), str(b)], text=True, capture_output=True)
    assert r.returncode in (0, 1), r.stderr
    return r.stdout

def newfile(path, label):
    return udiff('/dev/null', path, '/dev/null', label)

def split_hunks(diff_text):
    lines = diff_text.splitlines(keepends=True)
    head, hunks, cur = lines[:2], [], None
    for l in lines[2:]:
        if l.startswith('@@'):
            cur = [l]; hunks.append(cur)
        else:
            cur.append(l)
    return head, hunks

def apply_hunks(parent, head, hunks, out):
    p = WORK/'sel.diff'; p.write_text(''.join(head) + ''.join(''.join(h) for h in hunks))
    subprocess.run(['patch', '-s', '-o', str(out), str(parent), str(p)], check=True)

# ---- ggml-cpu ops: parent -> +attention -> +top-k -> +gated delta net
P0 = F/'glm-kpool-wide-0919/ops.combo.cpp'
P3 = W/'cpu/ops.f18.cpp'
head, hunks = split_hunks(udiff(P0, P3, 'a/ops.cpp', 'b/ops.cpp'))
kind = lambda h: 'gdn' if 'gdn' in ''.join(h).lower() or 'F18_GDN' in ''.join(h) else ('topk' if ('topk-fast' in ''.join(h) or 'f18_select' in ''.join(h)) else 'fa')
kinds = [kind(h) for h in hunks]
assert kinds.count('gdn') == 2 and kinds.count('topk') == 2 and kinds.count('fa') == 4, kinds
P1, P2 = WORK/'ops.p1.cpp', WORK/'ops.p2.cpp'
apply_hunks(P0, head, [h for h, k in zip(hunks, kinds) if k == 'fa'], P1)
apply_hunks(P0, head, [h for h, k in zip(hunks, kinds) if k in ('fa', 'topk')], P2)
assert udiff(P2, P3, 'a', 'b').count('\n@@') + 1 >= 2
D_FA   = udiff(P0, P1, 'a/ops.cpp', 'b/ops.cpp') + newfile(W/'cpu/f18-common.inc', 'b/f18-common.inc') + newfile(W/'cpu/fa-mqa.inc', 'b/fa-mqa.inc')
D_TOPK = udiff(P1, P2, 'a/ops.cpp', 'b/ops.cpp') + newfile(W/'cpu/topk-fast.inc', 'b/topk-fast.inc')
D_GDN  = udiff(P2, P3, 'a/ops.cpp', 'b/ops.cpp') + newfile(W/'cpu/gdn-rows.inc', 'b/gdn-rows.inc')

# ---- the rest are whole-file diffs
D_SPEC_B = udiff(W/'common/speculative.orig.cpp', W/'common/speculative.f18b.cpp', 'a/common/speculative.cpp', 'b/common/speculative.cpp')
D_SPEC_C = udiff(W/'common/speculative.f18b.cpp', W/'common/revd/speculative.f18c.cpp', 'a/common/speculative.cpp', 'b/common/speculative.cpp')
D_SAMP   = udiff(ENG/'common/sampling.cpp', W/'common/revd/sampling.f18c.cpp', 'a/common/sampling.cpp', 'b/common/sampling.cpp')
D_INC    = newfile(W/'common/revd/f18-coupling.inc', 'b/common/f18-coupling.inc') + newfile(W/'common/f18-topk-scan.inc', 'b/common/f18-topk-scan.inc')
# window w7 only: optional log of both sides + drafter temperature scale, on top of the deployed sources
D_LOG    = (udiff(W/'common/revd/sampling.f18c.cpp', W/'common/sampling.f18c.cpp', 'a/common/sampling.cpp', 'b/common/sampling.cpp')
          + udiff(W/'common/revd/speculative.f18c.cpp', W/'common/speculative.f18c.cpp', 'a/common/speculative.cpp', 'b/common/speculative.cpp')
          + udiff(W/'common/revd/f18-coupling.inc', W/'common/f18-coupling.inc', 'a/common/f18-coupling.inc', 'b/common/f18-coupling.inc'))
D_TEAM   = udiff(ENG/'ggml/src/ggml-cpu/ggml-cpu.cpp', W/'cpu/ggml-cpu.f18.cpp', 'a/ggml/src/ggml-cpu/ggml-cpu.cpp', 'b/ggml/src/ggml-cpu/ggml-cpu.cpp')
D_META   = ('[1] parent against the published engine source\n' + udiff(ENG/'ggml/src/ggml-backend-meta.cpp', F/'glm-fix/ggml-backend-meta.cpp', 'a/ggml/src/ggml-backend-meta.cpp', 'b/ggml/src/ggml-backend-meta.cpp')
          + '\n[2] this patch\n' + udiff(F/'glm-fix/ggml-backend-meta.cpp', W/'base/ggml-backend-meta.f18.cpp', 'a/ggml/src/ggml-backend-meta.cpp', 'b/ggml/src/ggml-backend-meta.cpp'))
D_QROWS  = udiff(F/'glm-mtp-kv-only-0919/candidate/glm5next.cpp', W/'llama/glm5next.f18.cpp', 'a/src/models/glm5next.cpp', 'b/src/models/glm5next.cpp')
assert sha(W/'common/speculative.orig.cpp') == sha(ENG/'common/speculative.cpp'), 'speculative.orig.cpp is not the engine source'

HEADERS = {p.stem: p.read_text() for p in (W/'publish/headers').glob('*.txt')}
OUT = {
 'glm5next-fa-mqa-cellsplit.patch':        D_FA,
 'topk-select-tie-fallback.patch':         D_TOPK,
 'glm5next-gdn-row-split.patch':           D_GDN,
 'mtp-draft-merge-fastpick-pad.patch':     D_SPEC_B,
 'coupled-sampling-fast-sampler.patch':    D_SAMP + D_INC + D_SPEC_C,
 'coupled-sampling-offline-fit.patch':     D_LOG,
 'cpu-numa-shared-team.patch':             D_TEAM,
 'meta-backend-small-uploads-blocking-dispatch.patch': D_META,
 'glm5next-mtp-query-rows.patch':          D_QROWS,
}
for name, body in OUT.items():
    h = HEADERS[name[:-6]].rstrip('\n') + '\n\n'
    assert max(len(l) for l in h.splitlines()) <= 140, name
    (REPO/'patches'/name).write_text(h + body)
    print(f'{name:44s} header {len(h.splitlines()):3d} lines, diff {len(body.splitlines()):5d} lines')
print('parents:', 'ops', sha(P0)[:12], '| speculative', sha(ENG/'common/speculative.cpp')[:12], '| sampling', sha(ENG/'common/sampling.cpp')[:12],
      '| ggml-cpu.cpp', sha(ENG/'ggml/src/ggml-cpu/ggml-cpu.cpp')[:12], '| glm5next candidate', sha(F/'glm-mtp-kv-only-0919/candidate/glm5next.cpp')[:12])
