#!/bin/bash
# qsplit_ab.sh <libdir> -- in-process A/B of the column-split activation quantiser (F18 bit 5) on the 8-layer proxy.
# For each arm it flips /dev/shm/f18-control.u32 while the server is idle, runs a warm decode with the op profiler armed,
# and prints the per-matrix MUL_MAT totals. Same process, same load: only the control word differs.
set -euo pipefail
LIBDIR=$(readlink -f "$1")
W=/home/user/InfoSystemic/AI-Server/serving/fleet-0920-flash18
"$W"/run/stop-port.sh 18141 > /dev/null
CTX=32768 LIBS="$LIBDIR" timeout 600 "$W"/run/proxy_revd.sh qsplit | tail -1
for arm in 31 63; do
  python3 -c "import struct,sys; open('/dev/shm/f18-control.u32','wb').write(struct.pack('<I', int(sys.argv[1])))" $arm
  echo "=== control word $arm ($([ $arm = 63 ] && echo 'qsplit ON' || echo 'qsplit off'))"
  ARM=/dev/shm/f18-optrace.arm "$W"/run/optrace.sh 18141 "$W"/run/proxy-qsplit.log "qsplit-$arm" mid
  python3 - "$W/results/optrace-qsplit-$arm.log" <<'PY'
import re, sys, collections
rows = collections.defaultdict(lambda: [0, 0.0])
graphs = []
cur = None
for line in open(sys.argv[1], errors='replace'):
    m = re.search(r'CPU_OP_PROFILE index=(\d+) cpu=(\d+) graph=(\S+) nodes=(\d+) total=([\d.]+)', line)
    if m: cur = (m[3], int(m[4])); graphs.append((cur, float(m[5]))); continue
    m = re.search(r"op=(MUL_MAT|MUL_MAT_ID) time=([\d.]+) ms bar=([\d.]+) ms name='([^']*)' src0_type=(\S+) src0_ne=\[([^\]]*)\]", line)
    if m and cur and cur[1] >= 200:
        k = (m[1], re.sub(r'-\d+$', '', m[4]), m[5], m[6])
        rows[k][0] += 1; rows[k][1] += float(m[2])
big = [t for (g, n), t in graphs if n >= 200]
print(f'  {len(big)} big graphs, mean {sum(big)/max(1,len(big)):.2f} ms')
for k, (n, t) in sorted(rows.items(), key=lambda kv: -kv[1][1])[:12]:
    print(f'  {k[0]:11s} {k[1]:18s} {k[2]:6s} {k[3]:20s} n={n:4d} {t:8.3f} ms')
PY
done
"$W"/run/stop-port.sh 18141 > /dev/null
