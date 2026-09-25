#!/usr/bin/env bash
# ExecStartPre guard for mimo-v26-pro.service: do not begin a 550 GiB load until the host can hold it.
#
# 2026-09-23 02:43: a global OOM killed MiMo (it carries oom_score_adj 1000, so it is the chosen victim) when another
# session's Qwen3-Next benchmark server allocated while MiMo held 544 GiB anon. systemd restarted MiMo 30 s later into
# the same crowded memory. Without this guard that is a loop: each attempt reads ~500 GB off the SATA SSD for ~25 min
# and ends in another OOM event that stalls every other job on the box.
#
# 2026-09-23 05:58: a SECOND kind of OOM. MiMo's weights are split across the four NUMA nodes and each quarter is
# mbind()-bound to its node (~136 GiB per node), so a node can run out while the host still has plenty: another
# session's Qwen3-Next server pinned to node 3 (46-52 GiB anon there) left node 3 short, and the kernel OOM-killed MiMo
# with constraint=CONSTRAINT_MEMORY_POLICY nodemask=3 near the end of its repack. So the guard now also requires every
# node to have MIMO_MIN_NODE_GIB (default 150 = 136 weights + KV + compute + drafter + margin) free or reclaimable
# (MemFree + FilePages - Shmem of that node).
#
# Waits until MemAvailable >= MIMO_MIN_AVAIL_GIB (default 575 = 544 GiB steady-state anon + KV + a margin) AND every
# node passes, logging every 5 minutes. The unit's TimeoutStartSec bounds the wait; on timeout systemd retries
# (Restart=on-failure).
set -u
need=${MIMO_MIN_AVAIL_GIB:-575}
need_node=${MIMO_MIN_NODE_GIB:-150}
last=0
node_avail() {   # prints "nodeN:GiB " for every NUMA node
  local d
  for d in /sys/devices/system/node/node[0-9]*; do
    awk -v n="${d##*/}" '/MemFree:/ {f=$4} /FilePages:/ {p=$4} /Shmem:/ {s=$4} END {printf "%s:%d ", n, (f + p - s) / 1048576}' "$d/meminfo"
  done
}
while true; do
  avail=$(awk '/^MemAvailable:/ {printf "%d", $2 / 1048576}' /proc/meminfo)
  nodes=$(node_avail)
  short=$(echo "$nodes" | tr ' ' '\n' | awk -F: -v n="$need_node" 'NF == 2 && $2 < n {printf "%s(%d GiB) ", $1, $2}')
  if [ "$avail" -ge "$need" ] && [ -z "$short" ]; then
    echo "wait-for-ram: MemAvailable ${avail} GiB >= ${need} GiB and every NUMA node >= ${need_node} GiB [${nodes}], starting the load"
    exit 0
  fi
  now=$(date +%s)
  if [ $((now - last)) -ge 300 ]; then
    echo "wait-for-ram: MemAvailable ${avail} GiB (need ${need}); per node [${nodes}] need ${need_node}; short: ${short:-none}. Largest other processes:"
    ps -eo rss,pid,comm --sort=-rss --no-headers | head -6 | awk '{printf "    %6.1f GiB  pid %s  %s\n", $1 / 1048576, $2, $3}'
    last=$now
  fi
  sleep 30
done
