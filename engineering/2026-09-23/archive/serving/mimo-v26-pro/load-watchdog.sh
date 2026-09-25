#!/usr/bin/env bash
# load-watchdog.sh <server pid>: started by launch-mimo-production.sh just before it execs the server (the pid it
# passes becomes llama-server). Until the server answers /health, every 20 s it checks that every NUMA node can still
# hold the rest of MiMo's share and, if one cannot, kills the load at once instead of letting it run into a per-node
# OOM kill 20-30 minutes later.
#
# Why: MiMo's weights are split across the four nodes and each quarter is mbind()-bound to its node (~136 GiB), so a
# node can run out while the host still has plenty. 2026-09-23 05:58 the kernel OOM-killed MiMo on node 3
# (constraint=CONSTRAINT_MEMORY_POLICY nodemask=3) near the end of a 35-minute reload: another session's benchmark
# (tools/bench_mc.py --node 3, a Qwen3-Next-80B server) had put ~50 GiB there after wait-for-ram.sh had let the load
# start, and it restarts its server within a minute of each run ending.
#
# Estimate per node: MemFree + FilePages - Shmem (free or reclaimable) + MiMo's resident anon / 4 (every tensor is
# split evenly across the nodes, so MiMo grows evenly) must stay >= MIMO_NODE_NEED_GIB (141 = 136 weights + KV +
# compute + drafter) + MIMO_NODE_MARGIN_GIB (4). SIGKILL counts as a failure, so systemd restarts the unit after
# RestartSec and wait-for-ram.sh holds it until every node has room again.
set -u
pid=$1
need=${MIMO_NODE_NEED_GIB:-141}
margin=${MIMO_NODE_MARGIN_GIB:-4}
port=${MIMO_PORT:-18190}
sleep "${MIMO_WATCHDOG_DELAY:-60}"
while kill -0 "$pid" 2>/dev/null; do
  if curl -s -m 3 "http://127.0.0.1:$port/health" 2>/dev/null | grep -q '"ok"'; then
    echo "load-watchdog: server healthy, done"
    exit 0
  fi
  mine=$(awk '/^RssAnon:/ {printf "%d", $2 / 4 / 1048576}' "/proc/$pid/status" 2>/dev/null)
  [ -z "$mine" ] && break
  short=""
  report=""
  for d in /sys/devices/system/node/node[0-9]*; do
    n=${d##*/}
    avail=$(awk '/MemFree:/ {f=$4} /FilePages:/ {p=$4} /Shmem:/ {s=$4} END {printf "%d", (f + p - s) / 1048576}' "$d/meminfo")
    report="$report $n:$((avail + mine))"
    if [ $((avail + mine)) -lt $((need + margin)) ]; then
      short="$short $n($((avail + mine)) GiB)"
    fi
  done
  if [ -n "$short" ]; then
    echo "load-watchdog: node(s)$short can no longer hold MiMo's ${need} GiB share (+${margin} margin; MiMo has ~${mine} GiB per node so far) -- killing the load before the kernel OOM-kills it"
    ps -eo rss,pid,comm --sort=-rss --no-headers | head -5 | awk '{printf "    %6.1f GiB  pid %s  %s\n", $1 / 1048576, $2, $3}'
    kill -KILL "$pid"
    exit 0
  fi
  sleep 20
done
