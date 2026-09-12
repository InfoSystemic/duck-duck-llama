#!/usr/bin/env bash
# After the server is up: move threads that still have the full CPU mask (main/HTTP) to the spare cores.
pid=$(pgrep -f 'llama-server.*--port 1809[1]' | head -1); [ -z "$pid" ] && exit 1
n=0; for t in /proc/$pid/task/*; do tid=$(basename $t); m=$(taskset -pc $tid 2>/dev/null | awk '{print $NF}'); if [ "$m" = "0-127" ]; then taskset -pc 15,31,47,63 $tid >/dev/null 2>&1 && n=$((n+1)); fi; done
echo "pinned $n unpinned threads to 15,31,47,63 (of $(ls /proc/$pid/task | wc -l))"
