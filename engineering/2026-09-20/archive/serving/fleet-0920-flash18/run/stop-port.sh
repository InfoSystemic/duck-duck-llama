#!/bin/bash
# stop-port.sh <port>... -- SIGTERM whatever listens on the given ports, wait for the listeners to clear
for prt in "$@"; do
  for p in $(ss -ltnpH "sport = :$prt" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u); do kill "$p" 2>/dev/null; done
done
for i in $(seq 120); do busy=0; for prt in "$@"; do ss -ltn | grep -q ":$prt " && busy=1; done; [ $busy = 0 ] && break; sleep 1; done
echo "ports $* clear"
