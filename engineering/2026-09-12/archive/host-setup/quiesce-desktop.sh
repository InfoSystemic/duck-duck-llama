#!/usr/bin/env bash
# quiesce-desktop.sh - confine interactive/desktop background load to the spare cores
# so it cannot land on the pinned inference worker cores. Fully reversible.
#
# Worker cores (15 threads/socket): 0-14, 16-30, 32-46, 48-62 (+ HT siblings).
# Spare physical cores 15,31,47,63 (+ siblings 79,95,111,127) already host the
# confined daemons (see isolate-background.sh). This script extends the same
# confinement to the desktop apps that isolate-background.sh does not cover:
# Paseo, Thunderbird, Chrome, Xorg, netdata plugins and ad-hoc test runners.
#
# Usage: ./quiesce-desktop.sh on     # confine (saves prior masks)
#        ./quiesce-desktop.sh off    # restore prior masks
#        ./quiesce-desktop.sh status
set -uo pipefail
SPARE="15,31,47,63,79,95,111,127"
STATE="${XDG_CACHE_HOME:-$HOME/.cache}/quiesce-desktop-prev.tsv"
# Never touch these: the model servers, the measurement harness, this shell, the agent.
PROTECT_RE='llama-server|glm-mtp-head|qwen-mtp-head|python3?|whisper-server|claude|bash|sshd|systemd'
# Background to confine.
TARGET_RE='^(Paseo|thunderbird.*|chrome.*|Xorg|netdata|apps\.plugin|orca-ide|node|nautilus|gnome-shell|Discord|slack)$'

targets() {
  ps -eo pid,comm --no-headers | while read -r pid comm; do
    [[ "$pid" == "$$" ]] && continue
    [[ "$comm" =~ ^($PROTECT_RE)$ ]] && continue
    [[ "$comm" =~ $TARGET_RE ]] && printf '%s\n' "$pid"
  done
}

case "${1:-status}" in
  on)
    : > "$STATE"
    n=0; skipped=0
    for pid in $(targets); do
      mask=$(taskset -pc "$pid" 2>/dev/null | sed 's/.*list: //') || { skipped=$((skipped+1)); continue; }
      [[ -z "$mask" ]] && { skipped=$((skipped+1)); continue; }
      [[ "$mask" == "$SPARE" ]] && continue          # already confined; do not record
      printf '%s\t%s\n' "$pid" "$mask" >> "$STATE"
      taskset -acp "$SPARE" "$pid" >/dev/null 2>&1 && n=$((n+1)) || skipped=$((skipped+1))
    done
    echo "confined=$n skipped=$skipped state=$STATE"
    ;;
  off)
    [[ -f "$STATE" ]] || { echo "no saved state at $STATE"; exit 1; }
    n=0
    while IFS=$'\t' read -r pid mask; do
      [[ -d "/proc/$pid" ]] || continue
      taskset -acp "$mask" "$pid" >/dev/null 2>&1 && n=$((n+1))
    done < "$STATE"
    echo "restored=$n"
    ;;
  status)
    echo "spare=$SPARE"
    printf '%-8s %-18s %s\n' PID COMM AFFINITY
    for pid in $(targets); do
      printf '%-8s %-18s %s\n' "$pid" "$(ps -o comm= -p "$pid" 2>/dev/null)" \
        "$(taskset -pc "$pid" 2>/dev/null | sed 's/.*list: //')"
    done
    ;;
  *) echo "usage: $0 on|off|status" >&2; exit 2;;
esac
