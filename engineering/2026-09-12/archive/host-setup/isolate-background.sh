#!/usr/bin/env bash
# isolate-background.sh - keep background load off the GLM-5.3 worker cores (run as root).
#
# Worker cores with the fast profile (15 threads/socket): 0-14, 16-30, 32-46, 48-62.
# Spare physical cores: 15, 31, 47, 63 (HT siblings 79, 95, 111, 127) -> background daemons + llama main thread.
# CI runners go to the HT siblings 64-127 (they only burst during CI; change SIBLINGS to $SPARE to fully isolate them).
#
# Usage: sudo ./isolate-background.sh          (persistent: writes systemd drop-ins)
#        sudo ./isolate-background.sh --revert
set -euo pipefail
SPARE="15,31,47,63,79,95,111,127"
SIBLINGS="64-127"
UNITS="docker.service containerd.service netdata.service"
# Periodic *user* jobs are the subtler problem: internal-refresh fires every ~5 min unpinned,
# and one descheduled thread on a worker core stalls every socket at each tensor-parallel
# barrier (measured ~30% loss on GLM-5.3). Confine them too.
USER_UNITS="internal-refresh internal-site-cd internal-source-sync internal-games-cd internal-portal-cd metrics-sampler calendar-sync analytics-weekly release-watch"
RUNNERS="ci-runner-1-1 ci-runner-2-1 ci-runner-3-1 ci-runner-4-1"

if [[ "${1:-}" == "--revert" ]]; then
  for u in $UNITS; do systemctl set-property "$u" AllowedCPUs= CPUWeight=; done
  docker update --cpuset-cpus "" --cpu-shares 1024 [client]-db $RUNNERS >/dev/null
  rm -f "$HOME"/.config/systemd/user/*.service.d/10-off-glm-cores.conf
  echo "reverted (daemons, containers and periodic user jobs unconfined)"
  exit 0
fi

# 1) daemons that burn CPU continuously: confine + lower CFS share (cgroup v2 cpuset + cpu.weight)
for u in $UNITS; do
  systemctl set-property "$u" AllowedCPUs="$SPARE" CPUWeight=50
done

# 1b) periodic user jobs (run as the desktop user, so drop-ins under ~/.config/systemd/user)
for u in $USER_UNITS; do
  unit_file="$HOME/.config/systemd/user/$u.service"
  [ -f "$unit_file" ] || continue
  d="$HOME/.config/systemd/user/$u.service.d"
  mkdir -p "$d"
  # NOTE: the user manager gets only `cpu memory pids` delegated (no cpuset), so AllowedCPUs
  # is silently ignored for --user units. CPUAffinity uses sched_setaffinity and does work.
  printf '[Service]\nCPUAffinity=%s\nAllowedCPUs=%s\nCPUWeight=50\nNice=5\n' "${SPARE//,/ }" "$SPARE" > "$d/10-off-glm-cores.conf"
done
su - "$(stat -c %U "$HOME")" -c 'XDG_RUNTIME_DIR=/run/user/$(id -u) systemctl --user daemon-reload' 2>/dev/null || true

# 2) containers (docker update applies live; cpuset + lower shares)
docker update --cpuset-cpus "$SPARE" --cpu-shares 256 [client]-db >/dev/null
for c in $RUNNERS; do docker update --cpuset-cpus "$SIBLINGS" --cpu-shares 256 "$c" >/dev/null; done

# 3) clear the dockerd/containerd event storm; live-restore is enabled so containers keep running
systemctl restart containerd docker

# 4) verify placement and load
sleep 8
echo "--- AllowedCPUs ---"; for u in $UNITS; do echo "$u $(systemctl show "$u" -p AllowedCPUs --value) weight=$(systemctl show "$u" -p CPUWeight --value)"; done
echo "--- containers ---"; docker inspect --format '{{.Name}} cpuset={{.HostConfig.CpusetCpus}} shares={{.HostConfig.CpuShares}}' [client]-db $RUNNERS
echo "--- top CPU users (5 s) ---"; top -b -d 5 -n 2 | awk '/^ *PID/{c++} c==2' | grep -vE "^ *PID" | sort -k9 -rn | head -6 | awk '{print $12, $9"%", "cpu="$0}' | cut -c1-60
