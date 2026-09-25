#!/usr/bin/env python3
"""Run a CPU-heavy job on borrowed cores, SIGSTOPping it whenever anyone else needs them.

The job is paused while any of these hold, and resumed once none has held for 1-2x GATE_RESUME_S
(randomized, so two gated jobs queue behind each other instead of livelocking):
  * a llama-server, or any process whose command line mentions /models/ (benchmarks, vision
    probes, converters, hashing), is using CPU -- wherever it runs, since it may be measuring;
  * another process uses more than half a core on the job's CPUs or their hyperthread
    siblings (pinned there, or with a running thread there right now);
  * the hyperthread siblings of the job's CPUs, which the job never uses, carry >= 3 cores.

One busy core stalls every ggml barrier, so contention would also cost the job itself
~1.7x -- and it would skew whatever the other process is measuring. Detection takes ~0.2 s
for a known model process, ~0.5 s for any other process, up to 2 s for diffuse sibling load.

    GATE_CPUS=0-14,16-30 yield-gate.py <command> [args...]    # exit status is the command's

Env: GATE_CPUS (the job's CPU list; without it only the model-process rule applies),
GATE_RESUME_S (default 5), GATE_LOG (default stderr), GATE_DEBUG=1 (per-second stats).
"""
import ctypes
import os
import random
import signal
import subprocess
import sys
import time

POLL_S = 0.1
SCAN_S = 0.5
RESUME_S = float(os.environ.get("GATE_RESUME_S", 5))
HZ = os.sysconf("SC_CLK_TCK")
NCPU = os.cpu_count()
MODEL_TICKS = 0.5 * HZ * POLL_S     # a model process using half a core, per poll
OTHER_TICKS = 0.5 * HZ * SCAN_S     # any other process using half a core, per scan
SIB_WINDOW = 20                     # siblings: 3 cores on average over 2 s, i.e. a parallel build.
SIB_TICKS = 3.0 * HZ * POLL_S * SIB_WINDOW   # The 1-3 cores of desktop/agent-shell noise on
                                             # these 45 CPUs pass; any single process that
                                             # keeps half a core there is a contender anyway.
# Agent CLIs and shells are not model workloads even when their arguments mention /models/;
# the heavy process such a shell starts has its own command line and is judged on that.
NOT_MODEL = {b"claude", b"node", b"codex", b"bash", b"sh", b"dash", b"zsh", b"timeout", b"time"}
DEBUG = os.environ.get("GATE_DEBUG") == "1"
LOG = open(os.environ["GATE_LOG"], "a", buffering=1) if os.environ.get("GATE_LOG") else sys.stderr


def log(msg):
    print(f"[yield-gate {time.strftime('%H:%M:%S')}] {msg}", file=LOG, flush=True)


def cpulist(text):
    cpus = set()
    for part in text.strip().split(","):
        if part:
            lo, _, hi = part.partition("-")
            cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def with_siblings(cpus):
    out = set()
    for c in cpus:
        with open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list") as f:
            out |= cpulist(f.read())
    return out


def busy_ticks(cpus):
    """user+nice+system+irq+softirq+steal ticks summed over cpus, from /proc/stat."""
    total = 0
    with open("/proc/stat") as f:
        for line in f:
            if not line.startswith("cpu"):
                break
            name, *v = line.split()
            if name != "cpu" and int(name[3:]) in cpus:
                total += int(v[0]) + int(v[1]) + int(v[2]) + int(v[5]) + int(v[6]) + int(v[7])
    return total


def stat(path):
    """(state, utime+stime ticks, last cpu, ppid) from /proc/<pid>[/task/<tid>]/stat, or None."""
    try:
        with open(path) as f:
            fields = f.read().rsplit(")", 1)[1].split()
        return fields[0], int(fields[11]) + int(fields[12]), int(fields[36]), int(fields[1])
    except (OSError, IndexError, ValueError):
        return None


def cmdline(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().split(b"\0")
    except OSError:
        return [b""]


def allowed_cpus(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("Cpus_allowed_list:"):
                    return cpulist(line.split(":", 1)[1])
    except OSError:
        pass
    return set()


def running_on(pid, cpus):
    """True if a thread of pid is running or runnable on one of cpus right now."""
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return False
    for tid in tids:
        s = stat(f"/proc/{pid}/task/{tid}/stat")
        if s and s[0] == "R" and s[2] in cpus:
            return True
    return False


def ancestors():
    """The gate's parent chain: the shell or agent that launched the job is part of the job."""
    pids, pid = set(), os.getppid()
    while pid > 1:
        pids.add(pid)
        s = stat(f"/proc/{pid}/stat")
        pid = s[3] if s else 0
    return pids


def die_with_parent():
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG


class Gate:
    def __init__(self, job_cpus):
        self.cpus = with_siblings(job_cpus)                 # the job's CPUs + their siblings
        self.sibs = self.cpus - job_cpus
        self.names = {}                                     # pid -> (exe, is model, is kernel thread)
        self.scan_ticks = {}                                # pid -> ticks at the last scan
        self.model_ticks = {}                               # model pid -> ticks at the last poll
        self.sib_last = busy_ticks(self.sibs)
        self.sib_recent = [0] * SIB_WINDOW
        self.contenders = []
        self.next_scan = 0.0
        self.peak_sib = 0

    def scan(self, exclude):
        """Full /proc pass: refresh the model-process list and find other CPU users on our cores."""
        contenders, seen = [], {}
        for d in os.listdir("/proc"):
            if not d.isdigit() or int(d) in exclude:
                continue
            pid = int(d)
            s = stat(f"/proc/{pid}/stat")
            if s is None:
                continue
            if pid not in self.names:
                argv = cmdline(pid)
                exe = argv[0].rsplit(b"/", 1)[-1]
                is_model = exe == b"llama-server" or (
                    exe not in NOT_MODEL and any(b"/models/" in a for a in argv))
                # Kernel threads (children of kthreadd) mostly do our own reclaim/compaction.
                # Electron rewrites argv[0] into its whole command line, hence the cut.
                name = exe.decode(errors="replace").split(" ")[0][:40] or "kernel"
                self.names[pid] = (name, is_model, s[3] == 2)
            seen[pid] = s[1]
            if not self.cpus or pid not in self.scan_ticks or self.names[pid][1] or self.names[pid][2]:
                continue
            if s[1] - self.scan_ticks[pid] > OTHER_TICKS:
                allowed = allowed_cpus(pid)
                pinned_here = len(allowed) < NCPU and allowed & self.cpus
                if pinned_here or (allowed & self.cpus and running_on(pid, self.cpus)):
                    contenders.append(f"{self.names[pid][0]}:{pid}")
        self.scan_ticks = seen
        self.names = {p: v for p, v in self.names.items() if p in seen}
        self.contenders = contenders

    def reasons(self, now, exclude):
        if now >= self.next_scan:
            self.scan(exclude)
            self.next_scan = now + SCAN_S
        reasons = list(self.contenders)
        for pid, (name, is_model, _) in self.names.items():
            if not is_model:
                continue
            s = stat(f"/proc/{pid}/stat")
            if s is None:
                continue
            # A model process seen for the first time counts as busy if it is running right now.
            prev = self.model_ticks.get(pid)
            if (s[1] - prev > MODEL_TICKS) if prev is not None else s[0] == "R":
                reasons.append(f"{name}:{pid}")
            self.model_ticks[pid] = s[1]
        if self.sibs:
            cur = busy_ticks(self.sibs)
            d, self.sib_last = cur - self.sib_last, cur
            self.peak_sib = max(self.peak_sib, d)
            self.sib_recent = self.sib_recent[1:] + [d]
            if sum(self.sib_recent) >= SIB_TICKS:
                load = sum(self.sib_recent) / (HZ * POLL_S * SIB_WINDOW)
                reasons.append(f"hyperthread siblings busy ({load:.1f} cores over {POLL_S * SIB_WINDOW:.0f}s)")
        return reasons


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    gate = Gate(cpulist(os.environ.get("GATE_CPUS", "")))
    child = subprocess.Popen(sys.argv[1:], preexec_fn=die_with_parent)
    exclude = {os.getpid(), child.pid} | ancestors()
    stopped, pause_start, paused_total, pauses = False, 0.0, 0.0, 0
    last_busy, next_debug, resume_after = -RESUME_S, 0.0, RESUME_S

    def forward(sig, _frame):
        log(f"got {signal.Signals(sig).name}, forwarding it to the job")
        if stopped:
            os.kill(child.pid, signal.SIGCONT)
        child.send_signal(sig)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, forward)

    try:
        while child.poll() is None:
            now = time.monotonic()
            reasons = gate.reasons(now, exclude)
            if reasons:
                last_busy = now
                if not stopped:
                    os.kill(child.pid, signal.SIGSTOP)
                    stopped, pause_start, pauses = True, now, pauses + 1
                    # Jitter: two gated jobs see each other as model processes; a shared fixed
                    # delay would have them resume together and collide forever.
                    resume_after = RESUME_S * random.uniform(1, 2)
                    log(f"paused: {'; '.join(reasons)}")
            elif stopped and now - last_busy >= resume_after:
                os.kill(child.pid, signal.SIGCONT)
                stopped = False
                paused_total += now - pause_start
                log(f"resumed after {now - pause_start:.1f}s")
            if DEBUG and now >= next_debug:
                log(f"siblings peak {gate.peak_sib} ticks/poll, contenders {gate.contenders or '-'}, "
                    f"{'stopped' if stopped else 'running'}")
                gate.peak_sib, next_debug = 0, now + 1
            time.sleep(POLL_S)
    finally:
        if stopped and child.poll() is None:
            os.kill(child.pid, signal.SIGCONT)
    if pauses:
        log(f"{pauses} pause(s), {paused_total:.1f}s paused in total")
    sys.exit(child.returncode if child.returncode >= 0 else 128 - child.returncode)


if __name__ == "__main__":
    main()
