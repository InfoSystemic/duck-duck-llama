#!/usr/bin/env python3
"""thread_profile.py PID [seconds] -- where does a multi-threaded server spend host time, without perf, ptrace or root.

Works on any process of the same user. For the given window it reports, per group of threads (pinned to a CPU list vs free to
roam), page faults, user and system CPU time, voluntary and involuntary context switches, threads created, and for the busiest
unpinned threads the run state, kernel wait channel and the CPUs they sat on (sampled every millisecond).

This is how two host-side costs of the CPU-NUMA tensor-parallel backend were found on 2026-09-20, on a machine where
perf_event_paranoid=4 and ptrace_scope=1 rule out perf, gdb -p and strace -p:
  * ~93 thread creations per decode cycle (set_tensor started one std::thread per NUMA device for every graph input); visible here
    as new thread ids, involuntary switches of the PINNED workers (32 per cycle, 6 after the fix) and minor faults;
  * the main thread and eight dispatch threads spinning with sched_yield() on the hyperthread siblings of worker cores: visible as
    unpinned threads in state R with system time far above user time, parked on CPUs N+64.
For the syscall view use strace on a process you start yourself:  strace -f --seccomp-bpf -e trace=mmap,munmap,brk,madvise -o out CMD
(8 MB madvise(MADV_DONTNEED) calls are thread stacks being recycled).

Drive the load yourself while this runs, e.g. one long completion request. Divide by your own cycle count.
"""
import collections, os, sys, time

def snap(pid):
    out = {}
    for t in os.listdir(f'/proc/{pid}/task'):
        try:
            s = open(f'/proc/{pid}/task/{t}/stat').read(); r = s[s.rindex(')') + 2:].split()
            st = open(f'/proc/{pid}/task/{t}/status').read().splitlines()
            get = lambda key: [l.split()[1] for l in st if l.startswith(key)][0]
            out[t] = dict(minflt=int(r[7]), utime=int(r[11]), stime=int(r[12]), aff=get('Cpus_allowed_list'),
                          vol=int(get('voluntary_ctxt_switches')), nonvol=int(get('nonvoluntary_ctxt_switches')))
        except (OSError, IndexError, ValueError):
            pass
    return out

def main():
    pid = int(sys.argv[1]); seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    hz = os.sysconf('SC_CLK_TCK'); ncpu = os.cpu_count()
    all_cpus = f'0-{ncpu - 1}'
    a = snap(pid)
    free = [t for t, v in a.items() if v['aff'] == all_cpus]
    state = collections.defaultdict(collections.Counter); wchan = collections.defaultdict(collections.Counter); where = collections.defaultdict(collections.Counter)
    t_end = time.time() + seconds
    while time.time() < t_end:
        for t in free:
            try:
                s = open(f'/proc/{pid}/task/{t}/stat').read(); r = s[s.rindex(')') + 2:].split()
                state[t][r[0]] += 1; where[t][int(r[36])] += 1
                wchan[t][open(f'/proc/{pid}/task/{t}/wchan').read().strip() or '-'] += 1
            except OSError:
                pass
        time.sleep(0.001)
    b = snap(pid)
    rows = [dict(tid=t, aff=b[t]['aff'], **{k: b[t][k] - a[t][k] for k in ('minflt', 'utime', 'stime', 'vol', 'nonvol')}) for t in b if t in a]
    print(f'pid {pid}, {seconds:.1f} s window: {len(b)} threads, {len([t for t in b if t not in a])} thread ids that did not exist at the start')
    for name, grp in (('pinned', [r for r in rows if r['aff'] != all_cpus]), ('unpinned', [r for r in rows if r['aff'] == all_cpus])):
        print(f"{name:9s} n={len(grp):4d}  faults {sum(r['minflt'] for r in grp):8d}  user {sum(r['utime'] for r in grp)/hz:8.2f} s  sys {sum(r['stime'] for r in grp)/hz:7.2f} s"
              f"  ctx switches voluntary {sum(r['vol'] for r in grp):8d}  involuntary {sum(r['nonvol'] for r in grp):7d}")
    busy = sorted((r for r in rows if r['aff'] == all_cpus), key=lambda r: -(r['utime'] + r['stime']))[:10]
    print('busiest unpinned threads:')
    for r in busy:
        t = r['tid']; n = max(1, sum(state[t].values()))
        print(f"  tid {t}{' (main)' if int(t) == pid else '':7s} user {r['utime']/hz:6.2f} s sys {r['stime']/hz:6.2f} s faults {r['minflt']:7d} | "
              f"state {', '.join(f'{k} {100*v/n:.0f}%' for k, v in state[t].most_common(2))} | wchan {', '.join(f'{k} {100*v/n:.0f}%' for k, v in wchan[t].most_common(2))} | "
              f"CPUs {', '.join(f'{c}:{100*v/n:.0f}%' for c, v in where[t].most_common(3))}")
    worst = sorted((r for r in rows if r['aff'] != all_cpus), key=lambda r: -r['nonvol'])[:5]
    print('pinned threads preempted most often:', ', '.join(f"cpu {r['aff']}: {r['nonvol']}" for r in worst))

if __name__ == '__main__':
    main()
