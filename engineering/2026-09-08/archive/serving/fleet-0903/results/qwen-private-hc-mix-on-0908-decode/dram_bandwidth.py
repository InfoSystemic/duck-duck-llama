"""Record all socket IMC channels. Counters measure system-wide DRAM traffic."""
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time


def summarize_samples(samples, start, end):
    chosen = [s for s in samples if s['start'] >= start and s['end'] <= end and s['valid']]
    if not chosen:
        return dict(valid=False, samples=0)
    seconds = sum(s['duration'] for s in chosen)
    sockets = []
    for index in range(len(chosen[0]['sockets'])):
        read = sum(s['sockets'][index]['read_bytes'] for s in chosen) / seconds / 1e9
        write = sum(s['sockets'][index]['write_bytes'] for s in chosen) / seconds / 1e9
        sockets.append(dict(socket=chosen[0]['sockets'][index]['socket'], read_gb_s=read,
                            write_gb_s=write, total_gb_s=read + write))
    read = sum(s['read_gb_s'] for s in sockets)
    write = sum(s['write_gb_s'] for s in sockets)
    return dict(valid=True, samples=len(chosen), sampled_seconds=seconds,
                read_gb_s=read, write_gb_s=write, total_gb_s=read + write,
                utilization_of_380=(read + write)/380, sockets=sockets,
                interval_min_gb_s=min(s['total_gb_s'] for s in chosen),
                interval_median_gb_s=statistics.median(s['total_gb_s'] for s in chosen),
                interval_max_gb_s=max(s['total_gb_s'] for s in chosen))


def parse_records(records, cpus, pmus, socket_ids):
    groups = {}
    errors = []
    for received, line in records:
        row = next(csv.reader([line]))
        if len(row) < 7 or not row[1].startswith('CPU'):
            continue
        try:
            elapsed = float(row[0]); cpu = int(row[1][3:]); value = float(row[2])
            event = row[4].strip(); running = float(row[6])
            pmu, counter, _ = event.split('/')
            if cpu not in cpus or pmu not in pmus or counter not in ('cas_count_read', 'cas_count_write'):
                continue
            scale = {'KiB': 1024, 'MiB': 2**20, 'GiB': 2**30}[row[3].strip()]
        except (ValueError, KeyError):
            errors.append(line.strip())
            continue
        group = groups.setdefault(elapsed, dict(values={}, received=received, valid=True))
        key = (cpu, pmu, counter)
        if key in group['values'] or running < 99:
            group['valid'] = False
        group['values'][key] = value * scale
    if not groups:
        return [], dict(errors=errors, valid=False)
    anchors = [g['received'] - elapsed for elapsed, g in groups.items()]
    anchor = statistics.median(anchors)
    expected = {(c,p,e) for c in cpus for p in pmus for e in ('cas_count_read','cas_count_write')}
    samples = []
    previous = 0
    for elapsed, group in sorted(groups.items()):
        duration = elapsed - previous
        valid = group['valid'] and set(group['values']) == expected and duration > 0
        sockets = []
        for cpu in cpus:
            read = sum(v for (c,p,e),v in group['values'].items() if c == cpu and e == 'cas_count_read')
            write = sum(v for (c,p,e),v in group['values'].items() if c == cpu and e == 'cas_count_write')
            sockets.append(dict(socket=socket_ids[cpu], cpu=cpu, read_bytes=read, write_bytes=write))
        total = sum(s['read_bytes']+s['write_bytes'] for s in sockets)
        samples.append(dict(start=anchor+previous, end=anchor+elapsed, duration=duration,
                            valid=valid, sockets=sockets, total_gb_s=total/duration/1e9))
        previous = elapsed
    return samples, dict(valid=all(s['valid'] for s in samples) and not errors, errors=errors,
                         required_counters_per_interval=len(expected), anchor_monotonic=anchor,
                         anchor_spread_seconds=max(anchors)-min(anchors),
                         timing_note='Align perf elapsed time to reception of each interval; exclude boundary intervals.')


class PerfDramRecorder:
    def __init__(self, directory, interval_ms=500):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        devices = Path('/sys/bus/event_source/devices')
        self.pmus = sorted(p.name for p in devices.glob('uncore_imc_*')
                           if (p/'events/cas_count_read').exists() and (p/'events/cas_count_write').exists())
        assert self.pmus, 'No IMC read/write PMUs'
        masks = {(devices/p/'cpumask').read_text().strip() for p in self.pmus}
        assert len(masks) == 1
        self.cpus = [int(c) for c in masks.pop().split(',')]
        self.socket_ids = {c:int(Path(f'/sys/devices/system/cpu/cpu{c}/topology/physical_package_id').read_text()) for c in self.cpus}
        assert len(set(self.socket_ids.values())) == 4, 'This goal requires all four sockets'
        self.command = ['sudo', '-n', 'perf', 'stat', '-a', '-A', '-C', ','.join(map(str,self.cpus)),
                        '-x', ',', '-I', str(interval_ms)]
        for pmu in self.pmus:
            for counter in ('cas_count_read','cas_count_write'):
                self.command += ['-e', f'{pmu}/{counter}/']
        self.command += ['--', '/bin/cat']
        self.records = []
        self.ready = threading.Event()
        self.process = None

    def start(self):
        (self.directory/'command.json').write_text(json.dumps(self.command, indent=2)+'\n')
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.PIPE, text=True, bufsize=1)
        def consume():
            with (self.directory/'perf.csv').open('w') as log:
                first_elapsed = None
                first_count = 0
                for line in self.process.stderr:
                    self.records.append((time.monotonic(), line))
                    log.write(line); log.flush()
                    row = next(csv.reader([line]))
                    if len(row) >= 7 and row[1].startswith('CPU'):
                        if first_elapsed is None: first_elapsed = row[0]
                        if row[0] == first_elapsed:
                            first_count += 1
                            if first_count == len(self.cpus)*len(self.pmus)*2: self.ready.set()
        self.reader = threading.Thread(target=consume, daemon=True)
        self.reader.start()
        if not self.ready.wait(10):
            self.stop()
            raise RuntimeError('Memory counters failed to produce a complete first interval')
        samples, metadata = parse_records(self.records.copy(), self.cpus, self.pmus, self.socket_ids)
        if not samples or not samples[0]['valid']:
            self.stop()
            raise RuntimeError('Incomplete or multiplexed memory-controller measurements')
        return self

    def stop(self):
        if self.process is None:
            return [], {}
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()  # cat exits; perf finishes without signaling any server
        rc = self.process.wait(timeout=15)
        self.reader.join(timeout=5)
        assert not self.reader.is_alive()
        samples, metadata = parse_records(self.records, self.cpus, self.pmus, self.socket_ids)
        metadata.update(exit_code=rc, pmus=self.pmus, cpus=self.cpus, socket_ids=self.socket_ids,
                        scope='System-wide physical DRAM traffic; includes unrelated host work.')
        (self.directory/'samples.json').write_text(json.dumps(dict(metadata=metadata,samples=samples),indent=2)+'\n')
        return samples, metadata
