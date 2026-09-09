#!/usr/bin/env python3
"""Check channel accounting, units, duration weighting and invalid counters."""
from dram_bandwidth import parse_records, summarize_samples

cpus = [0,16,32,48]
pmus = [f'uncore_imc_{i}' for i in range(6)]
sockets = dict(zip(cpus,range(4)))
records = []
for elapsed, duration in [(1,1),(3,2)]:
    for cpu in cpus:
        for pmu in pmus:
            for direction in ['read','write']:
                records.append((100+elapsed, f'{elapsed},CPU{cpu},{100*duration},MiB,{pmu}/cas_count_{direction}/,1000000000,100.00,,\n'))
samples, meta = parse_records(records,cpus,pmus,sockets)
assert meta['valid'] and len(samples)==2 and meta['required_counters_per_interval']==48
summary = summarize_samples(samples,100,104)
assert summary['samples']==2 and summary['sampled_seconds']==3
assert abs(summary['total_gb_s']-48*100*2**20/1e9)<1e-12
assert abs(summary['sockets'][0]['total_gb_s']-12*100*2**20/1e9)<1e-12
assert summarize_samples(samples,101,103)['samples']==1
bad,_ = parse_records(records[:-1],cpus,pmus,sockets)
assert not bad[-1]['valid']
bad,_ = parse_records(records+[records[-1]],cpus,pmus,sockets)
assert not bad[-1]['valid']
bad,_ = parse_records([(t,l.replace(',100.00,',',75.00,')) for t,l in records],cpus,pmus,sockets)
assert not any(s['valid'] for s in bad)
bad,meta = parse_records([(t,l.replace(',100,MiB,',',<not counted>,MiB,')) for t,l in records],cpus,pmus,sockets)
assert not meta['valid'] and meta['errors']
print('PASS: 48-counter accounting, byte units, duration weighting, window selection, missing/duplicate/multiplexed/uncounted rejection')
