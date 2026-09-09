"""Aggregate private-model page backing outside timed inference windows."""
from pathlib import Path
import re
import time


def audit_pages(pid):
    proc = Path('/proc') / str(pid)
    def identity():
        return (proc / 'stat').read_text().rsplit(') ', 1)[1].split()[19]
    start = identity()
    begun = time.monotonic()
    text = (proc / 'smaps').read_text()
    fields = ('Size', 'Rss', 'Anonymous', 'AnonHugePages', 'Swap')
    groups = {key: dict(mappings=0, **{field: 0 for field in fields})
              for key in ('hugepage_advised', 'other')}
    current = None
    def finish():
        if current is None:
            return
        group = groups['hugepage_advised' if current['advised'] else 'other']
        group['mappings'] += 1
        for field in fields:
            group[field] += current[field]
    for line in text.splitlines():
        if re.match(r'^[0-9a-f]+-[0-9a-f]+ ', line):
            finish()
            current = dict(advised=False, **{field: 0 for field in fields})
        elif current is not None and ':' in line:
            key, value = line.split(':', 1)
            if key in fields:
                amount, unit = value.split()
                assert unit == 'kB'
                current[key] = int(amount) * 1024
            elif key == 'VmFlags':
                current['advised'] = 'hg' in value.split()
    finish()
    assert identity() == start, 'Process identity changed during page audit'
    totals = {field: sum(group[field] for group in groups.values()) for field in fields}
    assert totals['Rss'] > 0 and 0 <= totals['AnonHugePages'] <= totals['Anonymous']
    return dict(pid=pid, start_ticks=start, time=time.time(), elapsed_seconds=time.monotonic() - begun,
                units='bytes', totals=totals, groups=groups,
                anonymous_hugepage_fraction=totals['AnonHugePages'] / totals['Anonymous'] if totals['Anonymous'] else 0,
                scope='Per-process smaps totals; collected outside timed decode. No addresses or mapping paths retained.')
