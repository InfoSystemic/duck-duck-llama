"""Read-only placement snapshot for a quiescent actual-model packed cache.

Call after warmup, outside request/timing regions:
    snapshot = capture_placement(candidate.cache, max_entries=12)
Cache values are (source_weight, source_scale, packed_object). The helper selects
insertion-order quantiles, inspects tensor metadata only, and queries residency
with move_pages(nodes=NULL). It never reads/touches tensor data, migrates pages,
changes affinity, creates an OpenMP pool, or retains tensor/cache references.
Thread records cover the existing process threads; their runtime ownership cannot
be inferred solely from their names. Capture before/after warmup with `previous`
to obtain per-thread CPU tick deltas without adding a sampling sleep.
"""
from bisect import bisect_right
from collections import Counter
import ctypes
import errno
import hashlib
import os
from pathlib import Path
import platform
import re
import time

SMAPS_FIELDS = {'Size', 'Rss', 'Pss', 'Anonymous', 'AnonHugePages', 'ShmemPmdMapped',
    'FilePmdMapped', 'KernelPageSize', 'MMUPageSize', 'THPeligible', 'VmFlags',
    'Shared_Clean', 'Shared_Dirty', 'Private_Clean', 'Private_Dirty', 'Locked'}
PACKED_FIELDS = ('packed', 'scales', 'sums', 'nan_masks', 'scale')


def _read(path, limit=1024):
    try:
        with Path(path).open() as stream:
            return stream.read(limit).strip()
    except OSError as error:
        return {'error': type(error).__name__, 'errno': error.errno}


def _status(path):
    wanted = {'Name', 'State', 'Cpus_allowed_list', 'Mems_allowed_list', 'Threads', 'VmRSS', 'VmSize'}
    result = {}
    try:
        with Path(path).open() as stream:
            for line in stream:
                key, _, value = line.partition(':')
                if key in wanted:
                    result[key] = value.strip()
    except OSError as error:
        result['error'] = {'type': type(error).__name__, 'errno': error.errno}
    return result


def _key_summary(key):
    if isinstance(key, (tuple, list)):
        return [_key_summary(item) for item in key[:8]]
    if isinstance(key, (int, float, bool)) or key is None:
        return key
    return str(key)[:160]


def _query_pages(addresses):
    if not addresses:
        return dict(return_code=0, errno=0, statuses=[], operation='query-only nodes=NULL flags=0')
    if platform.machine() not in ('x86_64', 'amd64'):
        return dict(return_code=-1, errno=errno.ENOSYS, error_name='ENOSYS', statuses=None,
            operation='query-only unavailable on this architecture')
    pages = (ctypes.c_void_p * len(addresses))(*addresses)
    statuses = (ctypes.c_int * len(addresses))(*([-999] * len(addresses)))
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    ctypes.set_errno(0)
    # x86-64 SYS_move_pages=279. A null nodes pointer only queries placement.
    result = libc.syscall(ctypes.c_long(279), ctypes.c_int(0), ctypes.c_ulong(len(addresses)),
        pages, ctypes.c_void_p(), statuses, ctypes.c_int(0))
    error = ctypes.get_errno() if result < 0 else 0
    return dict(return_code=result, errno=error, error_name=errno.errorcode.get(error) if error else None,
        statuses=list(statuses) if result >= 0 else None, operation='query-only nodes=NULL flags=0')


def _maps():
    records = []
    openmp = {}
    with Path('/proc/self/maps').open() as stream:
        for line in stream:
            fields = line.rstrip().split(None, 5)
            start, end = [int(x, 16) for x in fields[0].split('-')]
            path = fields[5] if len(fields) == 6 else ''
            record = dict(start=start, end=end, permissions=fields[1], offset=fields[2], path=path[:512])
            records.append(record)
            if path and re.search(r'lib(?:gomp|iomp|omp)(?:[-.0-9]|\.so)', Path(path).name, re.IGNORECASE):
                openmp.setdefault(path, []).append([start, end])
    return records, [dict(path=path, mapped_ranges=ranges) for path, ranges in sorted(openmp.items())]


def _mapping_details(wanted):
    details = {start: {} for start in wanted}
    try:
        with Path('/proc/self/numa_maps').open() as stream:
            for line in stream:
                fields = line.split()
                start = int(fields[0], 16)
                if start not in wanted:
                    continue
                nodes, attributes = {}, {}
                for field in fields[1:]:
                    if re.fullmatch(r'N\d+=\d+', field):
                        key, value = field.split('=')
                        nodes[key[1:]] = int(value)
                    elif '=' in field:
                        key, value = field.split('=', 1)
                        if key in {'anon', 'dirty', 'mapped', 'mapmax', 'active', 'kernelpagesize_kB'}:
                            attributes[key] = int(value) if value.isdigit() else value[:128]
                details[start]['numa_mapping'] = dict(policy=fields[1] if len(fields) > 1 else None,
                    node_pages=nodes, attributes=attributes, raw_line=line.strip()[:768])
    except OSError as error:
        for record in details.values():
            record['numa_error'] = {'errno': error.errno, 'type': type(error).__name__}
    try:
        current = None
        with Path('/proc/self/smaps').open() as stream:
            for line in stream:
                if re.match(r'^[0-9a-f]+-[0-9a-f]+ ', line):
                    start = int(line.split('-', 1)[0], 16)
                    current = details[start].setdefault('smaps', {}) if start in details else None
                elif current is not None:
                    key, separator, value = line.partition(':')
                    if separator and key in SMAPS_FIELDS:
                        text = value.strip()
                        if text.endswith(' kB') and text[:-3].strip().isdigit():
                            current[key + '_kB'] = int(text[:-3])
                        elif text.isdigit():
                            current[key] = int(text)
                        else:
                            current[key] = text[:256]
    except OSError as error:
        for record in details.values():
            record['smaps_error'] = {'errno': error.errno, 'type': type(error).__name__}
    return details


def capture_threads(*, max_threads=512, previous=None):
    """Snapshot existing threads without creating workers or sleeping."""
    if not 1 <= max_threads <= 512:
        raise ValueError('max_threads must be between 1 and 512')
    task_root = Path('/proc/self/task')
    tids = sorted(int(path.name) for path in task_root.iterdir() if path.name.isdigit())
    previous_threads = previous.get('threads', previous) if previous else {}
    prior = {row['tid']: row for row in previous_threads.get('records', [])}
    records = []
    for tid in tids[:max_threads]:
        row = dict(tid=tid)
        try:
            stat = (task_root / str(tid) / 'stat').read_text()
            fields = stat.rsplit(')', 1)[1].split()
            row.update(comm=stat[stat.find('(') + 1:stat.rfind(')')][:128], state=fields[0],
                user_ticks=int(fields[11]), system_ticks=int(fields[12]), start_ticks=int(fields[19]),
                last_processor=int(fields[36]))
            status = _status(task_root / str(tid) / 'status')
            row['cpus_allowed_list'] = status.get('Cpus_allowed_list')
            row['mems_allowed_list'] = status.get('Mems_allowed_list')
            before = prior.get(tid)
            if before and before.get('start_ticks') == row['start_ticks']:
                row['cpu_tick_delta'] = row['user_ticks'] + row['system_ticks'] - before['user_ticks'] - before['system_ticks']
        except (OSError, ValueError, IndexError) as error:
            row['error'] = {'type': type(error).__name__, 'errno': getattr(error, 'errno', None)}
        records.append(row)
    affinity_groups = {}
    for row in records:
        affinity_groups.setdefault(row.get('cpus_allowed_list', 'unreadable'), []).append(row['tid'])
    processors = sorted({row['last_processor'] for row in records if 'last_processor' in row})
    topology = {}
    for processor in processors:
        root = Path(f'/sys/devices/system/cpu/cpu{processor}')
        topology[str(processor)] = dict(socket=_read(root / 'topology/physical_package_id', 32),
            core=_read(root / 'topology/core_id', 32), nodes=sorted(path.name for path in root.glob('node[0-9]*')))
    return dict(total_threads=len(tids), sampled_threads=len(records), truncated=len(tids) > max_threads,
        clock_ticks_per_second=os.sysconf('SC_CLK_TCK'), records=records, affinity_groups=affinity_groups,
        last_processor_counts=dict(Counter(str(row['last_processor']) for row in records if 'last_processor' in row)),
        processor_topology=topology,
        limitation='Existing thread metadata; names alone do not establish OpenMP ownership. last_processor is a snapshot, not an affinity binding.')


def capture_placement(cache, *, max_entries=16, max_threads=512, previous=None, label=None):
    """Return a bounded JSON record; cache values must be (weight,scale,packed).

    `max_entries` accepts 1..24; use 12..24 for an actual model. `previous` may
    be a prior placement/thread snapshot to add identity-checked CPU tick deltas.
    The caller must serialize cache mutation and invoke this outside timing.
    """
    if not 1 <= max_entries <= 24:
        raise ValueError('max_entries must be between 1 and 24')
    started = time.time()
    page_size = os.sysconf('SC_PAGE_SIZE')
    count = len(cache)
    take = min(count, max_entries)
    indexes = ({0} if take == 1 else {i * (count - 1) // (take - 1) for i in range(take)}) if take else set()
    entries, pages, buffers = [], [], []
    for index, (key, value) in enumerate(cache.items()):
        if index not in indexes:
            continue
        row = dict(cache_index=index, key=_key_summary(key), buffers=[])
        entries.append(row)
        if not isinstance(value, (tuple, list)) or len(value) < 3:
            row['error'] = 'Expected (source_weight, source_scale, packed_object)'
            continue
        source_weight, source_scale, packed = value[:3]
        components = [('native_weight', source_weight), ('native_scale', source_scale)]
        components += [('packed_' + name, getattr(packed, name)) for name in PACKED_FIELDS if getattr(packed, name, None) is not None]
        aliases = {}
        for component, tensor in components:
            info = dict(component=component)
            row['buffers'].append(info)
            try:
                if getattr(tensor, 'is_meta', False) or getattr(tensor.device, 'type', None) != 'cpu':
                    info['error'] = 'Not a resident CPU tensor'
                    continue
                pointer = tensor.data_ptr()
                size = tensor.numel() * tensor.element_size()
                info.update(address=pointer, bytes=size, shape=list(tensor.shape), dtype=str(tensor.dtype))
                if (pointer, size) in aliases:
                    info['alias_of'] = aliases[(pointer, size)]
                    continue
                aliases[(pointer, size)] = component
                if size == 0:
                    info['samples'] = []
                    continue
                samples = []
                for location, address in [('start', pointer), ('middle', pointer + (size - 1) // 2), ('end', pointer + size - 1)]:
                    page = address - address % page_size
                    samples.append(dict(location=location, page_address=page, query_index=len(pages)))
                    pages.append(page)
                info['samples'] = samples
                buffers.append(info)
            except (AttributeError, TypeError, RuntimeError) as error:
                info['error'] = type(error).__name__ + ': ' + str(error)[:160]
    query = _query_pages(pages)
    statuses = query.pop('statuses')
    mappings, openmp = _maps()
    starts = [mapping['start'] for mapping in mappings]
    selected_maps = {}
    node_counts, page_error_counts = Counter(), Counter()
    for buffer in buffers:
        for sample in buffer['samples']:
            if statuses is not None:
                status = statuses[sample.pop('query_index')]
                sample['node_status'] = status
                if status >= 0:
                    node_counts[str(status)] += 1
                else:
                    sample['page_error'] = errno.errorcode.get(-status, str(-status))
                    page_error_counts[sample['page_error']] += 1
            else:
                sample.pop('query_index')
                sample['node_status'] = None
            index = bisect_right(starts, sample['page_address']) - 1
            if index >= 0 and sample['page_address'] < mappings[index]['end']:
                mapping = mappings[index]
                identity = hex(mapping['start'])
                sample['mapping_id'] = identity
                selected_maps[identity] = dict(mapping)
    details = _mapping_details({mapping['start'] for mapping in selected_maps.values()})
    for mapping in selected_maps.values():
        mapping.update(details[mapping['start']])
    runtime_env = {key: value for key, value in os.environ.items()
        if key.startswith(('OMP_', 'GOMP_', 'KMP_')) or key in ('MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}
    threads = capture_threads(max_threads=max_threads, previous=previous)
    return dict(schema_version=1, label=label, pid=os.getpid(), captured_at=started,
        capture_seconds=time.time() - started, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        page_size=page_size, cache_entries=count, sampled_entries=len(entries), selection_order='insertion-order evenly spaced quantiles',
        max_entries=max_entries, query_page_count=len(pages), sampled_page_node_counts=dict(node_counts),
        sampled_page_error_counts=dict(page_error_counts), move_pages=query, entries=entries, mappings=selected_maps,
        process_status=_status('/proc/self/status'), process_affinity=sorted(os.sched_getaffinity(0)),
        threads=threads, loaded_openmp_libraries=openmp, runtime_environment=runtime_env,
        transparent_hugepage={name: _read('/sys/kernel/mm/transparent_hugepage/' + name)
            for name in ('enabled', 'defrag', 'shmem_enabled', 'hpage_pmd_size')},
        limitations=['Metadata/query only: sampled pages do not prove placement of every cache page.',
            'numa_maps/smaps values cover whole containing mappings, which may include other tensors.',
            'Call outside timing and while cache mutation is serialized. No page migration, prefetch, data reads or affinity changes are performed.'])
