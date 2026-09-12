"""Opt-in NUMA row copies of native tensor bytes; no model/cache side effects.

Memory is privately owned anonymous mmap storage. torch.frombuffer retains the
mmap until the last storage alias dies, including aliases wrapped in Parameter.
No affinity or process/thread memory policy is changed. Callers enforce their
own memory budget before calling clone_rows; required_bytes gives that charge.
"""
from collections import Counter
import ctypes
from dataclasses import dataclass
import mmap
import os

import torch

PAGE_SIZE = mmap.PAGESIZE
_numa = ctypes.CDLL('libnuma.so.1', use_errno=True)
_mbind = _numa.mbind
_mbind.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int,
                  ctypes.POINTER(ctypes.c_ulong), ctypes.c_ulong, ctypes.c_uint]
_mbind.restype = ctypes.c_int
_move_pages = _numa.numa_move_pages
_move_pages.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.POINTER(ctypes.c_void_p),
                       ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
_move_pages.restype = ctypes.c_int


@dataclass(frozen=True)
class RowPlacement:
    address: int
    data_bytes: int
    allocated_bytes: int
    page_size: int
    nodes: tuple
    requested_byte_boundaries: tuple
    actual_byte_boundaries: tuple
    boundary_policy: str


def required_bytes(tensor):
    """Anonymous resident-memory charge; excludes the retained source tensor."""
    size = tensor.numel() * tensor.element_size()
    return (size + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_SIZE


def clone_rows(tensor, nodes=(0, 1, 2, 3), *, boundary_policy='strict'):
    """Clone bytes with consecutive row partitions bound to the supplied nodes.

    strict requires every interior row boundary to be page aligned. nearest_page
    explicitly permits boundary pages to cross requested row partitions; actual
    boundaries are recorded on result._goal_numa_0910. Small scale tensors usually
    do not have enough pages for exact quarters and may remain in existing storage.
    An attached tensor.scale reference is retained by identity, without copying it.
    """
    if (type(tensor) not in (torch.Tensor, torch.nn.Parameter)
            or tensor.device.type != 'cpu' or not tensor.is_contiguous()
            or tensor.is_neg() or tensor.is_conj() or tensor.requires_grad
            or tensor.ndim < 1 or tensor.numel() == 0):
        raise ValueError('clone_rows requires a nonempty contiguous, detached CPU tensor')
    nodes = tuple(nodes)
    if not nodes or any(type(n) is not int or not 0 <= n < 64 for n in nodes):
        raise ValueError('nodes must be a nonempty sequence of node IDs in [0, 63]')
    if boundary_policy not in ('strict', 'nearest_page'):
        raise ValueError('boundary_policy must be strict or nearest_page')
    rows = tensor.shape[0]
    row_bytes = tensor.numel() // rows * tensor.element_size()
    data_bytes = tensor.numel() * tensor.element_size()
    allocated_bytes = required_bytes(tensor)
    requested = tuple((rows * i // len(nodes)) * row_bytes for i in range(len(nodes) + 1))
    if boundary_policy == 'strict' and any(b % PAGE_SIZE for b in requested[1:-1]):
        raise ValueError('row partitions are not page aligned; explicitly choose nearest_page or retain source storage')
    boundaries = (0,) + tuple(min(allocated_bytes, (b + PAGE_SIZE // 2) // PAGE_SIZE * PAGE_SIZE)
                             for b in requested[1:-1]) + (allocated_bytes,)

    mapping = mmap.mmap(-1, allocated_bytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                        prot=mmap.PROT_READ | mmap.PROT_WRITE)
    try:
        # Keep the recorded binding granularity independent of transparent huge
        # pages. No physical pages are faulted until all mbind calls succeed.
        mapping.madvise(mmap.MADV_NOHUGEPAGE)
        address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
        for node, begin, end in zip(nodes, boundaries, boundaries[1:]):
            if end == begin:
                continue
            mask = ctypes.c_ulong(1 << node)
            if _mbind(address + begin, end - begin, 2, ctypes.byref(mask), 64, 0) != 0:
                error = ctypes.get_errno()
                raise OSError(error, f'mbind node {node} bytes [{begin}, {end}): {os.strerror(error)}')
        ctypes.memmove(address, tensor.data_ptr(), data_bytes)
        result = torch.frombuffer(mapping, dtype=torch.uint8, count=data_bytes)
        result = result.view(tensor.dtype).reshape(tensor.shape)
    except BaseException:
        mapping.close()
        raise
    result._goal_numa_0910 = RowPlacement(address, data_bytes, allocated_bytes, PAGE_SIZE,
                                         nodes, requested, boundaries, boundary_policy)
    if hasattr(tensor, 'scale'):
        result.scale = tensor.scale
    return result


def query_page_nodes(tensor):
    """Query actual resident page locations without migrating any pages.

    This is a diagnostic, not a per-token operation. It also accepts a Parameter
    that shares cloned storage; a Python metadata attribute is not required.
    """
    address = tensor.data_ptr()
    if address % PAGE_SIZE:
        raise ValueError('query requires a page-aligned tensor start')
    count = required_bytes(tensor) // PAGE_SIZE
    pages = (ctypes.c_void_p * count)(*(address + i * PAGE_SIZE for i in range(count)))
    status = (ctypes.c_int * count)()
    result = _move_pages(0, count, pages, None, status, 0)
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, f'query-only move_pages: {os.strerror(error)}')
    locations = list(status)
    if any(node < 0 for node in locations):
        raise RuntimeError(f'page location query failed: {Counter(locations)}')
    return locations
