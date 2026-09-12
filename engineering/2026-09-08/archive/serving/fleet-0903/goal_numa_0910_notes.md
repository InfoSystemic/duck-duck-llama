The helper is a component candidate and has not copied any real model weights.

```python
from goal_numa_0910 import clone_rows, required_bytes, query_page_nodes

charge = required_bytes(parameter)  # Check caller's anonymous-memory cap first.
copied = clone_rows(parameter, (0, 1, 2, 3))
placement = copied._goal_numa_0910  # Save diagnostic metadata before wrapping.
replacement = torch.nn.Parameter(copied, requires_grad=False)
```

For `ResidentStore`, copy after `load_parameters` in activation, before publishing
the expert as ready. Copy matrix weights at least 1 MiB; keep small scales in
their existing storage. Stage replacements until all copies succeed, then assign
the new Parameters into each submodule's `_parameters`. Reattach
`sub.weight.scale = sub.scale` after replacement. The helper itself retains an
existing `.scale` attribute by identity, but Parameter construction does not copy
arbitrary Python attributes.

Charge `required_bytes` for each live private copy against the caller's memory
budget. This is additional anonymous memory; copying a file-backed tensor does
not immediately evict the source file pages from the kernel's page cache. Do not
keep old tensors, copied tensors, or replacements in a persistent staging list.
Metadata (`RowPlacement`) contains only integers/tuples and does not retain storage.

The existing `ResidentStore.release` replacement with meta Parameters drops the
module's owned copies. The mmap is released when the final tensor/storage alias
dies. Verification confirms Parameter wrapping preserves storage lifetime and
final release unmaps it. No explicit `mmap.close` or tensor-based finalizer should
be added: either could invalidate a surviving storage alias.

Default strict mode requires interior row boundaries to be page aligned. For
unaligned small scales, retaining their original storage is simplest. Explicit
`boundary_policy='nearest_page'` records the actual rounded boundaries and can
assign no pages to some nodes when the entire tensor contains fewer than four
pages. Do not claim exact row quarters for that mode.

`query_page_nodes` uses query-only `move_pages` and returns actual physical page
locations. It is intended for setup diagnostics, not the inference path. The
fixture checks all pages in each of six 256 KiB format copies and independently
checks `/proc/self/numa_maps`: each copy has 16 pages on each of nodes 0–3.

Validation command:

```bash
taskset -c 0 /home/user/InfoSystemic/AI-Server/tools/deepseek-v41-cpu-reference-0910/venv/bin/python goal_numa_0910_check.py
```

The result is `results/goal_numa_0910/fixture-check.json`. No model, performance
trial, process-wide affinity change, or process/thread memory-policy change is
part of this helper or its fixture.
