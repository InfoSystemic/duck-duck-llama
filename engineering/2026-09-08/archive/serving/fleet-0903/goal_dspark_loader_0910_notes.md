This is optional checkpoint attachment code, not a speculative generation loop.
No real DSpark parameters have been loaded, and no DSpark forward pass has run.

The pinned full-width metadata plan validates 3 MTP stages (layer IDs 40–42),
block size 5, and target backbone layers 37–39. The backbone remains 40 layers.
Text-only common parameters require 713,427,336 native cache bytes and
880,194,440 runtime parameter bytes. The 384 routed experts retain native packed
FP4/E8M0 storage, totaling 7,219,445,760 bytes if every expert is eventually cached.
The 3 unused VL routing-bias tensors are excluded, exactly as in text-only backbone
loading. All comparisons use the pinned local catalog and source revision.

Suggested explicit setup, only after the root decides to test actual DSpark:

```python
from goal_dspark_store_0910 import initialize_mtp_cache, MTPResidentStore
from goal_dspark_loader_0910 import prepare_dspark, install_dspark

# mtp_cache must be separate from runtime.store.root. Initialize only once.
initialize_mtp_cache(mtp_cache, runtime.store.root)
mtp_output.mkdir(parents=True, exist_ok=True)
mtp_store = MTPResidentStore(mtp_cache, mtp_output, allow_download=False)
plan = prepare_dspark(runtime.module, runtime.config, mtp_store.catalog)

# This intentionally rejects absent common weights while downloads are disabled.
with runtime.lock:
    result = install_dspark(runtime.model, runtime.module, plan, mtp_store)
    runtime.config = plan.args
```

Enabling `allow_download=True` is a distinct caller decision and can fetch the
94 required common tensors. Later draft routing loads only selected real experts.
The MTP cache defaults to 8 GiB and retains the existing 100-GiB physical-memory
reserve in CoalescedStore. Its MTP-only journal can evict expert groups while
keeping common tensors pinned. Expert release uses ResidentStore's normal meta
replacement and scale-alias restoration. It stores real `mtp.*` names, uses its
own directory/marker/journal, and never modifies backbone records or eviction code.

`install_dspark` loads each stage while its embedding/head aliases remain None,
then assigns the existing backbone objects by identity. Those two large shared
weights are never enumerated for an MTP common download. Routed experts stay on
meta until activation. Common tensor arithmetic/conversions reuse the existing
`load_parameters`, including FP8 wo_a expansion and publisher-required BF16→F32
promotions. Derived window/frequency buffers are materialized from equations.

Attachment must happen before a fresh backbone prefill that collects the target
hidden states. `forward_spec(..., start_pos=0)` must then seed MTP window caches.
The supplied official `generate.py` does not call `forward_spec`, so there is no
publisher-provided acceptance loop to copy from that file. A future controller
must implement and verify target acceptance, first-token alignment, rejection
rollback, and MTP cache synchronization. DSpark draft attention intentionally
sees its whole draft block; only the target verifier determines accepted tokens.
For a 5-token draft, respect `start_pos + main_hidden_seq_len + 5 <= max_seq_len`
when indexing the pinned-size RoPE buffer.

The loader's returned `draft_execution_enabled=False` is intentional: attaching
weights does not enable draft generation in the endpoint. Native per-layer
optimizations installed only on the 40 backbone layers also need a separate
review before applying them to these 3 new stages. The separate store must be
closed by the caller; it is not automatically covered by runtime.store.close().
