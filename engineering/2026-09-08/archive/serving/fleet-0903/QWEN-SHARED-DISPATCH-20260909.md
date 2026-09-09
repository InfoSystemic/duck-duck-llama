# Proposed shared NUMA dispatch experiment

Status: source investigation only. No source transform, private build, graph
validation, or model measurement exists for this candidate. The Q6/Q8 goal
remains above 30 generated tok/s and above 190 decimal GB/s on fresh prose
and code in the same configuration, with repeated final measurements.

## Observed implementation

In ggml-backend-meta.cpp, each Meta context creates one driver thread per
NUMA device. dispatch_cpu_numa publishes a complete graph group with an
epoch and waits for all drivers to finish. Each driver calls the synchronous
ggml_backend_graph_compute wrapper for its own simple backend.

In ggml-cpu.cpp, each NUMA CPU backend owns a separate persistent dispatcher,
threadpool state, scratch buffer, and condition variables. Its asynchronous
submission callback hands the graph to that dispatcher. The dispatcher calls
ggml_graph_compute, which enters an OpenMP team in ggml-cpu.c. The target
and MTP draft have separate backend contexts.

Reusing only the ggml_threadpool_t data object would leave separate OpenMP
calling threads. A meaningful sharing experiment must also reuse the threads
that enter the parallel region. Actual team reuse and performance still need
runtime evidence; source inspection does not prove either.

## Candidate design

Use a new, default-off experimental flag in an isolated runtime. Preserve all
existing arithmetic, weight formats, quantization, graph fusion, affinity,
and public behavior when the flag is absent.

1. With the flag enabled, execute a NUMA CPU backend's graph synchronously
   in its caller instead of creating its extra asynchronous dispatcher.
   Keep separate per-backend threadpool state and scratch buffers. Meta
   already supplies concurrent per-device drivers. Ordinary single-backend
   calls must retain the documented completion semantics.
2. Reuse a Meta driver group between contexts with the same ordered NUMA
   device list. A shared group owns only dispatch threads and job state;
   each submitted job supplies the current backend and graph pointers.
3. Hold a submission mutex for the entire graph-group dispatch and completion.
   Every rank must observe the same group order. Interleaving graph groups
   on shared rank threads could deadlock context-specific fused collectives.
4. Retain a strong group reference in each using Meta context, with weak
   registry entries. Destruction must wait for the context's work to finish
   before freeing its private backend state. The last group owner joins its
   workers. A group must never access a destroyed context or borrowed graph.
5. Use the existing flag-off implementation unchanged as the control. Keep
   huge pages and block quantization disabled for this scheduling comparison.

This may remove duplicate OpenMP teams and one dispatch handoff. It may also
serialize work that previously overlapped. It is an experiment, not an assumed
gain. Different ordered device groups must not accidentally share rank state.

## Required evidence before adoption

- Reproduce the parent CPU and ggml-base libraries from their unchanged
  objects before replacing the relevant C++ objects in private libraries.
- Confirm the complete candidate source diff changes scheduling and lifetime
  handling only, with no numeric kernel or weight changes.
- Compare complete outputs against the parent for dense and routed-expert
  graphs, ordinary and fused reductions, multiple token batch sizes, and
  the actual four-socket expert layout.
- Exercise two contexts alternately and concurrently, proving that group
  submissions cannot interleave collective ranks or deadlock.
- Destroy one context, run the other, create a replacement, and run again.
  Verify group and worker lifetime, actual thread reuse, affinity, and cleanup.
- Load Q6/Q8 only after the graph/lifecycle checks pass. Run fresh prose and
  code with the existing resource gates and IMC attribution, compare complete
  outputs and draft counts, then repeat any apparent model gain.

The recorded source hashes are in
results/qwen-30tps-tools-0909/shared-dispatch-source-review.json. Published
or completed source files must stay unchanged; use new private versions for
the experiment.
