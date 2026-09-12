# Making four CPU sockets useful

Large-model inference on a multi-socket server is a placement problem as well as a compute problem. More worker threads can introduce remote memory traffic and synchronization without increasing useful work.

## Engineering

The project integrates CPU-NUMA devices into llama.cpp and works through the model-specific consequences: tensor partitioning, physical worker selection, repacked buffers, reductions, synchronization, and actual memory residency. The [original backend patch](../../patches/llama.cpp-b249-cpu-numa.patch) and [later complete source bundles](../../engineering/2026-09-12/source-bundles.json) preserve that progression.

The later experiments connect changes in scheduling to stable decode windows and hardware memory-controller counters. They measure raw decode and MTP separately, retain adjacent idle traffic, and compare outputs and speculative counts.

## Controlled result: the Flash Q8 barrier

An eight-run control/candidate/candidate/control experiment changed the dissemination barrier while holding the target, draft, worker settings, and arithmetic options constant. Each run generated 512 tokens for prose and code.

| Raw decode | Prose tok/s | Code tok/s | Prose adjusted GB/s | Code adjusted GB/s |
| --- | ---: | ---: | ---: | ---: |
| Control, two runs | 11.48–11.55 | 11.48–11.53 | 236.56–238.39 | 236.90–238.76 |
| Barrier, two runs | 11.61–11.68 | 11.69–11.69 | 239.34–239.56 | 240.36–241.37 |

The mean raw-decode improvement was 1.12% on prose and 1.62% on code. All 16 decode windows passed coverage checks for the 48 IMC counters, and matching complete output streams agreed. MTP2 gains were small and mixed, so the report retained its control setting. [Experiment and independent audit](../../engineering/2026-09-08/archive/serving/fleet-0903/FLASH-BANDWIDTH250-20260910.md).

These are generated-token measurements, including reasoning that reached the token cap. The adjusted counter values estimate model traffic on the shared host. They do not establish answer quality or a universal bandwidth ceiling.

## Capacity is also local

Later Full loads demonstrate a different NUMA failure: memory-policy OOM on node 3 despite a four-node machine. The latest inspected attempt failed even with NUMA repacking disabled. [Kernel evidence](../../engineering/2026-09-12/archive/serving/fleet-0912/results/glmfull-no-numa-repack-873660/kernel-oom.txt).

The resulting work includes a [page-placement utility with read-only file mappings](../../engineering/2026-09-12/archive/serving/fleet-0912/numa/README.md), complete inventories of staged model data, and a [quantization-aligned unequal-split correction](../../engineering/2026-09-12/archive/serving/fleet-0912/glmfull-split/README.md). The migration fixture moved 3 MiB of a 4 MiB file into four equal node populations with an unchanged full-file SHA-256. No successful bulk model migration is claimed here.

The inventories distinguish two footprints: the six staged Flash shards had approximately one quarter of their 185.99 GiB on node 3, while **32.717 of 36.734 GiB** of the inspected native DeepSeek tensor cache was on that node. The latter covered all pages of 5,530 manifested tensors at least 1 MiB; every full-file hash matched before and after, with zero bytes moved. [Inventory summary and original report digest](../../engineering/2026-09-12/archive/serving/fleet-0912/numa/deepseek-inventory-summary-20260912-0928.json). This exposes local capacity pressure without proving that migration will resolve the Full load failure.

The practical lesson is to measure placement, allocation peaks, and throughput together. Total free RAM, process affinity, and final model size are each incomplete descriptions of the problem.
