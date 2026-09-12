# GLM-5.3 Full v5b profile

This foreground launcher preserves the actual GLM Full process started on September 12, PID 4097083. It uses the `sr950-glm/build-dev2` runtime, whose GLM DSA tensor split implementation is required for Full. The newer GLM Flash runtime rejects this architecture in tensor split mode.

```sh
python3 launch.py --port 18131 --dry-run
python3 launch.py --port 18131
```

The launcher requires a vacant port and does not stop another process. It pins the runtime library directory and starts with the recorded inference environment and CPU affinity 0-127. All eleven model shards, the MTP draft and the chat template must exist. Defaults match the attempted measurement: 4096 total context, one stream, MTP depth 2, 15 worker threads per NUMA node and q8_0 KV. `--ctx-size` and `--parallel` permit larger configurations after memory and quality validation.

The v5b environment includes the previously selected attention/shared-expert q5_K and output q6_K load-time requantization and F16 routers. This is the existing fast recipe, not a claim of native-weight equivalence. Full-model semantic and throughput validation remains pending; historical speed figures are not fresh evidence.

Three recorded load attempts ended in kernel-confirmed memory-policy OOM on node 3: [PID 4097083](../results/glmfull-v5b-4097083/kernel-oom.txt), [PID 348880](../results/glmfull-v5b-348880/kernel-oom.txt), and [PID 873660 with NUMA repacking disabled](../results/glmfull-no-numa-repack-873660/kernel-oom.txt). None produced a completed benchmark. Disabling repacking alone did not resolve the allocation failure under the recorded host footprint.

The preferred hybrid MTP draft is in `/dev/shm`, so it must be staged again after reboot. Missing drafts fail preflight. The default `tuned` arm retains the balanced split. `--arm no-numa-repack` reproduces the repacking override. The selected runtime does not correctly align every unequal GLM DSA split to quantization blocks.

An isolated [split correction and library build](../glmfull-split/README.md) now passes metadata checks and loader preflight. `--arm split-candidate` selects that library and verifies its recorded build inputs; only this arm accepts `--tensor-split`. The candidate has not completed a full-model load or inference comparison and is not selected by default. A dry-run command is an implementation check, not evidence that the model will fit in RAM.
