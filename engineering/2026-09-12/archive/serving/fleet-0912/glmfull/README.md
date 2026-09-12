# GLM-5.3 Full v5b profile

This foreground launcher preserves the actual GLM Full process started on September 12, PID 4097083. It uses the `sr950-glm/build-dev2` runtime, whose GLM DSA tensor split implementation is required for Full. The newer GLM Flash runtime rejects this architecture in tensor split mode.

```sh
python3 launch.py --port 18131 --dry-run
python3 launch.py --port 18131
```

The launcher requires a vacant port and does not stop another process. It pins the runtime library directory and starts with the recorded inference environment and CPU affinity 0-127. All eleven model shards, the MTP draft and the chat template must exist. Defaults match the running measurement: 4096 total context, one stream, MTP depth 2, 15 worker threads per NUMA node and q8_0 KV. `--ctx-size` and `--parallel` permit larger configurations after memory and quality validation.

The v5b environment includes the previously selected attention/shared-expert q5_K and output q6_K load-time requantization and F16 routers. This is the existing fast recipe, not a claim of native-weight equivalence. The current run is undergoing full-model semantic and throughput validation; historical August speed figures are not fresh evidence.

The preferred hybrid MTP draft is currently in `/dev/shm`, so it must be staged again after reboot. Missing drafts fail preflight. Unequal tensor split ratios remain unsupported by this profile: the current source fails to round several GLM DSA splits to quantization block boundaries. The balanced split is retained until that issue has a full-model validation.
