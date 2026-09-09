# Qwen3.8-Flash-Next validated runtime

The SR950 run measured **20.277 tok/s on 1,024 generated prose tokens** and
**23.607 tok/s on a completed 325-token code answer**. It used 15 physical
workers per socket, four NUMA devices, a 4,096-token context, and two-token
MTP speculation. These are measured rates under the recorded host workload.

Run on the SR950 host from the AI-Server directory:

```bash
python3 serving/fleet-0903/launch-qwen-flash-20tps.py
```

The launcher serves `qwen-goal` at `http://127.0.0.1:18125/v1`. The benchmark
server has exited; this launcher starts a new server. Validation used one Flash
test model loaded at a time alongside the idle production server on port 18091.

The executable and libraries are pinned in
`engines/llama.cpp-q4e-goal-0904/validated-iq-batch3-bin`. Their SHA256 values,
runtime settings, full command and measured timings are in
[qwen-flash-20tps.json](qwen-flash-20tps.json).

All 480 focused numerical cases matched the previous output hashes. All ten
full response messages matched the reference, the generated merge function
retained its eight-case validation, and all four cached/fresh continuation
checks passed. Full responses and host-load measurements are in
[the benchmark result](results/q4e-goal-iq-batch3-threads-1024/result.json).
