# GLM-5.3-Flash validated baseline

Historical UD-IQ2_XXS runtime validation only. As of September 7, this artifact
has not been qualified for the user's near-lossless-quality requirement and
is not the selected target for the new serving plan. See the [current quality
policy](GLM-FLASH-QUALITY-POLICY-20260907.md). The checks below establish
correctness against the quantized reference, not equivalence to the released
model's task quality.

This runtime is correct on the recorded checks but **has not reached 20 tok/s**.
The standard 1024-token samples measured 15.621512 tok/s for prose and 16.689717
for code, with production inference idle. Completed direct answers measured 15.157353 (782 tokens) and
17.918006 (674 tokens). Host load, context, and output affect performance.

Start it on the SR950 when no other Flash test model is running:

```bash
python3 /home/user/InfoSystemic/AI-Server/serving/fleet-0903/launch-glm-flash-validated.py
```

The launcher serves `glm-flash-goal` at `http://127.0.0.1:18131`, with a 4096-token
context, four NUMA sockets, 15 workers per socket, and two MTP draft tokens.
It uses `engines/llama.cpp-glm5n-goal-0904/validated-chunk16-bin` and the persistent
Q8 draft file in `/models/gguf/GLM-5.3-Flash/MTP/`. It does not modify the
production server on port 18091. Stop the launched test server with Ctrl-C.

`glm-flash-validated.json` records the exact command, runtime flags, measured
samples, draft checksum, and binary checksums. The launcher clears inherited
GGML, graph-profile, MTP-control, and OpenMP overrides before applying those flags.

Evidence is under `results/glm5n-goal-x16-chunk16-1024-t15/`: all seven complete
messages match the reference, the generated code retains its eight-case proof,
and repeat, extend, trim-one, and trim-three cache comparisons pass. The snapshot
matches the tested binaries byte-for-byte. The generated launcher itself has
been syntax-checked; it has not been launched as a separate additional test.
