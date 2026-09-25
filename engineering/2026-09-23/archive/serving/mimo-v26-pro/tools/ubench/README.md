# Kernel micro-measurement kit (2026-09-23)

- `peak.cpp` -- vpdpbusd zmm issue rate: register operands vs `{1to16}` memory broadcast vs a mix (per thread, OpenMP)
- `fpeak.cpp` -- same for vfmadd231ps
- `freq.cpp` -- actual core clock under the AVX-512 heavy license (dependent vpaddd chain + dpbusd)
- `spin.c` -- `pause` spinner: pin it (nice 0) to the benchmark core's hyperthread sibling to get a clean core while
  other sessions' nice-19 jobs own the node, e.g. `taskset -c 113 ./spin 600 &` then bench on CPU 49
  (stop it with `pkill -x spin` -- NOT `pkill -f 'spin ...'`, which matches the invoking shell and kills it)
- `loopmca.py <objdump.s> [min_dpbusd]` -- llvm-mca-16 (-mcpu=cascadelake) estimate for every backward-jump loop with
  >= N vpdpbusd in a disassembly: `objdump -d --no-show-raw-insn -C lib.so | awk '/^<addr> /,/^$/' > f.s`
- `x16-kernel-proto.cpp` -- standalone copies of the shipped x16 GEMM kernel and the variants tried (A: bias-init,
  B: 40-byte activation blocks, R2: paired row groups = what shipped in 0922h, R3: three groups), interleaved timing
  and max|diff| vs the shipped kernel. `g++ -O3 -march=native -fopenmp -fPIC x16-kernel-proto.cpp -o proto`;
  `NR2=5 taskset -c 49 ./proto q8 1024 6144 510 10 64 1 5` / `./proto mx ...`
  (its mx_R3 has no odd-group tail: wrong output when a chunk leaves one group over -- a bench artifact)

Measured on this box (Xeon Gold 6242, single core, sibling idle): register-form vpdpbusd ~1.85/cycle, `{1to16}` form
~1.0-1.35/cycle, AVX-512 heavy clock 3.06 GHz single-core, ~2.7 GHz all-core.
- `fpeak2.cpp` -- FMA vs VNNI with 24 register accumulators, clock measured under FMA load (vaddps chain): both
  ~1.6-1.68 per cycle at ~3.1 GHz single-core -- use this as the practical peak, not 2/cycle
- `fe.cpp` -- front-end check: a 3.8 KB unrolled broadcast+2xdpbusd loop runs as fast as a 114 B one (1.9/cycle)
- `fma-ukernel-shapes.cpp` -- the simd-gemm FMA microkernel at 12x2 / 8x3 / 6x4 / 4x6 / 14x2 on L1-resident data
- `fa-tile-gemm-kblock.cpp` -- the flash-attention tile GEMM (QK^T 128x192x64, PV 128x64x128) shipped vs K-blocked,
  with a bit-identity check
