#!/bin/bash
D=~/InfoSystemic/AI-Server/serving/fleet-0911
P=~/InfoSystemic/AI-Server/serving/fleet-0903/results/glm-flash-q8-r8-ordered-k-0908/private-cpu/libggml-cpu.so.0.22.0
echo "=== stopping server for a quiet-box kernel measurement ==="
for p in $(pgrep -f '^/home/user/.*/bin/llama-server'); do kill $p 2>/dev/null; done
for i in $(seq 180); do pgrep -f '^/home/user/.*/bin/llama-server' >/dev/null || break; sleep 1; done
sleep 5
/dev/shm/bwprobe/quiet.sh >/dev/null 2>&1
echo "=== REAL GEMV kernel (ggml_gemv_q4_K_x16_q8_K), thread scaling, QUIET box ==="
for t in 1 4 8 15; do timeout 240 /dev/shm/bwprobe/gemvbw $P ggml_gemv_q4_K_x16_q8_K $t 4 2>&1 | tail -1; done
echo "=== pure-read reference, same thread count ==="
/dev/shm/bwprobe/membw 15 0 4 4 64 2>/dev/null | tail -1
echo "=== POST DONE ==="
