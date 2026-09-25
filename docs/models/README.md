# Model guides

Each guide separates historical model measurements, the latest prepared implementation, and unresolved validation. Model names, checkpoint revisions, and quantizations follow the original experiment records.

| Model | Implementation path | Guide |
| --- | --- | --- |
| GLM-5.3 Full | llama.cpp, GLM DSA tensor splitting, mixed runtime precision, detached MTP | [Full](glm-full.md) |
| GLM-5.3-Flash | llama.cpp, CPU-NUMA, Q8/Q4 targets, Q8 MTP2 | [Flash](glm-flash.md) |
| Qwen3.8-Flash-Next | llama.cpp, Q6 target, MTP and ngram experiments | [Qwen](qwen-flash-next.md) |
| DeepSeek-V4.1-Flash | Native FP8/FP4 CPU runtime; separate llama.cpp port | [DeepSeek](deepseek-v41.md) |
| MiMo-V2.6-Pro-RL | llama.cpp, four-way CPU tensor parallel, MXFP4 experts, Xiaomi's DFlash block drafter, F32 vision projector | [MiMo](mimo-v26-pro.md) |

For comparisons, start with [recorded results](../results.md). For executable prerequisites and source layers, use [reproduction](../reproducing.md) and [profiles](../../profiles/README.md).
