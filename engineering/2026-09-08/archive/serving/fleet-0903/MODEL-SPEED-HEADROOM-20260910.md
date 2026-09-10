# Model speed and remaining headroom, September 10, 2026 UTC

The user now expects Qwen3.8-Flash-Next to exceed 40 generated tok/s. Keep the selected Q6 precision as the engineering target. The existing all-model 250 GB/s request remains incomplete. These are single-conversation decode rates, including reasoning; prefill, aggregate concurrency, and short arithmetic checks are excluded.

| Model/configuration | Highest saved prose tok/s | Highest saved code tok/s | Qualification |
| --- | ---: | ---: | --- |
| Qwen Flash-Next Q6, experimental MTP4 | 24.87 | 32.15 | One best candidate run; repeated parent means 24.52/31.80, candidate means 24.37/31.84; no consistent scheduling gain or promotion |
| Flash Q4, selected Q8 MTP2 | 15.94 | 15.93 | Two matching-output repeats under recorded background CPU load; means 15.85/15.81 |
| Flash Q8, historical MTP2 | 15.38 | 16.54 | Maxima across control/candidate arms, not one superior configuration |
| Full mixed Q4, latest MTP2 evidence | 7.89 | 9.64 | Latest fully recorded fresh prose/code pair; earlier single prose run 8.50 |

Older Flash IQ2 long/direct outputs reached 17.92 tok/s. Full reached 15.70 tok/s on a passing specialized copy/edit replay with deeper speculation. Those are different quantizations or workloads and do not establish the selected configurations' general speed.

Current Flash Q4 uses 178.5-184.0 GB/s after adjacent-idle subtraction, around 47-48% of the stated capacity. Qwen experimental MTP4 runs use roughly 139-153 GB/s, 37-40%; Full latest MTP2 uses 197-199 GB/s, about 52%. Historical Flash Q8 raw decode reached 241.37 GB/s (63.5%) while emitting fewer tokens than MTP2. All four sockets contribute similar traffic in the examined counter windows. A missing memory socket does not explain these gaps.

Qwen 40 tok/s means 25 ms per generated token. Compared with repeated parent means, prose must fall from about 40.8 ms to 25 ms (39% less time), and code from 31.4 ms to 25 ms (20% less time). At the measured MTP4 traffic ratio, 40 tok/s corresponds to approximately 231 GB/s for prose and 189 GB/s for code. That fits inside the stated DRAM budget. It does not prove that every execution stage can meet the 25 ms budget. An ordinary raw Q6 weight-only estimate at 40 tok/s is about 279 GB/s, before state and activation traffic; speculation changes bytes per generated token.

The following arithmetic scales measured rates to higher sustained bandwidth at unchanged approximate physical traffic per generated token. It also assumes the compute, dependency, scheduling, and draft costs fall enough to permit that bandwidth. These are conditional planning figures, not achieved speeds, promises, or proven maxima. Each cell is prose/code.

| Measured configuration used as basis | At 250 GB/s | At 323 GB/s (85%) | At 353.4 GB/s (93%) |
| --- | ---: | ---: | ---: |
| GLM Flash Q4 MTP2 | 22.0/21.7 | 28.4/28.1 | 31.1/30.7 |
| Qwen Flash-Next Q6 MTP4 | 43.4/52.9 | 56.0/68.4 | 61.3/74.8 |
| GLM Full mixed Q4 MTP2 | 10.0/12.1 | 13.0/15.6 | 14.2/17.1 |

The next Qwen engineering priority is target matrix execution and how work passes between the four sockets, followed by draft output projection and verification scheduling. Existing traces assign about 58% of summed target socket-stage time to matrix operations and 11-12% to cross-socket reductions. The output projection is about 49% of draft graph time. These are instrumented stage proportions with following barriers included; concurrent socket times are summed. They are not removable fractions of request time. A fresh wall-time critical-path profile of the corrected Q6 stack is required before budgeting a large new kernel or scheduling change.

The corrected gather already produced repeated gains of about 3.9% prose and 2.5% code. Broader barrier/spin changes, huge pages, expanded Q6 layouts, and the latest ten-route tile change did not establish a consistent model gain. Repeating those settings alone is not evidence of a route to 40. The latest scheduling change was -0.60% on prose and +0.10% on code. Qwen Q4 removes only 8.69% of estimated active target bytes (9.51% equal-bandwidth weight-only speed headroom), which does not close the observed prose gap by itself.

Flash Q4 first needs an uncontended repeated baseline and profiling of its mixed Q4/Q5/Q6 expert kernels alongside the retained dense Q8 kernels. About 8.97 of its estimated 14.25 GB of raw active target weights remain dense/shared/other. Full needs fresh model profiling and validation of any transplanted optimizations; the available evidence does not identify its largest current bottleneck.

Large streaming bandwidth is a hardware budget. Layer dependencies, unpacking and arithmetic, small operations, state traffic, and synchronization can stop decode from issuing enough useful memory requests to spend that budget. This interpretation follows [Intel's CPU roofline guidance](https://www.intel.com/content/www/us/en/docs/advisor/get-started-guide/2023-0/identify-bottlenecks-using-cpu-roofline.html). Better speculation or less redundant traffic can improve tok/s while reducing GB/s.

- [Machine-readable budgets and source hashes](results/model-speed-headroom-0910.json)
- [Completed Flash Q4 switch](FLASH-Q4-SWITCH-20260910.md)
- [Qwen scheduling comparison](QWEN-DECODE-SCHEDULING-20260909.md)
- [Qwen operation attribution](results/qwen-shared-dispatch-ops-analysis-0909.json)
- [Earlier all-model bandwidth assessment](MODEL-250GBPS-20260909.md)
