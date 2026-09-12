# GLM-5.3-Flash

**Status:** repeated Q8 raw/MTP experiments and Q4 serving work are recorded. Durable launcher/library recipes are prepared; the newest unary recipe still needs a fresh matched model comparison.

## Runtime and evidence

The four-socket Flash work covers tensor distribution, quantized kernels, shared/expert execution, barriers, Q8 MTP2, and later Q4 staging. The controlled Q8 barrier experiment reached 239.34–241.37 adjusted GB/s in raw decode. Its MTP2 effect was small and mixed. [Controlled comparison](../../engineering/2026-09-08/archive/serving/fleet-0903/FLASH-BANDWIDTH250-20260910.md).

The later Q4 target with a Q8 MTP2 draft measured 15.76–15.94 prose and 15.68–15.93 code tok/s under the report's recorded background load. Those observations are separate from the Q8 barrier comparison. [Q4 switch and qualification](../../engineering/2026-09-08/archive/serving/fleet-0903/FLASH-Q4-SWITCH-20260910.md).

## Latest prepared recipe

The September 11 unary/scale comparison records 15.39 to 15.71 mean decode tok/s with three identical greedy outputs. The [September 12 recipe](../../engineering/2026-09-12/archive/serving/fleet-0912/glmflash/README.md) rebuilds the durable candidate library and checks the launch dependencies. Those build checks are not a new speed measurement.

## Reproduction entry points

- [Current host recipe, source overlays, and rebuild instructions](../../engineering/2026-09-12/archive/serving/fleet-0912/glmflash/README.md).
- [Fresh A/B/A controller](../../engineering/2026-09-12/archive/serving/fleet-0912/AB-PROFILES.md).
- [Parameterized Q8 and Q4 profiles](../../profiles/20260908/README.md).
- [Source bundle inventory](../../engineering/2026-09-12/source-bundles.json): distinguish the goal source from earlier bring-up trees.

The Q4 payload is staged on a volatile RAM-backed volume, approximately 186 GiB. Reboot staging, per-node placement, and coexistence with Full affect the operational recipe. Fresh controls, completed-answer checks, and larger-context measurements remain open.
