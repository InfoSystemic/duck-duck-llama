# KV rollback work completed

The corrected R2 A/B completed, restored production exactly, and passed the final restored-production control. The separate deployment then completed and passed native plus actual Codex cold/cached output gates. Production PID 1561764 is healthy and independently audited. The controller process has exited successfully; no test window remains running.

- Measured primary result: 15.2387 to 15.6078 tok/s (+2.42%), eight balanced cached Codex runs.
- Final production checks: 15.5796 cold / 15.7127 cached tok/s, exact reference text.
- Runtime libllama: `e77a573df69456745857677e71e4e9cc876b8294384fa7798629f3ebe7c3306a`.
- Retained wider-pool CPU: `3f957a341321b940d93be53c250cdd068825093faa2d9efda142ebe56427b1b3`.
- Profile arms absent; no benchmark control/probe env in production.
- Full record: [KV-RANGE-RESULT.md](KV-RANGE-RESULT.md).
- Machine evidence: `kv-range-promotion-report.json`, `kv-range-final-audit.json`.

The broader GLM tuning goal remains active. 18+ tok/s and the pre-existing cached/fresh consistency failure remain unresolved. The next step is a bounded investigation of remaining current-runtime costs, not a repetition of completed kernel sweeps.
