# MTP shared-port window stopped and restored

The controller restored the exact validated production libraries, model arguments and inference environment as PID 2506497. A separate snapshot confirmed health and idle slots. The window lasted 12.20 minutes including both model loads and shutdown.

All three native outputs in disabled mode matched their references. Before changing to enabled mode, the idle guard detected another request on :18131 and aborted. A Python client (PID 2485459) was observed connected while the test server shut down; the client disconnected independently. Its request overlapped the third native run, which slowed to 5.41 tok/s. The candidate optimization was never enabled, and this window supplies no valid A/B gain.

The earlier production control also has 19 extra prompt tokens and four extra generated tokens in global metrics compared with its own Codex usage. Its response text is correct, but its 13.69 tok/s rate is excluded. The request-accounting audit confirms the previous pooling and bounded-rollback primary measurements match their own Codex token counts.

The retry uses private port 18141 with normal production briefly stopped, retaining the same immutable candidate and all exact gates. It includes explicit request-owned counter reconciliation and restores production on :18131. See `mtp_kv_window_r2.py` and `mtp-kv-only-r2-window-manifest.json`.

Evidence: `mtp-kv-only-window-report.json`, `mtp-kv-only-window-controller.log`, `mtp-kv-only-restoration-audit.json`, `codex-request-accounting-audit.json`.
