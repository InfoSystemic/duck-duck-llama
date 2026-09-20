# First KV rollback window: diagnostic stop

The controller restored the original production runtime as PID 353630. The temporary window lasted 10.54 minutes. No candidate was promoted.

All six native outputs matched the saved production references. Retained-loop aggregate throughput was 15.299 tok/s and bounded-loop throughput was 15.699 tok/s. Prediction and draft-work counters matched across these native arms: False. This single ordered comparison is preliminary; the planned repeated Codex comparison did not run.

The controller stopped at the explicit mode-log gate. The candidate emitted its probe at INFO level, which the normal server verbosity filtered. The read-only control mapping check had passed. This was a diagnostic failure, not an observed output mismatch. The automatic finally/ExecStopPost restoration completed, and all original library hashes, command arguments and inference environment values matched.

The correction changes INFO to WARN for the opt-in mode probe. ELF comparison confirms only one executable byte differs, plus the 20-byte build ID; all other executable bytes are identical. The corrected binary is gated again and staged under separate r2 controller, manifest, report and log paths.

Evidence: [controller report](kv-range-window-report.json), [controller log](kv-range-window-controller.log), [restored runtime](kv-range-window-restored.json), [binary diagnostic comparison](/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx/glm-kv-range-0919/diagnostic-only-diff.json).
