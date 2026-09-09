# Proposed temporary Qwen split-policy trial

The private candidate changes the four-way expert partition from rotating
128/128/128/256 slices to equal 160/160/160/160 slices. The complete target
and MTP expert-path tests passed 64 timing phases and checked 4,915,200 output
values. Target IQ2 component improvements were 26-31%; actual model gains
remain unmeasured. The model goal is still at least 304 GB/s of adjusted
decode traffic for each of the three models.

The candidate is private libllama SHA256
`d213ba477ef5d3f2a68568b30a31595ef9578b6dce4b3b81e95da6f83a3dac8b`.
The exact commands and environments are in `plan.json`. Only the private
library resolution and `GGML_Q4E_EXPERT_EVEN_SPLIT=1` change for the candidate;
its endpoint is localhost:18155. All arithmetic backend libraries stay pinned.

## Concrete procedure

`qwen_split_trial.py` performs the following sequence after execution is
authorized:

1. Verify both original process identities, the pinned files, candidate hash,
   launch environment, working directory, affinity, and free trial port.
2. Measure the existing Qwen MTP2 service on port 18095 after 60 seconds idle:
   arithmetic/geography checks and prose/code requests with a 512-token limit.
3. After another 60-second idle gate, send SIGTERM to the exact existing Qwen
   process through a pidfd. Wait for it to exit; never use SIGKILL on it.
4. Wait for Full to be idle, then launch the private candidate on port 18155.
   Verify its mapped library and runtime settings, then collect the same
   guarded checks and MTP2 measurements.
5. Stop the owned candidate, wait for idle, and restore Qwen's original
   command, full in-memory environment, working directory, CPUs 0-127 main
   affinity, port 18095, and aliases `qwen-goal,qwen3.8-flash-next,flash-next`.
   Restoration produces a new Qwen PID. Its original pinned libraries remain
   unchanged on disk.
6. Verify the restored service and repeat the baseline measurement. Leave the
   restored original service running regardless of the candidate's speed.

The before/candidate/after results allow the candidate's speed and memory
traffic to be compared with nearby original-service measurements. The existing
fresh baseline is also retained. No permanent candidate adoption is included.

## Availability and recovery

Qwen on port 18095 is unavailable from its termination until its original
replacement becomes healthy. This includes the candidate load, measurement,
original-model reload, and idle gates. Each load has a 15-minute timeout;
incoming inference can extend the interruption because loading yields to it.
There is no guaranteed short outage duration.

Full's service is never signalled or reconfigured. The trial waits for idle,
and incoming Full work cancels the owned test. Request monitors close only
their benchmark socket. Cancellation or candidate failure enters restoration.
Transient monitoring failures during restoration wait and retry; competing
inference cancels only the owned restoration load before another idle attempt.
After successful restoration, the controller cannot terminate the handed-off
original service. A changed protected identity, port conflict, altered input,
or failure of the original model itself needs investigation; the controller
will not stop an unrelated process or silently load a second Flash model.

The default controller invocation is read-only. Execution is not queued.
The preserved-service constraint in `MODEL-BANDWIDTH-TARGETS-20260905.md`
currently includes the independently started Qwen PID 2308651/port 18095.
Temporarily interrupting that service requires an explicit exception before
using `--execute`.

## Validation completed

- `controller-preflight.json`: the default read-only controller invocation
  passed against the actual original services and candidate files.
- `lifecycle-check.json`: all 17 checks passed using disposable local HTTP
  processes, with zero requests or signals to real model services. Coverage
  includes failures before and after original termination, candidate failures,
  cancellation, cleanup failure, post-restoration failure, protected/reused/
  mismatched process identities, restoration handoff, and three injected
  restoration monitoring/peer-work cases using the production restoration loop.
- The child launcher preserved CPUs 0-127 while the test controller ran on
  CPU 63. Every disposable process was reaped by the test cleanup.
- Existing request guard checks cover nine normal/cancellation paths, including
  queues and monitoring failure; their evidence is `measurement-guard-check.json`.

The controller and check source hashes are recorded in the JSON evidence.
No actual split-candidate model trial has run.
