#!/usr/bin/env python3
"""Alternate resident private model-component graphs; never a model-bandwidth result."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import statistics
import subprocess
import threading
import time
import urllib.request

from dram_bandwidth import PerfDramRecorder, summarize_samples
from inference_contention_guard import activity


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--label", required=True)
parser.add_argument("--k", type=int, default=4096, help="K per Full socket or Qwen gate input width")
parser.add_argument("--rows", type=int, default=6144)
parser.add_argument("--matrices", type=int, default=8)
parser.add_argument("--tokens", type=int, choices=(1, 2, 3), default=3)
parser.add_argument("--seconds", type=float, default=4)
parser.add_argument("--rounds", type=int, default=2, help="Eight balanced phases per round")
parser.add_argument("--comparison", choices=("chunks", "counter_layout", "expert_split"), default="chunks")
parser.add_argument("--chunks", default="64,32", help="Baseline,candidate limits for the chunk comparison")
args = parser.parse_args()
assert re.fullmatch(r"[A-Za-z0-9_-]+", args.label)
assert 256 <= args.k <= 8192 and args.k % 256 == 0
assert 64 <= args.rows <= 16384 and args.rows % 64 == 0
assert 1 <= args.matrices <= 64 and 3 <= args.seconds <= 10 and 1 <= args.rounds <= 8

base = Path(__file__).resolve().parent
qwen_experts = args.comparison == "expert_split"
engine = base.parents[1] / "engines" / ("llama.cpp-q4e-goal-0904" if qwen_experts else "llama.cpp-sr950-glm")
libraries = engine / ("validated-iq-batch3-bin" if qwen_experts else "build-dev2/bin")
out = base / "results" / args.label
out.mkdir(exist_ok=False)
source = base / ("qwen-expert-graph-check.cpp" if qwen_experts else "cold-graph-check.cpp")
binary = out / ("qwen-expert-graph-check" if qwen_experts else "cold-graph-check")
allowed = {"4005448": 18091, "2308651": 18095}
chunks = tuple(args.chunks.split(","))
assert len(chunks) == 2 and len(set(chunks)) == 2 and all(c in ("16", "32", "64", "128", "256") for c in chunks)
if qwen_experts:
    assert args.k == 2560 and args.rows == 640 and 4 <= args.matrices <= 16 and args.matrices % 4 == 0
    assert set(chunks) == {"128", "32"}
modes = ("packed", "padded") if args.comparison == "counter_layout" else chunks
a_mode, b_mode = modes
order = [a_mode, b_mode, b_mode, a_mode, b_mode, a_mode, a_mode, b_mode] * args.rounds
weight_bytes = args.matrices * (args.rows // 16) * (4 * args.k // 256) * 2880
pool_bytes = weight_bytes
if qwen_experts:
    per_expert = args.matrices * 2 * (args.rows // 16) * (args.k // 256) * 2208
    weight_bytes = per_expert * (10 + 2 * (args.tokens - 1))
    pool_bytes = per_expert * 32
assert len(modes) * pool_bytes <= 2 * 1024**3, "Resident weight pools exceed 2 GiB"
result = dict(started=time.time(), pid=os.getpid(), config=vars(args), order=order,
              scope="Synthetic resident component graphs, not a model-decode measurement.",
              fixtures={}, phases=[], background_windows=[], completed=False)
owned = []
fixtures = {}
logs = []
recorder = None
guard = None


def save():
    temporary = out / "result.json.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(out / "result.json")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process_sample(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return int(fields[11]) + int(fields[12]), fields[19]


def inference_snapshot():
    current = {}
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            name = (p / "comm").read_text().strip()
            if name != "llama-server" and not name.startswith(("glm-mtp-head", "qwen-mtp-head")):
                continue
            ticks, start = process_sample(p.name)
            command = (p / "cmdline").read_bytes().split(b"\0")
            ports = [command[i + 1].decode() for i, x in enumerate(command[:-1]) if x == b"--port"]
            current[p.name] = (ticks, start, ports[-1] if ports else None)
        except (OSError, ValueError, IndexError):
            continue
    return current


def server_state(current=None):
    current = inference_snapshot() if current is None else current
    assert set(allowed) <= set(current), "An expected inference service is missing"
    state = {}
    for pid, port in allowed.items():
        assert current[pid][2] == str(port), "An expected inference port changed"

        def get(path):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/{path}", timeout=2) as response:
                return response.read().decode()

        slots = json.loads(get("slots"))
        queued = [float(line.split()[-1]) for line in get("metrics").splitlines()
                  if not line.startswith("#") and "requests_deferred" in line]
        assert len(queued) == 1, "Missing queue metric"
        state[pid] = dict(processing=any(s["is_processing"] for s in slots), queued=queued[0])
    foreign = sorted(set(current) - set(allowed))
    return dict(servers=state, foreign_pids=foreign,
                busy=bool(foreign) or any(s["processing"] or s["queued"] for s in state.values()))


def terminate_owned(sig=signal.SIGTERM):
    for process in list(owned):
        if process.poll() is None:
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass


class Guard:
    """Monitor CPU, queues, and identities; stop only owned subprocess groups."""
    def __init__(self):
        self.stop_event = threading.Event()
        self.failure = None
        self.thread = threading.Thread(target=self.monitor, daemon=True)

    def monitor(self):
        before, then = inference_snapshot(), time.monotonic()
        while not self.stop_event.wait(0.5):
            try:
                after, now = inference_snapshot(), time.monotonic()
                state = server_state(after)
                samples, churn = activity(before, after, now - then)
                assert not state["busy"] and not churn and not any(s["cpu_percent"] > 20 for s in samples), (
                    state, samples, churn)
                before, then = after, now
            except BaseException as error:
                self.failure = dict(time=time.time(), reason=repr(error),
                                    action="Terminate only private fixtures/build subprocesses.")
                (out / "contention.json").write_text(json.dumps(self.failure, indent=2) + "\n")
                terminate_owned()
                return

    def check(self):
        assert self.failure is None, f"Isolation guard stopped the test: {self.failure}"

    def pause(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.check()
            time.sleep(min(0.2, max(0, deadline - time.monotonic())))
        self.check()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=6)
        assert not self.thread.is_alive(), "Guard monitor did not finish"
        self.check()


def idle_gate():
    before, then = inference_snapshot(), time.monotonic()
    quiet_since = None
    last_report = 0
    while True:
        time.sleep(5)
        after, now = inference_snapshot(), time.monotonic()
        state = server_state(after)
        samples, churn = activity(before, after, now - then)
        busy = state["busy"] or churn or any(s["cpu_percent"] > 1 for s in samples)
        quiet_since = None if busy else now if quiet_since is None else quiet_since
        quiet = 0 if quiet_since is None else now - quiet_since
        status = dict(time=time.time(), quiet_seconds=quiet, required_quiet_seconds=60,
                      state=state, other_inference=samples, churn=churn)
        (out / "waiting-for-idle.json").write_text(json.dumps(status, indent=2) + "\n")
        if quiet >= 60:
            return status
        if now - last_report >= 30:
            print(json.dumps(dict(waiting_for_idle=True, quiet_seconds=quiet, busy=bool(busy))), flush=True)
            last_report = now
        before, then = after, now


def read_event(process, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        guard.check()
        if select.select([process.stdout], [], [], 0.2)[0]:
            line = process.stdout.readline()
            assert line and line.lstrip().startswith("{"), f"Unexpected fixture output: {line!r}"
            return json.loads(line)
        assert process.poll() is None, f"Fixture exited unexpectedly: {process.returncode}"
    raise TimeoutError("Fixture event timed out")


def run_build(command, cwd, log_name):
    guard.check()
    step = dict(command=command, cwd=str(cwd), log=log_name, started=time.time())
    result.setdefault("build_steps", []).append(step)
    save()
    with (out / log_name).open("w") as log:
        process = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        owned.append(process)
        deadline = time.monotonic() + 180
        while process.poll() is None:
            assert time.monotonic() < deadline, "Private build timed out"
            guard.pause(0.2)
        step.update(exit_code=process.returncode, finished=time.time())
        save()
        assert process.returncode == 0, f"Private build failed: {log_name}"


def placement(pid):
    nodes = {str(j): dict(pages=0, nonlocal_pages=0) for j in range(4)}
    for line in Path(f"/proc/{pid}/numa_maps").read_text().splitlines():
        fields = line.split()
        if len(fields) < 2 or fields[1] not in ("bind:0", "bind:1", "bind:2", "bind:3"):
            continue
        expected = fields[1][5:]
        for field in fields[2:]:
            match = re.fullmatch(r"N(\d+)=(\d+)", field)
            if match:
                node, count = match.groups()
                nodes[expected]["pages"] += int(count)
                if node != expected:
                    nodes[expected]["nonlocal_pages"] += int(count)
    page_size = os.sysconf("SC_PAGE_SIZE")
    assert all(n["pages"] * page_size >= pool_bytes // 4 and not n["nonlocal_pages"]
               for n in nodes.values()), nodes
    return dict(page_size=page_size, nodes=nodes)


def phase(mode, warmup=False):
    process = fixtures[mode]
    before = {m: process_sample(p.pid) for m, p in fixtures.items()}
    start = time.monotonic()
    process.stdin.write("go\n")
    process.stdin.flush()
    measured = read_event(process, args.seconds + 30)
    elapsed = time.monotonic() - start
    assert measured["event"] == "done" and measured["checksums_exact"] and measured["passes"] > 0
    entry = result["fixtures"][mode]
    assert measured["phase"] == entry["next_phase"]
    entry["next_phase"] += 1
    assert measured["end"] > measured["start"] and measured["end"] - measured["start"] >= args.seconds
    assert measured["bytes"] == measured["passes"] * weight_bytes
    inactive = {}
    for m, p in fixtures.items():
        if m == mode:
            continue
        ticks, identity = process_sample(p.pid)
        assert identity == before[m][1]
        inactive[m] = 100 * (ticks - before[m][0]) / os.sysconf("SC_CLK_TCK") / elapsed
    assert all(percent <= 5 for percent in inactive.values()), f"Inactive fixture used CPU: {inactive}"
    guard.check()
    record = dict(mode=mode, measurement=measured, inactive_fixture_cpu_percent=inactive,
                  started_monotonic=start, elapsed_seconds=elapsed, warmup=warmup)
    guard.pause(0.5)
    return record


def background():
    start = time.monotonic()
    guard.pause(5)
    window = [start + 0.5, time.monotonic() - 0.5]
    result["background_windows"].append(window)
    save()
    return len(result["background_windows"]) - 1


def interrupted(signum, frame):
    raise InterruptedError(f"Received signal {signum}")


for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(sig, interrupted)

save()
try:
    result["idle_gate"] = idle_gate()
    result["before_state"] = server_state()
    assert not result["before_state"]["busy"]
    guard = Guard()
    guard.thread.start()
    required_per_node = len(modes) * pool_bytes // 4 + 2 * pool_bytes // args.matrices + 512 * 1024**2
    result["node_free_kib"] = {}
    for node in range(4):
        lines = Path(f"/sys/devices/system/node/node{node}/meminfo").read_text().splitlines()
        free = int(next(line for line in lines if "MemFree:" in line).split()[-2])
        assert free * 1024 > required_per_node, f"Insufficient free memory on node {node}"
        result["node_free_kib"][str(node)] = free
    environment = {k: v for k, v in os.environ.items() if not k.startswith(("GGML_", "LLAMA_", "OMP_", "GOMP_"))}
    if qwen_experts:
        runtime = dict(item.decode().split("=", 1) for item in Path("/proc/2308651/environ").read_bytes().split(b"\0") if item.startswith(b"GGML_"))
        assert all(runtime.get(k) == "1" for k in ("GGML_CPU_IQ_R16_REPACK", "GGML_CPU_IQ_R16_NIBBLE2", "GGML_CPU_IQ_R16_BATCH3", "GGML_CPU_MOE_GATE_UP_FUSION", "GGML_CPU_NUMA_REPACK"))
        assert runtime.get("GGML_CPU_NUMA_THREADS") == "15"
        environment.update(runtime)
    else:
        baseline = json.loads((base / "results/glm53-draft-sweep-bandwidth-0905/result.json").read_text())
        assert baseline.get("finished") and not baseline.get("error")
        environment.update({k: v for k, v in baseline["runtime_env"].items() if k.startswith("GGML_")})
    environment.update(LD_LIBRARY_PATH=str(libraries), OMP_NUM_THREADS="1",
                       COLD_GRAPH_K_PER_SOCKET=str(args.k), COLD_GRAPH_ROWS=str(args.rows),
                       COLD_GRAPH_TOKENS=str(args.tokens), COLD_GRAPH_MATRICES=str(args.matrices),
                       COLD_GRAPH_GROUP="1", COLD_GRAPH_CONTROLLERS="4")
    result["fixture_environment"] = {k: v for k, v in environment.items()
                                     if k.startswith(("GGML_", "OMP_", "COLD_GRAPH_"))}
    dependencies = [source, Path(__file__).resolve(), base / "dram_bandwidth.py",
                    base / "inference_contention_guard.py",
                    engine / "ggml/src/ggml-backend-meta.cpp", engine / "ggml/src/ggml-cpu/repack.cpp",
                    engine / "ggml/src/ggml-cpu/arch/x86/repack.cpp"]
    dependencies += [p for p in libraries.glob("libggml*.so.*") if p.is_file()]
    if args.comparison == "counter_layout":
        dependencies.append(base / "build-private-counter-layout.py")
    result["input_sha256"] = {str(p): sha(p) for p in dependencies}
    for p in (source, Path(__file__).resolve()):
        (out / p.name).write_bytes(p.read_bytes())
    if args.comparison == "counter_layout":
        helper = base / "build-private-counter-layout.py"
        (out / helper.name).write_bytes(helper.read_bytes())
        spec = importlib.util.spec_from_file_location("private_counter_layout", helper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result["private_counter_build"] = module.build(engine, out / "counter-layout", run_build)
        save()
    command = ["g++", "-O3", "-std=c++17", "-march=native", "-pthread", str(source), "-o", str(binary)]
    command += ["-I" + str(engine / p) for p in ("include", "ggml/include", "ggml/src", "ggml/src/ggml-cpu")]
    command += ["-L" + str(libraries), "-lggml", "-lggml-cpu", "-lggml-base"]
    result["build_command"] = command
    save()
    run_build(command, base, "build.log")
    result["binary_sha256"] = sha(binary)
    for mode in modes:
        guard.check()
        log = (out / f"fixture-{mode}.log").open("w")
        logs.append(log)
        chunk = "64" if args.comparison == "counter_layout" else mode
        arm_environment = dict(environment)
        if args.comparison == "counter_layout":
            variant = result["private_counter_build"]["variants"][mode]
            arm_environment["LD_LIBRARY_PATH"] = variant["directory"] + ":" + str(libraries)
        command = ["taskset", "-c", "0-14,16-30,32-46,48-62",
                   str(binary), chunk, "15", "32", str(args.seconds)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                                   text=True, bufsize=1, env=arm_environment, start_new_session=True)
        owned.append(process)
        ready = read_event(process, 180)
        assert ready["event"] == "ready" and ready["pid"] == process.pid
        assert ready["chunk"] == int(chunk) and ready["workers"] == 60 and ready["group"] == 1
        assert ready["controller_cores"] == 4 and ready["bytes_per_pass"] == weight_bytes
        assert ready["tokens"] == args.tokens and ready["k"] == (args.k if qwen_experts else 4 * args.k)
        assert ready["rows"] == args.rows and ready["matrices"] == args.matrices
        fixtures[mode] = process
        result["fixtures"][mode] = dict(ready=ready, placement=placement(process.pid), next_phase=0)
        if qwen_experts:
            assert ready["experts"] == 32 and ready["used"] == 10 and ready["packed_weight_bytes"] == pool_bytes
            mapped = {line.split()[-1] for line in Path(f"/proc/{process.pid}/maps").read_text().splitlines() if "/libggml" in line}
            expected = {str(p.resolve()) for p in libraries.glob("libggml*.so.*") if p.is_file()}
            assert mapped == expected, f"Unexpected Qwen libraries: {mapped}"
            result["fixtures"][mode]["mapped_libraries"] = sorted(mapped)
        if args.comparison == "counter_layout":
            mapped = {line.split()[-1] for line in Path(f"/proc/{process.pid}/maps").read_text().splitlines()
                      if "/libggml-base.so." in line}
            assert mapped == {variant["library"]}, f"Unexpected libggml-base mapping: {mapped}"
            result["fixtures"][mode]["mapped_base_library"] = sorted(mapped)
        save()
        guard.pause(0.5)
    for key in ("weight_hash", "output_hash"):
        assert len({f["ready"][key] for f in result["fixtures"].values()}) == 1
    result["warmups"] = []
    for mode in modes:
        result["warmups"].append(phase(mode, warmup=True))
        save()
    recorder = PerfDramRecorder(out / "imc").start()
    previous_background = background()
    for block in range(args.rounds):
        block_phases = []
        for mode in order[block * 8:(block + 1) * 8]:
            record = phase(mode)
            record.update(index=len(result["phases"]), block=block,
                          background_before_index=previous_background)
            result["phases"].append(record)
            block_phases.append(record)
            save()
            print(json.dumps(dict(index=record["index"], mode=mode,
                                  graph_ms=record["measurement"]["graph_ms"],
                                  inactive_cpu_percent=record["inactive_fixture_cpu_percent"])), flush=True)
        previous_background = background()
        for record in block_phases:
            record["background_after_index"] = previous_background
        save()
    samples, metadata = recorder.stop()
    recorder = None
    assert metadata["valid"] and metadata["exit_code"] == 0
    result["counter_metadata"] = metadata
    result["backgrounds"] = [summarize_samples(samples, *window) for window in result["background_windows"]]
    assert all(b["valid"] for b in result["backgrounds"])
    for record in result["phases"]:
        measured = record["measurement"]
        window = [measured["start"] + 1, measured["end"] - 1]
        dram = summarize_samples(samples, *window)
        assert dram["valid"] and dram["samples"] >= 3
        backgrounds = [result["backgrounds"][record[key]]
                       for key in ("background_before_index", "background_after_index")]
        record.update(stable_window=window, dram=dram,
                      adjusted_read_gb_s=dram["read_gb_s"] - max(b["read_gb_s"] for b in backgrounds),
                      adjusted_total_gb_s=dram["total_gb_s"] - max(b["total_gb_s"] for b in backgrounds),
                      complete=True)
        record["read_counter_over_logical"] = record["adjusted_read_gb_s"] / measured["logical_gb_s"]
    for mode, process in fixtures.items():
        process.stdin.write("exit\n")
        process.stdin.flush()
        deadline = time.monotonic() + 10
        while process.poll() is None:
            assert time.monotonic() < deadline, "Fixture exit timed out"
            guard.pause(0.2)
        assert process.returncode == 0
        log_text = (out / f"fixture-{mode}.log").read_text()
        if not qwen_experts:
            match = re.search(r"fused in-graph all-reduce active: (\d+) boundaries \((\d+) tensors\)", log_text)
            assert match and tuple(map(int, match.groups())) == (args.matrices, args.matrices)
            assert "x16: repacking" in log_text
        result["fixtures"][mode]["exit_code"] = process.returncode
    assert result["binary_sha256"] == sha(binary)
    assert all(digest == sha(Path(p)) for p, digest in result["input_sha256"].items())
    if args.comparison == "counter_layout":
        private = result["private_counter_build"]
        assert all(digest == sha(Path(p)) for p, digest in private["input_sha256"].items())
        assert all(v["library_sha256"] == sha(Path(v["library"])) for v in private["variants"].values())
    ratios = []
    for index in range(0, len(result["phases"]), 2):
        pair = {r["mode"]: r["measurement"]["graph_ms"] for r in result["phases"][index:index + 2]}
        assert set(pair) == set(modes)
        ratios.append(pair[a_mode] / pair[b_mode])
    result["analysis"] = dict(
        baseline=a_mode, candidate=b_mode, pair_speedups=ratios,
        geometric_mean_speedup=math.exp(statistics.mean(math.log(r) for r in ratios)),
        median_pair_speedup=statistics.median(ratios),
        pairs_favoring_candidate=sum(r > 1 for r in ratios),
        median_graph_ms={mode: statistics.median(r["measurement"]["graph_ms"]
                                               for r in result["phases"] if r["mode"] == mode) for mode in modes},
        selected_for_production=False,
        scope="Adjacent balanced comparisons; no model gain follows from this synthetic result.")
    guard.close()
    guard = None
    result["after_state"] = server_state()
    assert not result["after_state"]["busy"]
    result["completed"] = True
    print(json.dumps(result["analysis"]), flush=True)
except BaseException as error:
    result["error"] = repr(error)
    raise
finally:
    terminate_owned()
    for process in owned:
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=3)
    if recorder is not None:
        try:
            recorder.stop()
        except BaseException as error:
            result["recorder_cleanup_error"] = repr(error)
    if guard is not None:
        try:
            guard.close()
        except BaseException as error:
            result["guard_cleanup_error"] = repr(error)
        result["guard_failure"] = guard.failure
    for log in logs:
        log.close()
    result["owned_exit_codes"] = {str(p.pid): p.poll() for p in owned}
    result["finished"] = time.time()
    save()
