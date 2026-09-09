#!/usr/bin/env python3
"""Compare private Qwen expert row scheduling with identical tensor splits."""
import argparse
import array
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
parser.add_argument("--comparison", choices=("expert_moe_schedule",), default="expert_moe_schedule")
parser.add_argument("--chunks", default="64,32", help="Baseline,candidate limits for the chunk comparison")
parser.add_argument("--qwen-weight-type", choices=("iq2_xs", "iq3_xxs", "q8_0"), default="iq2_xs")
parser.add_argument("--qwen-split-manifest", type=Path, help="Validated private split-policy manifest for the complete expert graph")
parser.add_argument("--task-rows", type=int, choices=(16, 32, 64), default=16)
parser.add_argument("--split-granularity", type=int, choices=(128, 32), default=128)
parser.add_argument("--qwen-cpu-manifest", type=Path, help="Reuse a completed private task-row CPU build")
args = parser.parse_args()
assert re.fullmatch(r"[A-Za-z0-9_-]+", args.label)
assert 256 <= args.k <= 8192 and args.k % 256 == 0
assert 64 <= args.rows <= 16384 and args.rows % 64 == 0
assert 1 <= args.matrices <= 64 and 3 <= args.seconds <= 10 and 1 <= args.rounds <= 8

base = Path(__file__).resolve().parent
qwen_moe = qwen_experts = True
assert qwen_experts or args.qwen_weight_type == "iq2_xs"
assert args.qwen_weight_type != "q8_0" or qwen_moe
assert qwen_moe == bool(args.qwen_split_manifest)
if args.qwen_split_manifest:
    args.qwen_split_manifest = str(args.qwen_split_manifest.resolve())
engine = base.parents[1] / "engines" / ("llama.cpp-q4e-goal-0904" if qwen_experts else "llama.cpp-sr950-glm")
libraries = engine / ("validated-iq-batch3-bin" if qwen_experts else "build-dev2/bin")
out = base / "results" / args.label
out.mkdir(exist_ok=False)
source = base / ("qwen-expert-graph-check.cpp" if qwen_experts else "cold-graph-check.cpp")
binary = out / ("qwen-expert-graph-check" if qwen_experts else "cold-graph-check")
allowed = {"4005448": 18091, "2308651": 18095}
protected_plan = base / "results/qwen-even-split-model-trial-0906-staging/plan.json"
protected = json.loads(protected_plan.read_text())["protected"]
chunks = tuple(args.chunks.split(","))
assert len(chunks) == 2 and len(set(chunks)) == 2 and all(c in ("16", "32", "64", "128", "256") for c in chunks)
if qwen_experts:
    assert args.k == 2560 and args.rows == 640 and 4 <= args.matrices <= 16 and args.matrices % 4 == 0
    assert set(chunks) == {"128", "32"}
modes = ("baseline", "candidate")
a_mode, b_mode = modes
order = [a_mode, b_mode, b_mode, a_mode, b_mode, a_mode, a_mode, b_mode] * args.rounds
weight_bytes = args.matrices * (args.rows // 16) * (4 * args.k // 256) * 2880
pool_bytes = weight_bytes
if qwen_experts:
    per_expert = args.matrices * 2 * (args.rows // 16) * (args.k // 256) * 2208
    if args.qwen_weight_type == "q8_0":
        per_expert = args.matrices * 2 * args.rows * (args.k // 32) * 34
    if qwen_moe:
        per_expert += args.matrices * args.k * (args.rows // 32) * (34 if args.qwen_weight_type == "q8_0" else 18)
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
        assert current[pid][1] == protected[pid]["start_ticks"], "A protected service identity changed"
        assert os.readlink(f"/proc/{pid}/exe") == protected[pid]["exe"], "A protected service executable changed"

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
        if qwen_moe:
            assert runtime.get("GGML_CPU_NUMA_FUSED_REDUCE") == "1"
            assert runtime.get("GGML_CPU_MOE_DOWN_WEIGHTED_SUM_FUSION", "0") == "0"
            if args.qwen_weight_type == "q8_0":
                assert runtime.get("GGML_CPU_Q8_0_REPACK") == runtime.get("GGML_CPU_Q8_0_REPACK_FORCE") == "1"
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
                    base / "build-private-qwen-task-rows.py", protected_plan,
                    base / "inference_contention_guard.py",
                    engine / "ggml/src/ggml-backend-meta.cpp", engine / "ggml/src/ggml-cpu/repack.cpp",
                    engine / "ggml/src/ggml-cpu/arch/x86/repack.cpp"]
    dependencies += [p for p in libraries.glob("libggml*.so.*") if p.is_file()]
    if qwen_moe:
        manifest_path = Path(args.qwen_split_manifest)
        assert manifest_path.is_relative_to(base / "results"), "Expected an existing private candidate"
        private_policy = json.loads(manifest_path.read_text())
        policy_library = Path(private_policy["library"])
        assert policy_library.parent == manifest_path.parent
        assert policy_library.is_relative_to(base / "results")
        assert private_policy["library_sha256"] == sha(policy_library)
        assert all(sha(Path(p)) == digest for p, digest in private_policy["input_sha256"].items())
        result["private_policy"] = dict(manifest=str(manifest_path), library=str(policy_library),
                                        library_sha256=private_policy["library_sha256"],
                                        input_sha256=private_policy["input_sha256"])
        dependencies += [manifest_path, policy_library, engine / "src/llama-model.cpp", engine / "src/llama-model.h"]
        dependencies += [p for p in libraries.glob("libllama.so.*") if p.is_file()]
        if args.qwen_weight_type == "q8_0":
            metadata_path = base / "results/qwen-mtp-expert-metadata-0906.json"
            draft = json.loads(metadata_path.read_text())
            metadata = draft["metadata"]
            assert metadata["general.architecture"] == "qwen4exp"
            expected_metadata = {"block_count": 49, "nextn_predict_layers": 1, "embedding_length": 2560,
                                 "expert_feed_forward_length": 640, "expert_count": 512, "expert_used_count": 10}
            assert all(metadata["qwen4exp." + key] == value for key, value in expected_metadata.items())
            assert draft["expert_tensors"] == [dict(name=f"blk.48.ffn_{kind}_exps.weight",
                                                     shape=[640, 2560, 512] if kind == "down" else [2560, 640, 512],
                                                     type="Q8_0") for kind in ("gate", "up", "down")]
            stat = Path(draft["path"]).stat()
            assert stat.st_size == draft["size"] and stat.st_mtime_ns == draft["mtime_ns"]
            result["mtp_metadata"] = dict(path=str(metadata_path), model=draft["path"],
                                          header_sha256=draft["header_sha256"], expected=expected_metadata)
            dependencies += [metadata_path, engine / "src/models/qwen4exp.cpp"]
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
    helper = base / "build-private-qwen-task-rows.py"
    (out / helper.name).write_bytes(helper.read_bytes())
    if args.qwen_cpu_manifest:
        cpu_manifest = args.qwen_cpu_manifest.resolve()
        assert cpu_manifest.is_relative_to(base / "results")
        private_cpu = json.loads(cpu_manifest.read_text())
        assert all(sha(Path(p)) == digest for p, digest in private_cpu["input_sha256"].items())
    else:
        spec = importlib.util.spec_from_file_location("private_qwen_task_rows", helper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        private_cpu = module.build(engine, out / "private-cpu", run_build)
        cpu_manifest = out / "private-cpu/manifest.json"
    cpu_library = Path(private_cpu["library"])
    assert cpu_library.parent == cpu_manifest.parent and cpu_library.is_relative_to(base / "results")
    assert sha(cpu_library) == private_cpu["library_sha256"]
    result["private_cpu"] = dict(manifest=str(cpu_manifest), **private_cpu)
    result["input_sha256"][str(cpu_manifest)] = sha(cpu_manifest)
    result["input_sha256"][str(cpu_library)] = sha(cpu_library)
    command = ["g++", "-O3", "-std=c++17", "-march=native", "-pthread", str(source), "-o", str(binary)]
    if qwen_experts and args.qwen_weight_type == "iq3_xxs":
        command.append("-DQWEN_IQ3_CHECK")
    if args.qwen_weight_type == "q8_0":
        command.append("-DQWEN_Q8_CHECK")
    if qwen_moe:
        command += ["-DQWEN_MOE_CHECK", "-I" + str(engine / "src")]
    command += ["-I" + str(engine / p) for p in ("include", "ggml/include", "ggml/src", "ggml/src/ggml-cpu")]
    command += ["-L" + str(libraries), "-lggml", "-lggml-cpu", "-lggml-base"]
    if qwen_moe:
        command += ["-lllama", "-ldl"]
    assert command.count("-o") == 1 and Path(command[command.index("-o") + 1]).parent.resolve() == out.resolve()
    assert not binary.exists(), "Refuse to replace an existing build output"
    result["build_command"] = command
    save()
    run_build(command, base, "build.log")
    result["binary_sha256"] = sha(binary)
    for mode in modes:
        guard.check()
        log = (out / f"fixture-{mode}.log").open("w")
        logs.append(log)
        chunk = str(args.split_granularity)
        arm_environment = dict(environment)
        if qwen_moe:
            arm_environment["GGML_Q4E_EXPERT_EVEN_SPLIT"] = "1" if args.split_granularity == 32 else "0"
            arm_environment["GGML_CPU_IQ_MOE_TASK_ROWS"] = str(args.task_rows) if mode == "candidate" else "0"
            arm_environment["COLD_GRAPH_OUTPUT_PATH"] = str(out / f"outputs-{mode}.f32")
            assert not Path(arm_environment["COLD_GRAPH_OUTPUT_PATH"]).exists()
            search = ([str(cpu_library.parent)] if mode == "candidate" else [])
            if args.split_granularity == 32:
                search.append(str(policy_library.parent))
            search.append(str(libraries))
            arm_environment["LD_LIBRARY_PATH"] = ":".join(search)
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
            assert ready["weight_type"] == args.qwen_weight_type
            mapped = {line.split()[-1] for line in Path(f"/proc/{process.pid}/maps").read_text().splitlines() if "/libggml" in line}
            expected = {str(p.resolve()) for p in libraries.glob("libggml*.so.*") if p.is_file()}
            if mode == "candidate":
                expected = {p for p in expected if "/libggml-cpu.so." not in p}
                expected.add(str(cpu_library))
            assert mapped == expected, f"Unexpected Qwen libraries: {mapped}"
            result["fixtures"][mode]["mapped_libraries"] = sorted(mapped)
            if qwen_moe:
                assert ready["down_checked"] and ready["max_down_scaled_error"] <= 2e-5
                expected_llama = policy_library if args.split_granularity == 32 else (libraries / "libllama.so.0").resolve()
                mapped_llama = {line.split()[-1] for line in Path(f"/proc/{process.pid}/maps").read_text().splitlines() if "/libllama.so." in line}
                assert mapped_llama == {str(expected_llama)}
                assert Path(ready["policy_library"]).resolve() == expected_llama
                result["fixtures"][mode]["mapped_policy_library"] = str(expected_llama)
        if args.comparison == "counter_layout":
            mapped = {line.split()[-1] for line in Path(f"/proc/{process.pid}/maps").read_text().splitlines()
                      if "/libggml-base.so." in line}
            assert mapped == {variant["library"]}, f"Unexpected libggml-base mapping: {mapped}"
            result["fixtures"][mode]["mapped_base_library"] = sorted(mapped)
        save()
        guard.pause(0.5)
    for key in (("weight_hash",) if qwen_moe else ("weight_hash", "output_hash")):
        assert len({f["ready"][key] for f in result["fixtures"].values()}) == 1
    if qwen_moe:
        compared = {}
        output_hashes = {}
        count = 3 * args.matrices * args.k * 10 * args.tokens
        for mode in modes:
            path = out / f"outputs-{mode}.f32"
            assert path.stat().st_size == count * 4
            values = array.array("f")
            assert values.itemsize == 4
            with path.open("rb") as stream:
                values.fromfile(stream, count)
            assert all(math.isfinite(value) for value in values)
            compared[mode] = values
            output_hashes[str(path)] = sha(path)
        differences = [abs(a - b) for a, b in zip(compared[a_mode], compared[b_mode])]
        scaled = [diff / (1 + max(abs(a), abs(b))) for a, b, diff in zip(compared[a_mode], compared[b_mode], differences)]
        assert max(scaled) == 0 and len(set(output_hashes.values())) == 1, "Identical splits must give exact outputs across schedulers"
        result["cross_layout_outputs"] = dict(values=count, max_absolute_error=max(differences),
                                               max_scaled_error=max(scaled), tolerance=0,
                                               sha256=output_hashes)
        del compared, differences, scaled
        guard.check()
        save()
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
        if qwen_experts:
            if mode == "candidate" and args.qwen_weight_type != "q8_0":
                assert f"IQ_MOE_TASK_ROWS rows={args.task_rows} " in log_text, "Private scheduler was not exercised"
            else:
                assert "IQ_MOE_TASK_ROWS" not in log_text
            layout = "q8_0_8x8" if args.qwen_weight_type == "q8_0" else f"{args.qwen_weight_type}_16x8"
            assert f"with {layout}" in log_text, "Intended gate/up packed layout was not exercised"
            if qwen_moe:
                down_layout = "q8_0_8x8" if args.qwen_weight_type == "q8_0" else "iq4_nl_8x8"
                assert f"with {down_layout}" in log_text, "Intended down packed layout was not exercised"
                match = re.search(r"fused in-graph all-reduce active: (\d+) boundaries \((\d+) tensors\)", log_text)
                assert match and tuple(map(int, match.groups())) == (args.matrices, args.matrices)
        else:
            match = re.search(r"fused in-graph all-reduce active: (\d+) boundaries \((\d+) tensors\)", log_text)
            assert match and tuple(map(int, match.groups())) == (args.matrices, args.matrices)
            assert "x16: repacking" in log_text
        result["fixtures"][mode]["exit_code"] = process.returncode
    assert result["binary_sha256"] == sha(binary)
    assert all(digest == sha(Path(p)) for p, digest in result["input_sha256"].items())
    assert all(sha(Path(p)) == digest for p, digest in private_cpu["input_sha256"].items())
    assert sha(cpu_library) == private_cpu["library_sha256"]
    if qwen_moe:
        assert all(sha(Path(p)) == digest for p, digest in private_policy["input_sha256"].items())
        assert all(sha(Path(p)) == digest for p, digest in result["cross_layout_outputs"]["sha256"].items())
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
