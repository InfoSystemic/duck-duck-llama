#!/usr/bin/env python3
"""Run baseline -> tuned -> baseline on fresh, exclusively owned servers.

Observation deadlines leave live processes running and save their exact identity.
Resume that run with --resume; a timeout never starts a replacement process.
"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
STAGES = ("baseline-before", "tuned", "baseline-after")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def identity(pid):
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        fields = raw[raw.rfind(")") + 2:].split()
        return None if fields[0] == "Z" else fields[19]
    except FileNotFoundError:
        return None


def alive(handle):
    return bool(handle and handle.get("start_ticks") is not None and identity(handle["pid"]) == handle["start_ticks"])


def argv_for(pid):
    return [part.decode() for part in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if part]


def option(argv, name):
    if argv.count(name) != 1:
        raise RuntimeError(f"expected exactly one {name} in server argv")
    return argv[argv.index(name) + 1]


def owned_listener(pid, port):
    sockets = set()
    for fd in Path(f"/proc/{pid}/fd").iterdir():
        try:
            target = os.readlink(fd)
        except FileNotFoundError:
            continue
        if target.startswith("socket:["):
            sockets.add(target[8:-1])
    for table in ("tcp", "tcp6"):
        for row in Path(f"/proc/{pid}/net/{table}").read_text().splitlines()[1:]:
            fields = row.split()
            if fields[3] == "0A" and int(fields[1].split(":")[-1], 16) == port and fields[9] in sockets:
                return True
    return False


def require_server(handle, profile, port, listener=False):
    if not alive(handle):
        raise RuntimeError("owned server is terminal or its PID identity changed")
    pid = handle["pid"]
    argv = argv_for(pid)
    if Path(os.readlink(f"/proc/{pid}/exe")).resolve() != Path(profile["command"][0]).resolve():
        raise RuntimeError("owned PID is not the expected server executable")
    for key in ("--model", "--spec-draft-model"):
        if Path(option(argv, key)).resolve() != Path(option(profile["command"], key)).resolve():
            raise RuntimeError(f"owned server has unexpected {key}")
    if int(option(argv, "--port")) != port:
        raise RuntimeError("owned server has unexpected port")
    if profile["model"] not in option(argv, "--alias").split(","):
        raise RuntimeError("owned server lacks the exact model alias")
    if listener and not owned_listener(pid, port):
        raise RuntimeError("HTTP port is not owned by the benchmarked PID")


def stop_owned(handle, profile, port):
    if not alive(handle):
        return "already terminal"
    require_server(handle, profile, port)
    # A pidfd prevents a PID-reuse race between identity checking and signalling.
    fd = os.pidfd_open(handle["pid"])
    try:
        require_server(handle, profile, port)
        signal.pidfd_send_signal(fd, signal.SIGTERM)
        deadline = time.monotonic() + 30
        while alive(handle) and time.monotonic() < deadline:
            time.sleep(0.25)
        if alive(handle):
            signal.pidfd_send_signal(fd, signal.SIGKILL)
            deadline = time.monotonic() + 15
            while alive(handle) and time.monotonic() < deadline:
                time.sleep(0.25)
        if alive(handle):
            raise RuntimeError("owned server did not terminate; no subsequent arm may start")
        return "terminated owned PID"
    finally:
        os.close(fd)


def check_benchmark(directory, tokens):
    summary = read_json(directory / "summary.json")
    if not summary.get("passed") or not summary.get("finished_utc"):
        raise RuntimeError("benchmark failed or did not finish; refusing the next arm")
    if len(summary.get("quality", [])) != 3 or len(summary.get("rows", [])) != 3:
        raise RuntimeError("expected three quality checks and one pass through three benchmark prompts")
    for name, expected in (("arithmetic", "391"), ("fact", "canberra"), ("state", "10")):
        raw = read_json(directory / f"quality-{name}.json")
        choice = raw["response"]["choices"][0]
        content = (choice.get("message", {}).get("content") or "").casefold()
        if choice.get("finish_reason") != "stop" or not re.search(r"(?<!\w)" + expected + r"(?!\w)", content):
            raise RuntimeError(f"quality answer {name} lacks its exact answer token")
        if re.search(r"\b(?:not|isn't)\s+(?:\*\*)?" + expected + r"\b", content):
            raise RuntimeError(f"quality answer {name} negates its expected answer")
    for row in summary["rows"]:
        if not row.get("passed") or row.get("rep") != 0:
            raise RuntimeError("invalid or repeated benchmark row")
        timing = row["timings"]
        if timing.get("predicted_n") != tokens or timing.get("cache_n", 0) != 0:
            raise RuntimeError("short completion or prompt-cache reuse makes throughput incomparable")
    return summary


def compare(run_dir, stages):
    summaries = [check_benchmark(run_dir / stage / "benchmark", stages[stage]["tokens"]) for stage in STAGES]
    models = {summary["model"] for summary in summaries}
    if len(models) != 1:
        raise RuntimeError("model identity differs across arms")
    rows = []
    for prompt in ("prose", "code", "analysis"):
        raw = [read_json(run_dir / stage / "benchmark" / f"{prompt}-0.json") for stage in STAGES]
        if any(item["request"] != raw[0]["request"] for item in raw[1:]):
            raise RuntimeError(f"{prompt}: benchmark requests differ across arms")
        if any(item["response"].get("content") != raw[0]["response"].get("content") for item in raw[1:]):
            raise RuntimeError(f"{prompt}: greedy output differs across arms; review raw responses")
        timings = [item["response"]["timings"] for item in raw]
        if len({timing.get("prompt_n") for timing in timings}) != 1:
            raise RuntimeError(f"{prompt}: prompt token counts differ across arms")
        rates = [timing["predicted_per_second"] for timing in timings]
        control = statistics.mean((rates[0], rates[2]))
        rows.append({"prompt": prompt, "baseline_before": rates[0], "tuned": rates[1],
                     "baseline_after": rates[2], "control_mean": control,
                     "tuned_change_pct": 100 * (rates[1] / control - 1),
                     "control_drift_pct": 100 * (rates[2] / rates[0] - 1),
                     "greedy_output_identical": True})
    return {"quality_and_comparison_passed": True, "model": next(iter(models)), "rows": rows,
            "note": "One fresh-server sample per arm and prompt; this comparison does not establish statistical significance or automatically promote a candidate."}


def compare_to_first(run_dir, stage):
    if stage == STAGES[0]:
        return
    for prompt in ("prose", "code", "analysis"):
        first = read_json(run_dir / STAGES[0] / "benchmark" / f"{prompt}-0.json")
        current = read_json(run_dir / stage / "benchmark" / f"{prompt}-0.json")
        if first["request"] != current["request"]:
            raise RuntimeError(f"{stage}/{prompt}: request differs from the initial baseline")
        if first["response"].get("content") != current["response"].get("content"):
            raise RuntimeError(f"{stage}/{prompt}: greedy output differs; no subsequent arm will start")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("qwen", "glmflash"))
    parser.add_argument("--port", type=int, default=18131)
    parser.add_argument("--tokens", type=int, default=192)
    parser.add_argument("--load-seconds", type=int, default=3600)
    parser.add_argument("--benchmark-seconds", type=int, default=3600)
    parser.add_argument("--request-seconds", type=int, default=600)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.resume and not args.model:
        parser.error("--model or --resume is required")
    if args.tokens < 32 or min(args.load_seconds, args.benchmark_seconds, args.request_seconds) < 1:
        parser.error("at least 32 tokens and positive observation/request deadlines required")
    if args.resume:
        run_dir = args.resume.resolve()
        state = read_json(run_dir / "state.json")
        if args.model and args.model != state["model"]:
            parser.error("--model disagrees with resumed run")
        args.model, args.port, args.tokens = state["model"], state["port"], state["tokens"]
    else:
        run_dir = (args.out or HERE / "results" / ("ab-" + args.model + "-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))).resolve()
        state = {"model": args.model, "port": args.port, "tokens": args.tokens,
                 "created_utc": now(), "status": "new", "stages": {}, "events": []}
    profile_dir = HERE / args.model
    profile = read_json(profile_dir / "profile.json")
    if args.dry_run:
        print(json.dumps({"run_dir": str(run_dir), "sequence": STAGES, "model": profile["model"],
                          "port": args.port, "fresh_server_per_arm": True, "reps_per_server": 1,
                          "launch_commands": [[sys.executable, str(profile_dir / "launch.py"), "--arm",
                                               "tuned" if stage == "tuned" else "baseline", "--port", str(args.port)] for stage in STAGES],
                          "live_processes_started": False, "resume_state": state if args.resume else None}, indent=2))
        return 0
    if not args.resume:
        run_dir.mkdir(parents=True, exist_ok=False)
    lock = open(run_dir / ".runner.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def event(kind, **details):
        row = {"at": now(), "event": kind, **details}
        state["events"].append(row)
        write_json(run_dir / "state.json", state)
        print(json.dumps(row), flush=True)

    children = []
    current = None
    try:
        if state["status"] == "failed":
            raise RuntimeError("failed runs cannot resume into later arms; inspect failure and start a new run")
        for stage in STAGES:
            arm = "tuned" if stage == "tuned" else "baseline"
            record = state["stages"].get(stage)
            if record and record["status"] == "complete":
                continue
            stage_dir = run_dir / stage
            current = record
            if record is None:
                stage_dir.mkdir()
                preflight = subprocess.run([sys.executable, str(profile_dir / "launch.py"), "--arm", arm,
                                            "--port", str(args.port), "--dry-run"], capture_output=True, text=True)
                (stage_dir / "preflight.json").write_text(preflight.stdout)
                (stage_dir / "preflight.stderr").write_text(preflight.stderr)
                if preflight.returncode:
                    raise RuntimeError(f"{stage}: launch preflight failed; no server started")
                command = [sys.executable, str(profile_dir / "launch.py"), "--arm", arm, "--port", str(args.port)]
                with open(stage_dir / "server.log", "ab", buffering=0) as output:
                    child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                children.append(child)
                start_ticks = identity(child.pid)
                if start_ticks is None:
                    raise RuntimeError("launcher exited before its PID identity could be recorded")
                record = {"status": "loading", "tokens": args.tokens, "launch_command": command,
                          "server": {"pid": child.pid, "start_ticks": start_ticks}, "started_utc": now()}
                state["stages"][stage] = record
                current = record
                event("server_started", stage=stage, **record["server"])
            server = record["server"]
            if record["status"] == "loading":
                deadline, next_log = time.monotonic() + args.load_seconds, 0
                while True:
                    if not alive(server):
                        raise RuntimeError(f"{stage}: owned server became terminal during load")
                    try:
                        require_server(server, profile, args.port, listener=True)
                        with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=3) as response:
                            healthy = json.load(response).get("status") == "ok"
                        require_server(server, profile, args.port, listener=True)
                        if healthy:
                            break
                    except (RuntimeError, urllib.error.URLError, TimeoutError):
                        pass
                    if time.monotonic() >= deadline:
                        state["status"] = "observation_timeout"
                        event("load_observation_timeout_process_preserved", stage=stage, **server)
                        return 3
                    if time.monotonic() >= next_log:
                        event("verified_load_wait", stage=stage, **server,
                              io=Path(f"/proc/{server['pid']}/io").read_text().splitlines())
                        next_log = time.monotonic() + 30
                    time.sleep(3)
                event("server_healthy_and_listener_owned", stage=stage, **server)
                command = [sys.executable, str(HERE / "benchmark_live.py"), "--pid", str(server["pid"]),
                           "--model", profile["model"], "--port", str(args.port), "--reps", "1",
                           "--tokens", str(args.tokens), "--wait-seconds", "30", "--timeout", str(args.request_seconds),
                           "--out", str(stage_dir / "benchmark")]
                with open(stage_dir / "benchmark.log", "ab", buffering=0) as output:
                    child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                children.append(child)
                record["benchmark"] = {"pid": child.pid, "start_ticks": identity(child.pid)}
                record["status"] = "benchmarking"
                event("benchmark_started", stage=stage, **record["benchmark"])
            if record["status"] == "benchmarking":
                deadline, next_log = time.monotonic() + args.benchmark_seconds, 0
                while alive(record["benchmark"]):
                    require_server(server, profile, args.port, listener=True)
                    if time.monotonic() >= deadline:
                        state["status"] = "observation_timeout"
                        event("benchmark_observation_timeout_processes_preserved", stage=stage,
                              server=server, benchmark=record["benchmark"])
                        return 3
                    if time.monotonic() >= next_log:
                        event("verified_benchmark_wait", stage=stage, benchmark=record["benchmark"])
                        next_log = time.monotonic() + 30
                    time.sleep(3)
                require_server(server, profile, args.port, listener=True)
                check_benchmark(stage_dir / "benchmark", args.tokens)
                compare_to_first(run_dir, stage)
                record["status"] = "stopping"
                event("benchmark_quality_passed", stage=stage)
            if record["status"] == "stopping":
                result = stop_owned(server, profile, args.port)
                record["status"] = "complete"
                record["finished_utc"] = now()
                event("server_stopped", stage=stage, result=result, **server)
            for child in children:
                child.poll()
        comparison = compare(run_dir, state["stages"])
        write_json(run_dir / "comparison.json", comparison)
        state["status"] = "complete"
        event("comparison_complete", comparison=str(run_dir / "comparison.json"))
        return 0
    except KeyboardInterrupt:
        state["status"] = "interrupted"
        event("runner_interrupted_live_processes_preserved")
        return 130
    except Exception as error:
        state["status"] = "failed"
        event("failed", error=repr(error))
        if current and alive(current.get("server")) and not alive(current.get("benchmark")):
            try:
                result = stop_owned(current["server"], profile, args.port)
                event("owned_server_cleanup", result=result)
            except Exception as cleanup_error:
                event("cleanup_refused_or_failed_process_preserved", error=repr(cleanup_error))
        return 1
    finally:
        for child in children:
            child.poll()
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
