#!/usr/bin/env python3
"""Run an isolated Flash-model TP case and preserve complete responses."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request
from inference_contention_guard import InferenceContentionGuard, wait_for_idle

p = argparse.ArgumentParser()
p.add_argument("label")
p.add_argument("--binary", default="/dev/shm/q4e-fast/build-fast/bin/llama-server")
p.add_argument("--model", default="/models/gguf/Qwen3.8-Flash-Next/UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf")
p.add_argument("--alias", default="qwen-goal")
p.add_argument("--runtime", choices=["llama", "ik"], default="llama")
p.add_argument("--ctx-size", type=int, default=4096)
p.add_argument("--mtp")
p.add_argument("--spec-type", choices=["draft-mtp", "ngram-simple,draft-mtp"], default="draft-mtp")
p.add_argument("--ngram-n", type=int, default=4)
p.add_argument("--ngram-m", type=int, default=4)
p.add_argument("--draft-n", type=int, default=1)
p.add_argument("--draft-sweep", help="Comma-separated MTP draft limits, bounded by --draft-n")
p.add_argument("--draft-p-min", type=float, default=0)
p.add_argument("--devices", default="CPU-NUMA0,CPU-NUMA1,CPU-NUMA2,CPU-NUMA3")
p.add_argument("--stage", type=int, default=13)
p.add_argument("--direct", type=int, default=0)
p.add_argument("--threads", type=int, default=15)
p.add_argument("--numa-poll", type=int, choices=range(101), default=100)
p.add_argument("--thread-sweep", help="Comma-separated active NUMA worker counts, bounded by --threads")
p.add_argument("--fused", type=int, default=0)
p.add_argument("--merge", type=int, default=0)
p.add_argument("--repack", type=int, default=0)
p.add_argument("--huge-pages", type=int, choices=[0, 1], default=0)
p.add_argument("--port", type=int, default=18105)
p.add_argument("--bench", action="store_true")
p.add_argument("--bench-direct", action="store_true")
p.add_argument("--bench-tokens", type=int, default=160)
p.add_argument("--cache-check", action="store_true")
p.add_argument("--profile-cpu", action="store_true")
p.add_argument("--profile-phase", action="store_true")
p.add_argument("--profile-perf", action="store_true")
p.add_argument("--profile-answer-prefix", action="store_true")
a = p.parse_args()
if a.spec_type != "draft-mtp" and (a.runtime != "llama" or not a.mtp):
    p.error("combined ngram/MTP speculation requires a llama MTP model")
if not (1 <= a.ngram_n <= 1024 and a.ngram_n <= a.ngram_m <= 1024):
    p.error("ngram sizes require 1 <= n <= m <= 1024")
sweep_threads = []
sweep_drafts = []
if a.draft_sweep:
    try:
        sweep_drafts = list(dict.fromkeys(int(value) for value in a.draft_sweep.split(",")))
    except ValueError:
        p.error("--draft-sweep requires comma-separated integers")
    if (a.runtime != "llama" or not a.mtp or not a.bench or a.thread_sweep or
            not all(1 <= value <= a.draft_n for value in sweep_drafts)):
        p.error("--draft-sweep requires llama MTP, --bench, limits within --draft-n, and no --thread-sweep")
initial_draft_n = sweep_drafts[0] if sweep_drafts else a.draft_n
if a.thread_sweep:
    try:
        sweep_threads = list(dict.fromkeys(int(value) for value in a.thread_sweep.split(",")))
    except ValueError:
        p.error("--thread-sweep requires comma-separated integers")
    if a.runtime != "llama" or not a.bench or not all(1 <= value <= a.threads for value in sweep_threads):
        p.error("--thread-sweep requires --bench, llama runtime, and counts from 1 to --threads")
if a.bench_direct and (not a.bench or a.runtime != "llama"):
    p.error("--bench-direct requires --bench and the llama runtime")
if a.runtime == "ik" and (a.mtp or a.profile_cpu or a.profile_phase or a.profile_perf):
    p.error("The ik comparison supports target-only execution without this CPU profiler")
out = Path(__file__).resolve().parent / "results" / a.label
out.mkdir(exist_ok=True)
with socket.socket() as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", a.port))
env = dict(os.environ, GGML_CPU_NUMA_DEVICES="1", GGML_CPU_NUMA_THREADS=str(a.threads),
           GGML_CPU_NUMA_POLL=str(a.numa_poll), GGML_CPU_NUMA_REPACK=str(a.repack), GGML_Q4E_SPLIT=str(a.stage),
           GGML_CPU_NUMA_FUSED_REDUCE=str(a.fused), GGML_CPU_NUMA_MERGE_REDUCE=str(a.merge),
           GGML_CPU_REPACK_LOAD_THREADS="16",
           GGML_CPU_NUMA_DIRECT_ALLREDUCE=str(a.direct), GGML_CPU_NUMA_HUGEPAGES=str(a.huge_pages))
env["LD_LIBRARY_PATH"] = str(Path(a.binary).resolve().parent) + ":" + env.get("LD_LIBRARY_PATH", "")
threads_file = out / "numa-threads"
if sweep_threads:
    threads_file.write_text(str(a.threads) + "\n")
    env["GGML_CPU_NUMA_THREADS_FILE"] = str(threads_file)
def set_active_threads(value):
    temporary = threads_file.with_suffix(".tmp")
    temporary.write_text(str(value) + "\n")
    os.replace(temporary, threads_file)
drafts_file = out / "draft-n"
if sweep_drafts:
    drafts_file.write_text(str(initial_draft_n) + "\n")
    env["LLAMA_MTP_DRAFT_N_FILE"] = str(drafts_file)
def set_active_drafts(value):
    temporary = drafts_file.with_suffix(".tmp")
    temporary.write_text(str(value) + "\n")
    os.replace(temporary, drafts_file)
cpu_profile_arm = out / "cpu-profile.arm"
phase_profile_arm = out / "phase-profile.arm"
if a.profile_cpu:
    cpu_profile_arm.unlink(missing_ok=True)
    env.update(GGML_CPU_OP_PROFILE="*", GGML_CPU_OP_PROFILE_ARM_FILE=str(cpu_profile_arm),
               GGML_CPU_OP_PROFILE_SKIP="0", GGML_CPU_OP_PROFILE_COUNT="96" if a.mtp else "16")
if a.profile_phase:
    phase_profile_arm.unlink(missing_ok=True)
    env["LLAMA_GRAPH_PHASE_ARM_FILE"] = str(phase_profile_arm)
cmd = [a.binary, "--host", "127.0.0.1", "--port", str(a.port), "--model",
       a.model, "--alias", a.alias, "--load-mode", "mmap", "--fit", "off", "--ctx-size", str(a.ctx_size),
       "--flash-attn", "on", "--batch-size", "512", "--ubatch-size", "256", "--parallel", "1",
       "--gpu-layers", "999", "--device", a.devices, "--split-mode", "tensor", "--tensor-split",
       ",".join("1" for _ in a.devices.split(",")), "--threads", str(a.threads),
       "--threads-batch", str(a.threads), "--jinja", "--reasoning-format", "deepseek",
       "--reasoning-preserve", "--no-webui", "--metrics", "--verbosity", "4"]
if a.runtime == "ik":
    cmd = ["numactl", "--physcpubind=0-63", "--interleave=all", a.binary,
           "--host", "127.0.0.1", "--port", str(a.port), "--model", a.model,
           "--alias", a.alias, "--ctx-size", str(a.ctx_size), "--flash-attn", "on",
           "--batch-size", "512", "--ubatch-size", "256", "--parallel", "1",
           "--gpu-layers", "0", "--threads", str(a.threads), "--threads-batch", str(a.threads),
           "--numa", "numactl", "--jinja", "--reasoning-format", "deepseek",
           "--reasoning", "off", "--metrics", "--verbosity", "4"]
    if a.repack:
        cmd.append("--run-time-repack")
if a.mtp:
    cmd += ["--spec-type", a.spec_type, "--spec-draft-model", a.mtp,
            "--spec-draft-device", a.devices, "--spec-draft-ngl", "all",
            "--spec-draft-n-max", str(a.draft_n), "--spec-draft-p-min", str(a.draft_p_min)]
    if a.spec_type == "ngram-simple,draft-mtp":
        cmd += ["--spec-ngram-simple-size-n", str(a.ngram_n), "--spec-ngram-simple-size-m", str(a.ngram_m)]
result = {"config": vars(a), "command": cmd, "started": time.time(), "checks": []}
result["runtime_env"] = {k: v for k, v in env.items() if k.startswith(("GGML_", "LLAMA_GRAPH_PHASE", "LLAMA_MTP_DRAFT_N_FILE", "OMP_", "GOMP_"))}
(out / "config.json").write_text(json.dumps(result, indent=2))
def request(path, payload=None, timeout=600):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}/{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def host_cpu_snapshot():
    samples = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            stat = (proc / "stat").read_text()
            fields = stat[stat.rfind(")") + 2:].split()
            samples[proc.name] = [int(fields[11]) + int(fields[12]), fields[19],
                                  stat[stat.find("(") + 1:stat.rfind(")")]]
        except (OSError, ValueError, IndexError):
            pass
    pressure = {}
    for name in ["cpu", "memory", "io"]:
        try:
            pressure[name] = Path("/proc/pressure/" + name).read_text()
        except OSError:
            pass
    return {"processes": samples, "pressure": pressure, "time": time.monotonic()}

def inference_cpu_ticks():
    samples = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            process_name = (proc / "comm").read_text().strip()
            if process_name != "llama-server" and not process_name.startswith(("glm-mtp-head", "qwen-mtp-head")):
                continue
            fields = (proc / "stat").read_text().split()
            args = (proc / "cmdline").read_bytes().split(b"\0")
            port = args[args.index(b"--port") + 1].decode() if b"--port" in args else None
            samples[proc.name] = (int(fields[13]) + int(fields[14]), fields[21], port)
        except (OSError, ValueError, IndexError):
            continue
    return samples

def run_check(prompt, expected, reasoning_budget, effective_threads, effective_draft_n=None):
    host_before = host_cpu_snapshot()
    cpu_before = inference_cpu_ticks()
    request_started = time.time()
    started = time.monotonic()
    payload = {"model": a.alias, "messages": [
        {"role": "user", "content": prompt}], "temperature": 0, "seed": 42,
        "max_tokens": a.bench_tokens if expected is None else 64, "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False}}
    if reasoning_budget is not None:
        payload["reasoning_budget_tokens"] = reasoning_budget
    response = request("v1/chat/completions", payload)
    content = response["choices"][0]["message"].get("content") or ""
    check = {"prompt": prompt, "expected": expected, "response": response,
             "reasoning_budget_tokens": reasoning_budget,
             "started": request_started, "finished": time.time(),
             "wall_s": time.monotonic() - started,
             "pass": None if expected is None else content.strip().rstrip(".") == expected}
    check["effective_numa_threads"] = effective_threads
    check["effective_draft_n"] = a.draft_n if effective_draft_n is None else effective_draft_n
    check["effective_mtp_draft_n"] = check["effective_draft_n"]
    check["effective_ngram_max"] = a.ngram_m if a.spec_type == "ngram-simple,draft-mtp" else None
    cpu_after = inference_cpu_ticks()
    check["other_inference_cpu_percent"] = [
        {"pid": int(pid), "port": sample[2], "cpu_percent": round(
            100 * (sample[0] - cpu_before[pid][0]) / os.sysconf("SC_CLK_TCK") / check["wall_s"], 2)}
        for pid, sample in cpu_after.items()
        if pid != str(server.pid) and pid in cpu_before and sample[1] == cpu_before[pid][1]]
    check["throughput_contended"] = any(
        sample["cpu_percent"] > 100 for sample in check["other_inference_cpu_percent"])
    check["other_inference_process_churn"] = sorted(
        (set(cpu_before) ^ set(cpu_after)) - {str(server.pid)})
    check["throughput_contended"] |= bool(check["other_inference_process_churn"])
    host_after = host_cpu_snapshot()
    host_seconds = host_after["time"] - host_before["time"]
    host_load = []
    for pid, item in host_after["processes"].items():
        previous = host_before["processes"].get(pid)
        if pid == str(server.pid) or previous is None or item[1] != previous[1]:
            continue
        percent = 100 * (item[0] - previous[0]) / os.sysconf("SC_CLK_TCK") / host_seconds
        if percent >= 5:
            host_load.append({"pid": int(pid), "name": item[2], "cpu_percent": round(percent, 2)})
    check["other_host_cpu_percent"] = sorted(host_load, key=lambda item: -item["cpu_percent"])[:20]
    check["host_pressure"] = {"before": host_before["pressure"], "after": host_after["pressure"]}
    print(json.dumps({"effective_numa_threads": effective_threads, "effective_draft_n": check["effective_draft_n"], "pass": check["pass"], "content": content[:180], "reasoning_budget_tokens": reasoning_budget,
                      "timings": response.get("timings")}), flush=True)
    return check

result["idle_before_launch"] = wait_for_idle(inference_cpu_ticks, out / "waiting-for-idle.json")
result["launch_started"] = time.time()
with (out / "server.log").open("w") as log:
    server = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    (out / "server.pid").write_text(str(server.pid))
    guard = InferenceContentionGuard(server, inference_cpu_ticks, out / "contention-guard.json")
    guard.start()
    try:
        deadline = time.monotonic() + 900
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"server exited {server.returncode}")
            try:
                request("health", timeout=2)
                break
            except (OSError, ValueError):
                if time.monotonic() >= deadline:
                    raise TimeoutError("model load exceeded 900 seconds")
                time.sleep(2)
        print(f"{a.label}: healthy after {time.time()-result['launch_started']:.1f}s", flush=True)
        result["memory_after_load"] = {}
        for filename, keys in [("status", ("VmRSS:", "RssAnon:", "RssFile:")),
                               ("smaps_rollup", ("AnonHugePages:", "FilePmdMapped:", "Pss:"))]:
            try:
                lines = Path(f"/proc/{server.pid}/{filename}").read_text().splitlines()
                result["memory_after_load"].update({line.split(":", 1)[0]: line.split(":", 1)[1].strip()
                                                    for line in lines if line.startswith(keys)})
            except OSError:
                pass
        checks = [("What is 17 * 23? Reply with only the number.", "391", None),
                  ("Name the capital city of France. Reply with one word.", "Paris", None),
                  ("Complete: 2, 4, 6, 8, 10, 12. Give the next three numbers only.", "14, 16, 18", None)]
        if a.bench:
            checks += [("Explain how a refrigerator moves heat. Give a detailed explanation in plain English.", None, None),
                       ("Write a Python function that merges two sorted lists. Include an explanation of its time complexity.", None, None)]
            if a.bench_direct:
                checks += [(prompt, expected, 0) for prompt, expected, _ in checks[-2:]]
        for prompt, expected, reasoning_budget in checks:
            if expected is None and not all(c["pass"] for c in result["checks"] if c["expected"] is not None):
                result["benchmark_skipped"] = "correctness check failed"
                break
            check = run_check(prompt, expected, reasoning_budget, a.threads, initial_draft_n)
            result["checks"].append(check)
            (out / "result.json").write_text(json.dumps(result, indent=2))
        if sweep_threads:
            result["thread_sweep"] = [{"threads": a.threads, "checks": result["checks"], "baseline": True}]
            for active_threads in sweep_threads:
                if active_threads == a.threads:
                    continue
                set_active_threads(active_threads)
                case = {"threads": active_threads, "checks": []}
                result["thread_sweep"].append(case)
                for prompt, expected, reasoning_budget in checks:
                    if reasoning_budget is not None:
                        continue
                    if expected is None and not all(check["pass"] for check in case["checks"] if check["expected"] is not None):
                        break
                    case["checks"].append(run_check(prompt, expected, reasoning_budget, active_threads))
                    (out / "result.json").write_text(json.dumps(result, indent=2))
            def sweep_score(case):
                bench = [check for check in case["checks"] if check["expected"] is None and check["reasoning_budget_tokens"] is None]
                if len(bench) != 2 or any(check["throughput_contended"] for check in bench):
                    return 0
                return min(check["response"]["timings"]["predicted_per_second"] for check in bench)
            winner = max(result["thread_sweep"], key=sweep_score)
            result["selected_numa_threads"] = winner["threads"]
            result["thread_sweep_scores"] = {str(case["threads"]): sweep_score(case) for case in result["thread_sweep"]}
            set_active_threads(winner["threads"])
        if sweep_drafts:
            result["draft_sweep"] = [{"draft_n": initial_draft_n, "checks": result["checks"], "baseline": True}]
            for active_drafts in sweep_drafts[1:]:
                set_active_drafts(active_drafts)
                case = {"draft_n": active_drafts, "checks": []}
                result["draft_sweep"].append(case)
                for prompt, expected, reasoning_budget in checks:
                    if reasoning_budget is not None:
                        continue
                    if expected is None and not all(check["pass"] for check in case["checks"] if check["expected"] is not None):
                        break
                    case["checks"].append(run_check(prompt, expected, reasoning_budget, a.threads, active_drafts))
                    (out / "result.json").write_text(json.dumps(result, indent=2))
            def draft_score(case):
                bench = [check for check in case["checks"] if check["expected"] is None and check["reasoning_budget_tokens"] is None]
                if len(bench) != 2 or any(check["throughput_contended"] for check in bench):
                    return 0
                return min(check["response"]["timings"]["predicted_per_second"] for check in bench)
            winner = max(result["draft_sweep"], key=draft_score)
            result["selected_draft_n"] = winner["draft_n"]
            result["draft_sweep_scores"] = {str(case["draft_n"]): draft_score(case) for case in result["draft_sweep"]}
            set_active_drafts(winner["draft_n"])
        if (a.profile_cpu or a.profile_phase or a.profile_perf) and all(c["pass"] for c in result["checks"] if c["expected"] is not None):
            result["profile_responses"] = {}
            result["profile_effective_numa_threads"] = result.get("selected_numa_threads", a.threads)
            result["profile_effective_draft_n"] = result.get("selected_draft_n", initial_draft_n)
            profile_prefix = None
            if a.profile_answer_prefix:
                candidates = [c for c in result["checks"] if c["expected"] is None and
                              (c["response"]["choices"][0]["message"].get("content") or
                               c["response"]["choices"][0]["message"].get("reasoning_content"))]
                chosen = max(candidates, key=lambda c: c["response"]["timings"]["predicted_n"])
                formatted = request("apply-template", {"messages": [{"role": "user", "content": chosen["prompt"]}],
                                    "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
                message = chosen["response"]["choices"][0]["message"]
                reasoning = message.get("reasoning_content") or ""
                content = message.get("content") or ""
                body = content
                if reasoning:
                    body = ("" if formatted.rstrip().endswith("<think>") else "<think>") + reasoning
                    if content:
                        body += "</think>" + content
                text = formatted + body
                profile_prefix = request("tokenize", {"content": text, "add_special": False,
                                                       "parse_special": True})["tokens"][:-32]
                result["profile_prefix"] = {"prompt": chosen["prompt"], "tokens": profile_prefix,
                                            "token_count": len(profile_prefix), "removed_tail_tokens": 32,
                                            "includes_reasoning": bool(reasoning)}
            for mode, enabled, arm in [("phase", a.profile_phase, phase_profile_arm),
                                       ("cpu", a.profile_cpu, cpu_profile_arm)]:
                if not enabled:
                    continue
                payload = {"model": a.alias, "messages": [{"role": "user", "content": checks[-1][0]}],
                           "temperature": 0, "seed": 42, "max_tokens": 1, "cache_prompt": True,
                           "chat_template_kwargs": {"enable_thinking": False}}
                endpoint, limit_key = "v1/chat/completions", "max_tokens"
                if profile_prefix is not None:
                    endpoint, limit_key = "completion", "n_predict"
                    payload = {"prompt": profile_prefix, "temperature": 0, "seed": 42,
                               "n_predict": 1, "cache_prompt": True}
                request(endpoint, payload)
                arm.touch()
                payload[limit_key] = 16
                try:
                    result["profile_response"] = request(endpoint, payload)
                    result["profile_responses"][mode] = result["profile_response"]
                finally:
                    arm.unlink(missing_ok=True)
            if a.profile_perf:
                perf_data = out / "perf.data"
                perf_cmd = ["sudo", "-n", "perf", "record", "-F", "199", "-e", "cycles",
                            "-p", str(server.pid), "-o", str(perf_data), "--", "sleep", "6"]
                perf_payload = {"prompt": profile_prefix if profile_prefix is not None else "Explain how a refrigerator works.",
                                "temperature": 0, "seed": 42, "n_predict": 1, "cache_prompt": True}
                request("completion", perf_payload)
                with (out / "perf-record.log").open("w") as perf_log:
                    perf = subprocess.Popen(perf_cmd, stdout=perf_log, stderr=subprocess.STDOUT)
                    perf_payload["n_predict"] = 128
                    try:
                        result["perf_profile_response"] = request("completion", perf_payload)
                    finally:
                        perf_exit = perf.wait(timeout=20)
                result["perf_profile"] = {"command": perf_cmd, "record_exit": perf_exit,
                                          "effective_numa_threads": result.get("selected_numa_threads", a.threads),
                                          "note": "Separate request after throughput measurements; cycle sampling only."}
                if perf_exit == 0:
                    with (out / "perf-report.txt").open("w") as perf_report:
                        reported = subprocess.run(["sudo", "-n", "perf", "report", "--stdio", "--no-children",
                                                   "--percent-limit", "1", "-i", str(perf_data)],
                                                  stdout=perf_report, stderr=subprocess.STDOUT, timeout=60)
                    result["perf_profile"]["report_exit"] = reported.returncode
            print("Profile captured after throughput measurements", flush=True)
        if a.cache_check:
            formatted = request("apply-template", {"messages": [{"role": "user", "content":
                "Explain how a refrigerator moves heat. Give a detailed explanation in plain English."}],
                "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
            prompt_tokens = request("tokenize", {"content": formatted, "add_special": False,
                                                "parse_special": True})["tokens"]
            common = {"temperature": 0, "seed": 42, "return_tokens": True}
            prime = request("completion", dict(common, prompt=prompt_tokens, n_predict=24, cache_prompt=False))
            full = prompt_tokens + prime["tokens"]
            variants = [("repeat", prompt_tokens), ("extend", full),
                        ("trim_one", full[:-1]), ("trim_three", full[:-3])]
            result["cache_checks"] = []
            for label, tokens in variants:
                reference = request("completion", dict(common, prompt=tokens, n_predict=24, cache_prompt=False))
                request("completion", dict(common, prompt=prompt_tokens, n_predict=24, cache_prompt=False))
                cached = request("completion", dict(common, prompt=tokens, n_predict=24, cache_prompt=True))
                passed = bool(reference["tokens"]) and reference["tokens"] == cached["tokens"]
                result["cache_checks"].append({"label": label, "pass": passed, "prompt_tokens": tokens,
                                               "reference": reference, "cached": cached})
                (out / "result.json").write_text(json.dumps(result, indent=2))
                print(json.dumps({"cache_check": label, "pass": passed,
                                  "cached_tokens": cached.get("timings", {}).get("cache_n")}), flush=True)
                if not passed:
                    raise RuntimeError("cached continuation mismatch: " + label)
    except Exception as e:
        result["error"] = str(e)
        raise
    finally:
        contention = guard.stop()
        if contention is not None:
            result["contention_abort"] = contention
            result["error"] = "Benchmark cancelled because other inference became active or changed"
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        result["finished"] = time.time()
        result["server_exit"] = server.returncode
        (out / "result.json").write_text(json.dumps(result, indent=2))
