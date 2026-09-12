#!/usr/bin/env python3
"""Run a fixed baseline or candidate recipe in the foreground."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def replace_value(command, option, value):
    index = command.index(option)
    command[index + 1] = str(value)


def readable(path):
    return path.is_file() and os.access(path, os.R_OK)


def main():
    config = json.loads((HERE / "profile.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=config["arms"], default="tuned")
    parser.add_argument("--port", type=int, default=18131)
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--tensor-split", help="Four nonnegative ratios; requires the split-candidate arm")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be between 1024 and 65535")
    if args.parallel < 1 or args.ctx_size < 512 * args.parallel:
        parser.error("--ctx-size must provide at least 512 tokens per parallel slot")

    command = list(config["command"])
    for option, value in (("--port", args.port), ("--ctx-size", args.ctx_size),
                          ("--parallel", args.parallel)):
        replace_value(command, option, value)
    recipe = config["arms"][args.arm]
    if args.tensor_split is not None:
        if args.arm != "split-candidate":
            parser.error("Unequal splits require the reviewed split-candidate library")
        try:
            ratios = [float(value) for value in args.tensor_split.split(",")]
        except ValueError:
            parser.error("--tensor-split must contain four finite nonnegative numbers")
        if len(ratios) != 4 or any(not math.isfinite(value) or value < 0 for value in ratios) or sum(ratios) <= 0:
            parser.error("--tensor-split must contain four finite nonnegative ratios with positive sum")
        replace_value(command, "--tensor-split", args.tensor_split)
    replace_value(command, "--spec-type", recipe["spec_type"])
    # No caller environment enters the inference process.
    environment = {
        "HOME": "/home/kwebb", "USER": "kwebb", "LOGNAME": "kwebb",
        "PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8", "TMPDIR": "/tmp",
        **config["runtime_env"],
        **recipe.get("runtime_env", {}),
    }
    if "library_directory" in recipe:
        environment["LD_LIBRARY_PATH"] = recipe["library_directory"] + ":" + environment["LD_LIBRARY_PATH"]
    if recipe["unary"]:
        environment["GGML_CPU_PARALLEL_UNARY"] = "4096"
        environment["LD_LIBRARY_PATH"] = str(HERE / "build") + ":" + environment["LD_LIBRARY_PATH"]

    errors = []
    if "build_verification" in recipe:
        try:
            verified = json.loads(Path(recipe["build_verification"]).read_text())
            if not verified["passed"] or not verified["baseline_relink_byte_identical"]:
                raise ValueError("candidate build did not pass")
            candidate_library = Path(recipe["library_directory"]) / "libllama.so.0"
            expected = {**verified["source_and_link_inputs_sha256"],
                        str(candidate_library): verified["candidate_library_sha256"]}
            for path, digest in expected.items():
                if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                    raise ValueError(f"candidate dependency changed: {path}")
        except (OSError, ValueError, KeyError) as error:
            errors.append(f"candidate build preflight failed: {error}")
    required = [Path(command[0]), Path("/usr/bin/taskset")]
    for option in ("--model", "--spec-draft-model", "--chat-template-file"):
        model = Path(command[command.index(option) + 1])
        required.append(model)
        match = re.fullmatch(r"(.*)-[0-9]{5}-of-([0-9]{5})\.gguf", model.name)
        if match:
            required.extend(model.with_name(f"{match[1]}-{i:05d}-of-{match[2]}.gguf")
                            for i in range(1, int(match[2]) + 1))
    for directory in environment["LD_LIBRARY_PATH"].split(":"):
        if not Path(directory).is_dir():
            errors.append(f"missing library directory: {directory}")
    if recipe["unary"]:
        required.append(HERE / "build" / "libggml-cpu.so.0")
    for path in dict.fromkeys(required):
        if not readable(path):
            errors.append(f"missing or unreadable file: {path}")
    if readable(Path(command[0])) and not os.access(command[0], os.X_OK):
        errors.append(f"binary is not executable: {command[0]}")

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", args.port))
    except OSError as error:
        errors.append(f"port {args.port} is not idle: {error}")

    loader = ""
    if not errors:
        try:
            result = subprocess.run(["/usr/bin/ldd", command[0]], env=environment,
                                    capture_output=True, text=True, timeout=15)
            loader = result.stdout + result.stderr
            if result.returncode or "not found" in loader:
                errors.append(f"library resolution failed: {loader.strip()}")
            cpu_line = next((line for line in loader.splitlines()
                             if "libggml-cpu.so.0 =>" in line), "")
            if recipe["unary"] and str(HERE / "build" / "libggml-cpu.so.0") not in cpu_line:
                errors.append("loader did not select the candidate CPU library")
            if "library_directory" in recipe:
                llama_line = next((line for line in loader.splitlines() if "libllama.so.0 =>" in line), "")
                if str(Path(recipe["library_directory"]) / "libllama.so.0") not in llama_line:
                    errors.append("loader did not select the split-candidate library")
        except (OSError, subprocess.SubprocessError) as error:
            errors.append(f"library preflight failed: {error}")

    invocation = ["/usr/bin/taskset", "-c", "0-127", *command]
    if args.dry_run:
        print(json.dumps({"model": config["model"], "arm": args.arm,
                          "command": invocation, "runtime_env": environment,
                          "library_resolution": loader.splitlines(),
                          "preflight_errors": errors}, indent=2))
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    if args.dry_run:
        return 0
    print(f"Starting {config['model']} arm={args.arm} port={args.port}", flush=True)
    os.execve(invocation[0], invocation, environment)


if __name__ == "__main__":
    raise SystemExit(main())
