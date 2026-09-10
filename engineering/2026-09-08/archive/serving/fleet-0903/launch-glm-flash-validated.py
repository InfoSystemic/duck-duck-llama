#!/usr/bin/env python3
import json
import hashlib
import fcntl
import os
from pathlib import Path
import sys

selected = Path(__file__).with_name("glm-flash-selected.json")
config = json.loads((selected if selected.exists() else Path(__file__).with_name("glm-flash-validated.json")).read_text())
if selected.exists():
    from glm_flash_q8_trial import memory_status, node_memory_status
    from qwen_split_trial import inference_snapshot
    lifecycle = selected.parent / "results/qwen-q6-trial-0907/lifecycle.lock"
    launch_lock = lifecycle.open("a")
    fcntl.flock(launch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if inference_snapshot():
        raise RuntimeError("Another model is loaded. Switch the active model before loading Flash.")
    if memory_status()["MemAvailable"] <= 260000000000 or any(
        node["estimated_available"] <= 60000000000 for node in node_memory_status().values()
    ):
        raise RuntimeError("Insufficient available RAM to load the selected Flash model with its reserve.")
    for record in config["model_records"]:
        model = Path(record["path"])
        if not model.exists():
            raise RuntimeError("Selected Flash weights are missing. Restore the staged model files before launching.")
        info = model.stat()
        if (info.st_size, info.st_ino, info.st_mtime_ns) != (record["bytes"], record["inode"], record["mtime_ns"]):
            raise RuntimeError("Selected Flash model identity changed; verify the model before launching.")
    for path, expected in config["runtime_sha256"].items():
        with Path(path).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise RuntimeError("Selected Flash runtime changed; validate it before launching.")
    os.sched_setaffinity(0, config["affinity"])
env = {key: value for key, value in os.environ.items()
       if not key.startswith(("GGML_", "LLAMA_GRAPH_PHASE", "LLAMA_MTP_", "OMP_", "GOMP_"))
       and key != "LD_LIBRARY_PATH"}
env.update(config["runtime_env"])
os.execvpe(config["command"][0], config["command"] + sys.argv[1:], env)
