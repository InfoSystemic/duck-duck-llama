#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys
from exclusive_model_launch import acquire_exclusive_model_launch, ModelLaunchConflict

config = json.loads(Path(__file__).with_name("qwen-flash-20tps.json").read_text())
try:
    launch_lease = acquire_exclusive_model_launch(Path(__file__).resolve().parent)
except ModelLaunchConflict as error:
    raise SystemExit(f"Qwen launch refused: {error}")
env = {key: value for key, value in os.environ.items()
       if not key.startswith(("GGML_", "LLAMA_GRAPH_PHASE", "LLAMA_MTP_DRAFT_N_FILE", "OMP_", "GOMP_"))}
env.update(config["runtime_env"])
os.execvpe(config["command"][0], config["command"] + sys.argv[1:], env)
