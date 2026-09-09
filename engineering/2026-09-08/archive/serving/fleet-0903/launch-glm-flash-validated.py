#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

config = json.loads(Path(__file__).with_name("glm-flash-validated.json").read_text())
env = {key: value for key, value in os.environ.items()
       if not key.startswith(("GGML_", "LLAMA_GRAPH_PHASE", "LLAMA_MTP_DRAFT_N_FILE", "OMP_", "GOMP_"))}
env.update(config["runtime_env"])
os.execvpe(config["command"][0], config["command"] + sys.argv[1:], env)
