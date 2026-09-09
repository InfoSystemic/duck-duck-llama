#!/usr/bin/env python3
import json
from pathlib import Path
import subprocess
import time

base = Path(__file__).resolve().parent
result = base / 'results/glm5n-goal-ngram4-mtp2-1024-t15/result.json'
while True:
    try:
        if 'server_exit' in json.loads(result.read_text()):
            break
    except (FileNotFoundError, ValueError):
        pass
    time.sleep(3)
raise SystemExit(subprocess.call(['python3', str(base / 'preserve-glm-q8-mtp.py')]))
