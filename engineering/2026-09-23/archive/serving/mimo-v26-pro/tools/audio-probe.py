#!/usr/bin/env python3
"""audio-probe.py [port] [wav] -- does MiMo's audio input work end to end?

Default clip: the first 9.5 s of Open Speech Repository OSR_us_000_0010 (Harvard sentences, public domain):
"The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background. It's easy to tell the depth
of a well." PASS = the transcript contains most of the content words.
"""
import base64, json, re, sys, time, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18190
WAV = sys.argv[2] if len(sys.argv) > 2 else "/tmp/mimo-vis/audio/osr10-9s.wav"
WORDS = ["birch", "canoe", "slid", "smooth", "planks", "glue", "sheet", "dark", "blue", "background", "depth", "well"]

body = {"messages": [{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": base64.b64encode(open(WAV, "rb").read()).decode(), "format": "wav"}},
            {"type": "text", "text": "Transcribe this audio exactly."}]}],
        "max_tokens": 120, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
t0 = time.time()
r = json.load(urllib.request.urlopen(req, timeout=1800))
text = (r["choices"][0]["message"].get("content") or "").strip()
hits = [w for w in WORDS if re.search(r"\b" + w, text, re.I)]
ok = len(hits) >= 0.75 * len(WORDS)
print(f"{'PASS' if ok else 'FAIL'}  audio {time.time() - t0:.1f}s  prompt_tokens={r.get('usage', {}).get('prompt_tokens')}  "
      f"words {len(hits)}/{len(WORDS)}  {text.replace(chr(10), ' ')[:300]!r}")
