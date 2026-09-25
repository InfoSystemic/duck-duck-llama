#!/usr/bin/env python3
"""vision-real-probe.py [port] -- real-image checks with verifiable answers (OCR, counting, colour).

The earlier probes used flat synthetic squares, which hid the F16 overflow (every patch identical, no outlier tokens).
These are real images already on the box, each with a question whose answer can be checked by string match.
"""
import base64, json, re, sys, time, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18190

CASES = [
    ("/usr/share/cups/doc-root/images/smiley.jpg",
     "What is drawn in this image, and what colours are the face and its eyes? Answer in one sentence.",
     [r"smil", r"blue", r"yellow"]),
    ("/usr/share/desktop-base/joy-theme/login/sddm-preview.jpg",
     "What time and what date are shown on this screen? Answer exactly as displayed.",
     [r"00[:.]27", r"16", r"novembre|november", r"2016"]),
    ("/usr/share/netdata/web/v3/static/img/referral/logs.jpg",
     "In this dashboard, how many 'critical' events and how many 'error' events are shown in the summary under the chart? "
     "Also, what value appears in the _HOSTNAME column? Answer briefly.",
     [r"\b5\b", r"\b10\b", r"packages"]),
    ("/usr/share/netdata/web/v3/static/img/referral/logs.jpg",
     "Quote the error message text that repeats in the MESSAGE column.",
     [r"kex_exchange_identification", r"closed by remote host"]),
]


def ask(path, question, think=False, n=160):
    mime = "image/png" if path.endswith(".png") else "image/jpeg"
    data = base64.b64encode(open(path, "rb").read()).decode()
    body = {"messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
                {"type": "text", "text": question}]}],
            "max_tokens": n, "temperature": 0, "chat_template_kwargs": {"enable_thinking": think}}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    return (r["choices"][0]["message"].get("content") or "").strip(), time.time() - t0, r.get("usage", {})


passed = 0
for path, q, pats in CASES:
    a, dt, usage = ask(path, q)
    ok = all(re.search(p, a, re.I) for p in pats)
    passed += ok
    print(f"{'PASS' if ok else 'FAIL'}  {path.split('/')[-1]:18s} {dt:6.1f}s  prompt_tokens={usage.get('prompt_tokens')}  "
          f"{a.replace(chr(10), ' ')[:230]!r}", flush=True)
print(f"\nvision real-image checks: {passed}/{len(CASES)}")
