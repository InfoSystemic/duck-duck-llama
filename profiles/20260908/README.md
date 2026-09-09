# September 8 SR950 profiles

These parameterized profiles accompany the [engineering source snapshot](../../engineering/2026-09-08/README.md). They describe measured configurations on a four-socket Xeon Gold 6242 server. They do not download weights, build an engine, manage services, or reserve memory.

| Profile | Required source | Status |
| --- | --- | --- |
| [qwen-flash-next-q6-mtp4.json](qwen-flash-next-q6-mtp4.json) | Qwen goal patch plus selected Q6 overlay | Selected UD-Q6_K_XL target, Q8 draft, MTP4, 15 workers/socket |
| [glm-flash-q8-raw.json](glm-flash-q8-raw.json) | GLM Flash goal patch plus Q8 raw overlay | Retained Q8 raw experimental stack; no MTP in this profile |
| [glm-full-q4-mixed-mtp2.json](glm-full-q4-mixed-mtp2.json) | Current Full/SR950 patch | Q4-based mixed runtime with hybrid draft; MTP default 2 and maximum 32 |

Render a command from the repository root:

```bash
python3 tools/render_profile.py profiles/20260908/qwen-flash-next-q6-mtp4.json \
  --set SERVER=/path/to/llama.cpp/build/bin/llama-server \
  --set LIBRARY_PATH=/path/to/llama.cpp/build/bin \
  --set MODEL=/models/Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf \
  --set DRAFT=/models/mtp-Qwen3.8-Flash-Next-Q8_0.gguf \
  --set PORT=8080
```

The renderer prints a shell-quoted command and does not execute it. Each profile lists its required parameters. Full additionally requires `CHAT_TEMPLATE`; the recorded template is archived [here](../../engineering/2026-09-08/archive/serving/glm-sr950/chat-template-glm-5.3-llamacpp.jinja). `HOST` defaults to `127.0.0.1`. Point `LIBRARY_PATH` to a complete matching runtime; do not accidentally mix libraries from different source lines.

Load one large model at a time on this server. Full is not required to remain resident alongside either Flash model. The historical [exclusive model-load guard](../../engineering/2026-09-08/archive/serving/fleet-0903/exclusive_model_launch.py) and its fixtures document the overlap checks used during tuning. The rendered command itself is not a service manager or a concurrency guard.

Use fresh benchmark requests with `cache_prompt=false` and verify `cache_n=0`. Qwen's explicit prompt-cache extension has an unresolved parity failure. Qwen also ignores the request-level speculative settings in this source version: change MTP depth through launch flags and reload, then verify actual draft counts. Full's patched server has a different request-control contract; its MTP default and maximum are distinct.

The profiles retain recorded compute settings while parameterizing paths, bind address and port. Ephemeral thread-control files are replaced by the recorded fixed 15-worker value. GLM Flash's older lower-quant launcher remains historical; its Q8 source and weights have not been broadly certified as near-lossless. The saved Flash MTP2 rate belongs to an earlier stack and is not a benchmark of this raw profile with MTP added.
