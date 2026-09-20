# GLM Flash Codex / Paseo integration audit, 2026-09-19

User objective: "engineer and tune GLM-5.3-Flash to work as well as possible on this machine (SR950 ~400GB/s), and increase how many tok/s it gets when ran from Codex CLI inside of Paseo".

## Actual request path

- Standalone local Paseo daemon: home /home/user/.paseo, host smeagol, listen 127.0.0.1:6767, daemon/CLI version 0.8.0. Provider diagnostic reports Codex CLI 0.154.0 and codex-local ready.
- /home/user/.paseo/config.json -> agents.providers.codex-local -> /home/user/.local/libexec/paseo/codex-smeagol.
- Wrapper sources /home/user/.local/libexec/paseo/smeagol-model-route.sh, rewrites CLI model to GLM-5.3-Flash, and overrides model_providers.smeagol.base_url.
- Wrapper sets CODEX_HOME=/home/user/.codex-glm. That config selects wire_api=responses, HTTP SSE, no websockets. Local requests therefore go directly to /v1/responses on the chosen llama-server; Flash does not traverse the legacy :18092 proxy.
- PID3963426 for the actual Paseo baseline confirmed --model GLM-5.3-Flash, base_url=http://127.0.0.1:18131/v1, model_context_window=32768, model_auto_compact_token_limit=30720, model_reasoning_effort=low, model_reasoning_summary=none, Flash catalog JSON, and app-server --enable goals. CPU affinity 0-127; memory nodes0-3.
- Paseo thread/start and turn/start separately send lowercase glm-5.3-flash. Their values override the launch-time uppercase selection.

## Baseline rollout evidence

Source: /home/user/.codex-glm/sessions/2026/09/19/rollout-2026-09-19T13-09-16-01a0bb12-a427-7413-9e06-deec546c5843.jsonl

- session_meta.payload.base_instructions.text: 20,751 characters (stock fallback), rather than 504 characters configured in the Flash catalog.
- turn_context.payload.model: glm-5.3-flash.
- turn_context.payload.collaboration_mode.settings.reasoning_effort: null.
- turn_context.payload.summary: auto.
- event_msg/token_count.payload.info.last_token_usage: 8,752 input tokens, zero cached tokens, 312 output tokens.
- event_msg/task_complete.payload: duration_ms=176176; time_to_first_token_ms=149616.
- Developer text sizes: skills3048 chars; permission instructions4065 chars; collaboration1329 chars. These are active functionality/policy, not redundant copies to remove blindly.
- No CODEX_HOME AGENTS.md, AGENTS.override.md, or instructions.md was present. Base config has no MCP servers. Captured request tool metadata was summarized without storing full prompts.

## Routing regression and prepared repair

At read-only audit time, /health on18094 refused in33.8ms;18131 returned200 in1.3ms. Existing code still preferred18094 and only fell back to18131. Thus an experimental server on18094 could silently capture production requests.

Prepared patch changes only Flash's implicit route to18131; explicit wrapper CLI base_url overrides remain later in argv and take precedence. It does not restart services. When production is unavailable it marks18131 down instead of substituting an experiment.

- Backup: smeagol-model-route.before.sh
- Candidate: smeagol-model-route.proposed.sh
- Patch: flash-production-routing.patch
- Manifest: flash-production-routing-manifest.json
- Validation: test_flash_production_routing.py; flash-production-routing-test.json
- 24 Flash cases and36 non-Flash cases pass; both files pass bash -n. No network/inference in mocked tests.
- Exact rollback command is recorded in the manifest.
- Initial live mutation was rejected by automatic approval review for production traffic risk/authorization. Root subsequently reported applying the exact reviewed patch after reviewing the concrete tests. Consult manifest/live source for current deployment state.

## Catalog alias repair

Original catalog has only slug GLM-5.3-Flash. Lowercase Paseo model misses it. Real app-server request captures against a loopback recorder prove this; it is not inferred merely from spelling.

Candidate adds lowercase glm-5.3-flash with identical capability metadata and sets default_reasoning_level=low for both aliases, matching the established wrapper intent. No Paseo adapter patch is bundled. No tool/skill disabling is introduced.

- Backup: glm-5.3-flash.catalog.before.json
- Candidate: glm-5.3-flash.catalog.proposed.json
- Patch/manifest: flash-catalog-alias.patch; flash-catalog-alias-manifest.json
- Offline app-server capture runner: test_flash_catalog_capture.py
- Results: flash-catalog-capture-results.json

Capture matrix:
1. Before + uppercase: instructions504 chars; reasoning.high; intended11 tool names including apply_patch.
2. Before + lowercase: instructions20751 chars; reasoning{};10 tool names, missing apply_patch.
3. Candidate + uppercase: instructions504 chars; reasoning.low; intended11 tool names.
4. Candidate + lowercase: instructions504 chars; reasoning.low; intended11 tool names.
5. Candidate + lowercase + explicit collaboration effort low: same as4.

All candidate aliases retain original uppercase tool schemas exactly after excluding dynamic description text (which now advertises an extra model alias). Instructions SHA256 matches original catalog base instructions. The loopback stub returned controlled HTTP400 errors and performed zero inference.

Catalog changes require a fresh app-server/thread for a clean before/after timing comparison: running processes and existing rollouts can retain earlier catalog/base instructions.

## Separate reasoning-default observation

Installed adapter:
 /home/user/.local/share/paseo-cli/lib/node_modules/@getpaseo/cli/node_modules/@getpaseo/server/dist/server/server/agent/providers/codex-app-server-agent.js

- loadCollaborationModes/resolveCollaborationMode around2633/2707 cache collaboration settings.
- ensureThread around3914 later resolves model/default thinking and assigns config.thinkingOptionId without refreshing those cached settings.
- buildTurnStartParams around2994 sends effort low but also cached collaboration settings that omit reasoning_effort.
- Baseline daemon log records effort low while the rollout stores collaboration reasoning_effort null.
- Offline HTTP capture reproduces that omitted collaboration effort uses model catalog default (high in original uppercase; low in candidate).
- setThinkingOption at3357 refreshes cached collaboration mode. Explicit low at agent creation or via update therefore supplies explicit collaboration reasoning.
- Adapter remains unchanged; this is a distinct upstream ordering issue.

## Context and throughput interpretation

The32K cap is in codex-smeagol lines32-43. Its comment explicitly dates it to2026-09-15 measurements of3.4 decode tok/s and~5 prefill tok/s, anticipating long prefill delays. It is a latency mitigation from an older build, not the model's native context capacity. Route/catalog advertise1,048,576. Do not change it solely from theoretical bandwidth; verify real longer prompts first.

Paseo's provider inspection returns only plan_mode, no fast_mode for codex-local. There are no named launch profiles. Adding a fast service tier is not an established local-server optimization.

The legacy /usr/local/bin/codex-glm wrapper differs materially: it does not source the routing table and defaults its selected-model branch to DeepSeek, leaving the base config endpoint. Reproducible Flash tests must use codex-smeagol or the configured codex-local provider.

## Reproduction

Offline safety gate:
 python /home/user/sr950-strategy/codex-bench-0919/test_flash_catalog_capture.py

The runner is importable; run_case(port,catalog,rpc_model,collaboration_effort) launches isolated app-server processes using a temporary CODEX_HOME and explicit stub endpoint. It currently expects the recorder to supply a captured request; root may extend it separately for real timing.

For real timing, use a fresh app-server launched through codex-smeagol in the Paseo test workspace and the same lowercase thread/start and turn/start payloads. Record start, first text delta, turn completion, token-usage events, server prompt/decode timings and rollout path. A warm follow-up is necessary to distinguish prompt cache effects from decode improvement. Keep tools/capabilities enabled and test a real shell/edit round trip separately.

Official source references consulted:
- https://learn.chatgpt.com/docs/config-file/config-advanced
- https://learn.chatgpt.com/docs/app-server
Current Paseo Markdown docs were requested through browser and direct read but access failed; local0.8.0 CLI help, provider diagnostics, and installed adapter code were authoritative here.
