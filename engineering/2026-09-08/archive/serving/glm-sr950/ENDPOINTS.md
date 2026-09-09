# GLM-5.3 Full — serving endpoints (2026-09-02, v5b stack)

Model: GLM-5.3 Full (non-Flash) UD-Q4_K_XL, 32K context, on smeagol (4x Xeon Gold 6242, no GPU).
Engine: `engines/llama.cpp-sr950-glm/build-dev2/bin/llama-server`
Profile: `serving/glm-sr950/model.glm53-q4-fast-v5b-ngram.env` (v5b fast stack **+ the
`ngram-mod,draft-mtp` composite**, verified 2026-09-03).
Service: `systemctl --user start glm53-sr950` (enabled, linger on; stays "activating" until
health passes, so use `--no-block` if you do not want to wait ~21 min).

## Measured (isolated host, 2026-09-02)

| workload | tok/s |
|---|---:|
| agentic / file-edit replay (n=18, p=0.75), 3/3 exact checks | **15.23 aggregate** (per workload 15.27 / 15.48 / 14.69) |
| agentic replay (n=12, p=0.6), 3/3 exact | 14.72 aggregate |
| general chat with MTP n=2 | 9.36 |
| raw decode, no speculation | 7.51 |
| prompt ingest | ~35 |

The `ngram-mod` composite is what lifts agentic work from 13.4 to 15.2: it drafts from
repetition already in the context, which is what file-edit traffic is made of. It was
previously believed fatal on GLM-5.3 (HTTP 500 on every request); that is fixed in the
current engine. `n_match=8` (not the old default 24) is what makes it pay. Cold prose did
not regress (9.36 vs 9.55, inside run-to-run spread).

## Endpoints

| use | URL | model id | notes |
|---|---|---|---|
| local, general | `http://127.0.0.1:18091/v1` | `glm-sr950` | llama-server directly; OpenAI-compatible |
| local, agentic | `http://127.0.0.1:18092/v1` | `glm-sr950-agentic` | proxy injects `speculative.n_max=18, p_min=0.75` |
| **LAN, agentic** | `http://10.0.0.1:18093/v1` | `glm-sr950-agentic` | same injection; ufw allows 10.0.0.0/24 only |
| Tailscale | `http://198.51.100.1:18093/v1` | `glm-sr950-agentic` | tailscale0 is allowed by ufw already |

No API key is required (any value is accepted). The endpoints are unauthenticated, so the
firewall scope (LAN subnet / Tailscale) is the only access control.

`glm-sr950` on any endpoint uses the general default (n=2). The `-agentic` id only differs
because the proxy adds the speculative parameters; llama-server ignores the alias by itself,
so a client hitting 18091 directly must send them in the request body:
`{"speculative.n_max": 18, "speculative.p_min": 0.75}`.

## Example for a LAN user

```bash
curl http://10.0.0.1:18093/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-sr950-agentic","messages":[{"role":"user","content":"Hello"}],"max_tokens":200}'
```

OpenAI SDK: `base_url="http://10.0.0.1:18093/v1"`, `api_key="local"`, `model="glm-sr950-agentic"`.

## Services

- llama-server: started from `serving/glm-sr950/launch-glm-sr950.sh` with
  `GLM_SR950_CONFIG=.../model.glm53-q4-fast-v5b.env` (binds 0.0.0.0:18091). Load takes ~21 min.
  After `/health` is ok, run `host-setup/pin_main.sh` to move the main/HTTP threads to the spare cores.
- LAN proxy: `glm-agentic-proxy-lan.service` (systemd --user, enabled, linger on) ->
  `~/.local/libexec/paseo/glm-agentic-proxy.py` on 0.0.0.0:18093.
- Paseo proxy on 127.0.0.1:18092 is started separately by Paseo's own tooling.
- Background isolation: `host-setup/isolate-background.sh` (docker/containerd/netdata/mssql on
  cores 15,31,47,63; keeps them off the 60 worker cores). Worth ~4% and removes stragglers.

## Do not enable

`GGML_GLM_DSA_DENSE=1` and `GGML_CPU_X16_DUAL=1` (the "v6" additions) produce fluent nonsense
on the full model (`17*23` -> "0.5, 0.5, 0.5", replay 0/3) even though they pass a first-token
oracle. See `../../GLM53-MAX-TOKS-PLAN.md`.
