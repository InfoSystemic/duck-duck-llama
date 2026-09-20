# GLM-5.3-Flash on :18131 — restored for Paseo (2026-09-18)

**Status: LIVE, verified end-to-end through Paseo, and enabled at boot.**

Paseo's default model in every local provider (`claude-local`, `claude-local-full`,
`codex-local`, `copilot-local`, `kimi-local`) is `glm-5.3-flash`, which routes to
`http://127.0.0.1:18131`. That endpoint was dead. It is now serving.

---

## 1 · What was actually wrong

Not one fault — three, stacked.

1. **No llama-server was running at all.** `:18131`, `:18094`, `:18091`, `:18083`,
   `:18172` were all down. The two running agentic proxies (`:18092`, `:18093`) both
   pointed at the dead `:18091` and returned `[Errno 111] Connection refused`.

2. **The RAM blocker.** 411 GB of the 756 GB box was pinned in tmpfs: the 186 GB
   Flash model payload **plus** a 180 GB requant artifact at
   `/dev/shm/glm53-fast/GLM-5.3-Flash-fast-Q4K.gguf`. That left the four NUMA nodes
   with roughly 74 / 97 / 119 / **47** GB available, while the production launcher's
   `--split-mode tensor --tensor-split 1,1,1,1` needs **~73 GB free per node**.
   The load OOM-killed after 153 s — the same wall `SR950-SESSION-LOG-20260916.md`
   §6 hit five times.

3. **A boot race.** `glm53-sr950.service` (GLM-5.3 Full) was **enabled**. It needs
   ~490 GB and can never coexist with Flash, so every boot started a fight the box
   could not win.

The requant was freed with owner approval (rebuildable in ~10 min; the exact
`llama-quantize` recipe is preserved in
`serving/glm-sr950/REQUANT-FREED-20260918.md`). MemAvailable went **345 → 525 GB**
and every node to **117-136 GB** free. The load then succeeded in **247 s**.

---

## 2 · Current state

| item | value |
|---|---|
| endpoint | `http://127.0.0.1:18131` (loopback only) |
| model id / aliases | `GLM-5.3-Flash`, `glm-5.3-flash`, `glm-flash-goal`, `glm-flash-q4` |
| context | 1,048,576 native, 2 unified slots |
| params / quant | 320.76 B, Q4_K - Medium, 186 GiB |
| RSS | ~277 GiB |
| unit | `glm53-flash-production.service` — **enabled + active** |
| launcher | `fleet-0912-ctx/launch-glm-flash-native.sh` (the validated recipe) |

### Measured 2026-09-18 (this restore, no other model running)

| probe | result |
|---|---|
| decode, 2 × 512 tokens | 12.77 / 13.77 → **13.27 tok/s** mean |
| prefill | 66.5 prompt tok/s (5,354-token prompt) |
| correctness | `17*23` → **391** correct |
| long context | correctly answered over a 5.3K-token prompt |
| tool calling | valid `tool_calls`, well-formed JSON arguments |
| Anthropic `/v1/messages` | verified (the wire `claude-smeagol` uses) |
| **Paseo end-to-end** | `claude-smeagol --model glm-5.3-flash` → `FLASH-PASEO-OK`, exit 0, 35 s |

> **Correcting the record:** `smeagol-model-route.sh` (2026-09-15) states Flash was
> "measured directly, four ways, all ~3 tok/s" and uses that to steer long context to
> DeepSeek-V4.1-Flash instead. On **this** production recipe it is **13.27 tok/s** —
> 4.4× that figure. The 3 tok/s numbers came from a different engine/config. The
> `smeagol_advise_model()` crossover table in that file is worth re-deriving.

---

## 3 · The OOM incident (read this before benchmarking)

After the server was up, it was OOM-killed **three times** and restarted in a loop.
Root cause was **not** this service:

- A concurrent agent session was bringing up the *same* GLM-5.3-Flash model from
  `/tmp/bench-glm53-256k.sh`, which pins to **NUMA node 3**:
  `numactl --cpunodebind=3 --membind=3 ... --ctx-size 262144`.
  A 186 GB model plus 256K KV bound to one 193 GB node, while production already
  held node 3 → node-3 allocation failure → global OOM.
- Victim selection then picked **our** server, because systemd gives user services
  `oom_score_adj=200` — a *preference* for being killed.

Kernel evidence:

```
llama-bench invoked oom-killer: gfp_mask=0x140dca(...)
oom-kill:constraint=CONSTRAINT_MEMORY_POLICY,nodemask=3,...,task=llama-server
Out of memory: Killed process 3961932 (llama-server) anon-rss:290664512kB oom_score_adj:200
```

`MemoryMax=380G` did what it was meant to (the kill stayed inside the unit's cgroup
rather than taking `sqlservr` or `Xorg` host-wide), but it does **not** stop the
kernel choosing this process as victim.

**Verified: a `systemd --user` unit cannot lower `oom_score_adj`.** A probe unit
requesting `OOMScoreAdjust=-500` came up at `100`; the user manager lacks
`CAP_SYS_RESOURCE`. Protecting the server would need either a **system** unit
(`User=user`, `OOMScoreAdjust=-500`) or an `ExecStartPost` that writes
`/proc/<pid>/oom_score_adj` via sudo. **Neither was applied** — it changes the
service model, and this box already has a documented protocol for the same problem
(`SR950-SESSION-LOG-20260916.md` §1: SIGSTOP the campaign *and* stop production
before measuring). If you benchmark GLM-5.3-Flash again, stop
`glm53-flash-production.service` first.

---

## 4 · Operating it

```bash
# status
systemctl --user status glm53-flash-production.service
curl -s http://127.0.0.1:18131/health

# the usual fleet switcher still works (SIGTERM exits cleanly, no restart fight)
~/InfoSystemic/AI-Server/serving/fleet-0912-ctx/profile.sh status

# stop / start the production Flash server
systemctl --user stop  glm53-flash-production.service
systemctl --user start glm53-flash-production.service   # ~120-250 s to health

# switch to a different fleet profile — stop the unit FIRST, or it will race
systemctl --user stop glm53-flash-production.service
~/InfoSystemic/AI-Server/serving/fleet-0912-ctx/profile.sh dsv41-tuned
```

`Restart=on-failure` (not `always`) is deliberate: `stop-by-port.sh` and
`profile.sh` kill by port with SIGTERM, which exits 0 and therefore does **not**
trigger a restart. Real crashes still do.

---

## 5 · Changes made

| change | detail |
|---|---|
| freed 180 GB | `rm /dev/shm/glm53-fast/GLM-5.3-Flash-fast-Q4K.gguf`; recipe in `serving/glm-sr950/REQUANT-FREED-20260918.md` |
| new unit | `~/.config/systemd/user/glm53-flash-production.service` (enabled, `MemoryMax=380G`, `Conflicts=glm53-sr950.service`) |
| disabled | `glm53-sr950.service` boot start — it can never load alongside Flash |
| Paseo metadata | corrected the stale `13-15 tok/s (19-22 two-stream)` / "co-resident with Qwen" label on `glm-5.3-flash` in all 5 providers; backup at `~/.paseo/config.json.bak-20260918-flash-live` |
| journal cap | `/etc/systemd/journald.conf.d/99-size-cap.conf` (`SystemMaxUse=512M`, `SystemKeepFree=2G`) — see §6 |

To revert everything except the freed artifact:

```bash
systemctl --user disable --now glm53-flash-production.service
systemctl --user enable glm53-sr950.service
sudo rm /etc/systemd/journald.conf.d/99-size-cap.conf && sudo systemctl restart systemd-journald
cp ~/.paseo/config.json.bak-20260918-flash-live ~/.paseo/config.json
```

The Paseo daemon reads provider metadata at startup; **restart Paseo (or the daemon)**
to pick up the corrected labels. Routing itself needed no change and was already live.

---

## 6 · The disk is the top remaining risk

**The root filesystem is at 100 % — 638 MB free of 935 GB.** This is pre-existing,
and it is the most dangerous thing on the box.

It went from ~2.5 GB free to **0 bytes** during this work, because each verbosity-2
model load writes roughly 100 MB into the journal and the restarts in §3 multiplied
that. `journalctl --vacuum-size=200M` freed 801 MB, and the new journald cap stops
it recurring — but the underlying pressure is untouched.

Reclaimable without touching anyone's data:

| candidate | size |
|---|---|
| `/tmp/serving.tgz` | 3.1 GB |
| `/tmp/perf-prefill.data` | 2.6 GB |
| `/tmp/glm53-ngram-256k.log` | 2.4 GB |
| docker images (1.07 GB reclaimable), 12 local volumes | ~1.4 GB |
| `/dev/shm` residue (7.2 GB MTP-hybrid GGUF + build dirs) | 9.4 GB |

A full root filesystem will break model staging, logs, and any service that needs
scratch space. **This deserves attention before anything else on this box.**

---

## 7 · Still outstanding (not touched)

- **`glm53-flash-q4.service`** (port `:18094`, the SR950 "20t" profile) is present but
  disabled and never started. Note the router tries **18094 first, then 18131** — if
  someone ever starts it, routing silently moves. It needs ~300 GB anon and so cannot
  coexist with the current server.
- **GLM-5.3 Full, Qwen3.8-Flash-Next, DeepSeek-V4.1-Flash are all still down**, and
  `:18092` / `:18093` still proxy to the dead `:18091`. Those proxy units are enabled,
  so they keep returning 502 until GLM-5.3 Full is brought up.
- The `claude-code:unrecognized_model` warning in the agent stderr is cosmetic —
  Claude Code does not know the name `GLM-5.3-Flash`; `CLAUDE_CODE_MAX_CONTEXT_TOKENS`
  is set explicitly by the wrapper to compensate.