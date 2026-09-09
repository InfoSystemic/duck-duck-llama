# Qwen3.8-Flash-Next MTP candidate on SR950

Pinned candidate:

- Sidecar: `drluoto/Qwen3.8-Flash-Next-MTP-GGUF`
- Revision: `67de7592b670ef454a903574d5e2aa6c8e1d6b46`
- File: `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` (4,142,897,248 bytes)
- SHA-256: `b9880220df29fc224bbce408c867cd5d9c021263b754033ea624b669e374f4ec`
- Runtime bundle: `/home/kwebb/.local/opt/llama.cpp-numa-mtp-b249-3173a56/bin`

This third-party sidecar is an opt-in performance candidate. It is not part of
the base Qwen quant and is not promoted until the deterministic-output,
speculative-acceptance, and quality suites pass on this host.

After download approval, fetch and verify the sidecar:

```bash
./download-qwen38-mtp.sh
```

Benchmark it with the MTP-capable candidate runtime while retaining the
existing base launcher and model placement:

```bash
QWEN38_BUNDLE_BIN=/home/kwebb/.local/opt/llama.cpp-numa-mtp-b249-3173a56/bin \
QWEN38_MTP_MODEL=/models/gguf/Qwen3.8-Flash-Next/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf \
  /home/kwebb/.local/bin/launch-qwen38-flash-next.sh 5820
```

The launcher defaults to `draft-mtp`, three draft tokens, and probability
minimum 0.6 when `QWEN38_MTP_MODEL` is set. The current non-speculative
runtime remains the default when it is unset.
