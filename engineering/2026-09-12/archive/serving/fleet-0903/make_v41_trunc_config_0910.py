#!/usr/bin/env python3
"""Rewrite a DeepSeek-V4.1-Flash config.json for an N-layer truncated, text-only,
Engram-free SPEED build. Every per-layer list has to be truncated too, or the loader
disagrees with itself about which layers are KV/index sources (same class of problem
make_trunc.py solves for GGUF)."""
import json, sys
from pathlib import Path

n = int(sys.argv[1]); src = Path(sys.argv[2]); dst = Path(sys.argv[3])
cfg = json.loads(src.read_text())
t = cfg["text_config"]

t["num_hidden_layers"] = n
# layer-id lists: keep only ids inside the truncated trunk
for key in ("kv_source_layer_ids", "index_source_layer_ids", "engram_layer_ids", "dspark_target_layer_ids"):
    if key in t and isinstance(t[key], list):
        t[key] = [i for i in t[key] if i < n]
# engram is dropped entirely in a speed build
t["engram_layer_ids"] = []
# per-layer arrays
if isinstance(t.get("compress_ratios"), list):
    t["compress_ratios"] = t["compress_ratios"][:n]
# V4.1 uses compression ratios 1 and 2; the deepseek4 loader only accepts 0, 4 and 128
# ("DeepSeek-V4 loader only supports compression ratios 0, 4, and 128"). NO_COMPRESS=1
# zeroes them for a SPEED-ONLY build -- attention then runs uncompressed, which changes the
# attention cost but leaves the routed experts (>90% of per-token bytes) untouched.
if __import__("os").environ.get("NO_COMPRESS", "0") != "0":
    t["compress_ratios"] = [0] * n
    t["kv_source_layer_ids"] = []
# scalars that name a layer
if t.get("candidate_source_layer_id", 0) >= n:
    t["candidate_source_layer_id"] = max((i for i in t.get("index_source_layer_ids", []) if i < n), default=0)
# V4-Flash had "hash layers" (token-id -> expert-id `tid2eid` routing tables). V4.1 has
# none -- its Engram tables are a different mechanism (16M-vocab embedding lookup).
t.setdefault("num_hash_layers", 0)
# V4.1 shares compressed KV: only kv_source_layer_ids own a compressor; every other layer
# reads that source layer's cache. The deepseek4 graph assumes each ratio!=0 layer owns one,
# so for a SPEED probe zero the ratio on non-source layers (they have no compressor tensors
# anyway). Their attention then runs uncompressed; the dominant MoE cost is unchanged.
if __import__("os").environ.get("RATIO_SOURCES_ONLY", "0") != "0":
    src = set(t.get("kv_source_layer_ids") or [])
    t["compress_ratios"] = [r if i in src else 0 for i, r in enumerate(t["compress_ratios"])]
# short-context speed build: no indexer (see INDEXER_SKIP in conversion/deepseek.py)
if __import__("os").environ.get("INDEXER_SKIP","0") != "0":
    t["index_source_layer_ids"] = []
    t["candidate_source_layer_id"] = 0
# no MTP shards downloaded
t["num_nextn_predict_layers"] = 0
t["dspark_target_layer_ids"] = []
# text-only
cfg.pop("vision_config", None)
# ModelBase flattens `text_config` only in TextModel.__init__, which runs AFTER
# ModelBase.__init__ -> index_tensors(). index_tensors already needs num_hidden_layers,
# so emit the text keys at top level too (the later flatten is then a no-op).
for k, v in t.items():
    cfg.setdefault(k, v)
dst.write_text(json.dumps(cfg, indent=2) + "\n")
print(f"wrote {dst}: {n} layers, kv_src={t.get('kv_source_layer_ids')}, "
      f"idx_src={t.get('index_source_layer_ids')}, cand={t.get('candidate_source_layer_id')}, "
      f"engram={t['engram_layer_ids']}, nextn={t['num_nextn_predict_layers']}")
