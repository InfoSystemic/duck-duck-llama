#!/usr/bin/env python3
# Pass 2 of the MiMo-V2.6-Pro-RL conversion: stream the 128 expert-parallel safetensors shards
# into the holes the skeleton pass reserved in the GGUF, one shard at a time, then delete it.
# Peak disk = the GGUF + DOWNLOAD_AHEAD shards; the 534 GiB checkpoint never exists on disk at once.
#
# Every shard is sha256-checked against the pinned HF manifest before use. Each routed expert is
# repacked losslessly (same bit shuffle as ModelBase.repack_mxfp4_blocks) and pwritten at
# tensor_offset + expert * bytes_per_expert. A sample per shard is read back and compared
# bit-exact, and dequantized through gguf-py against an independent E2M1 x 2^(e-127) reference.
# State is per shard (fill-state/<shard>.done), so the script resumes where it stopped.
#
#   fill_experts.py            fill everything (resumable)
#   fill_experts.py --verify   no downloads: check that every expert slot is written and no hole remains

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
import re
import struct
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "convert" / "gguf-py"))

import numpy as np  # noqa: E402
import gguf  # noqa: E402

REPO = "XiaomiMiMo/MiMo-V2.6-Pro-RL"
REV = "54b10491b1811c76aa9681a9d0ff872396a4064c"
ROOT = Path("/models/mimo-v26-pro")
SRC = ROOT / "src"
# downloads land on the NVMe root disk so they don't share the SATA SSD with the GGUF writes
INCOMING = Path(os.environ.get("MIMO_INCOMING", "/home/user/mimo-v26-incoming"))
STATE = ROOT / "fill-state"
GGUF_GLOB = "MiMo-V2.6-Pro-RL-MXFP4_MOE-*-of-*.gguf"
MANIFEST = HERE / "hf-manifest-54b10491.json"
DOWNLOAD_AHEAD = int(os.environ.get("MIMO_DOWNLOAD_AHEAD", "3"))
DL_THREADS = int(os.environ.get("MIMO_DL_THREADS", "2"))
WRITE_THREADS = 4

E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)
PROJ_TENSOR = {"gate": "ffn_gate_exps", "up": "ffn_up_exps", "down": "ffn_down_exps"}
EXPERT_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight(_scale)?$")


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def repack(packed: np.ndarray, scale: np.ndarray) -> np.ndarray:
    # U8 codes [rows, cols/2] (element 2i in the low nibble) + U8 E8M0 [rows, cols/32]
    # -> ggml block_mxfp4 per 32 values: scale byte, then 16 bytes holding j (low) | j+16 (high)
    rows, packed_cols = packed.shape
    nb = packed_cols // 16
    src = packed.reshape(rows, nb, 16)
    vals = np.stack((src & 0x0F, src >> 4), axis=-1).reshape(rows, nb, 32)
    qs = vals[:, :, :16] | (vals[:, :, 16:] << 4)
    return np.concatenate((scale.reshape(rows, nb, 1), qs), axis=-1)


def reference_dequant(packed: np.ndarray, scale: np.ndarray) -> np.ndarray:
    rows = packed.shape[0]
    v = np.stack((E2M1[packed & 15], E2M1[packed >> 4]), axis=-1).reshape(rows, -1)
    return v * np.repeat(np.exp2(scale.astype(np.float32) - 127), 32, axis=1)


class Safetensors:
    def __init__(self, path: Path):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.header.pop("__metadata__", None)
        self.base = 8 + n
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def u8(self, name: str) -> np.ndarray:
        t = self.header[name]
        if t["dtype"] != "U8":
            raise ValueError(f"{self.path.name}: {name} is {t['dtype']}, expected U8")
        a, b = t["data_offsets"]
        return self.mm[self.base + a:self.base + b].reshape(t["shape"])


def gguf_slots(gguf_dir: Path, n_expert: int) -> dict[tuple[int, str], dict]:
    # (layer, proj) -> file, offset of expert 0, bytes per expert, rows, cols
    slots: dict[tuple[int, str], dict] = {}
    files = sorted(gguf_dir.glob(GGUF_GLOB))
    if not files:
        raise SystemExit(f"no GGUF splits in {gguf_dir}")
    for path in files:
        reader = gguf.GGUFReader(path)
        for t in reader.tensors:
            m = re.match(r"^blk\.(\d+)\.(ffn_gate_exps|ffn_up_exps|ffn_down_exps)\.weight$", t.name)
            if not m:
                continue
            if t.tensor_type != gguf.GGMLQuantizationType.MXFP4:
                raise ValueError(f"{t.name} is {t.tensor_type.name}, expected MXFP4")
            cols, rows, ne = (int(x) for x in t.shape)
            if ne != n_expert or t.n_bytes != n_expert * rows * (cols // 32) * 17:
                raise ValueError(f"{t.name}: unexpected shape {list(t.shape)} / {t.n_bytes} bytes")
            proj = {v: k for k, v in PROJ_TENSOR.items()}[m.group(2)]
            slots[(int(m.group(1)), proj)] = dict(file=str(path), offset=int(t.data_offset),
                                                  per_expert=t.n_bytes // n_expert, rows=rows, cols=cols)
        del reader
    return slots


def shard_plan(n_expert: int, moe_layers: list[int]) -> dict[str, list[tuple[int, int, str]]]:
    weight_map = json.load(open(SRC / "model.safetensors.index.json"))["weight_map"]
    plan: dict[str, set[tuple[int, int, str]]] = {}
    where: dict[tuple[int, int, str, bool], str] = {}
    for name, fn in weight_map.items():
        if m := EXPERT_RE.match(name):
            key = (int(m.group(1)), int(m.group(2)), m.group(3))
            where[(*key, bool(m.group(4)))] = fn
            plan.setdefault(fn, set()).add(key)
    expected = {(l, e, p) for l in moe_layers for e in range(n_expert) for p in PROJ_TENSOR}
    got = {k[:3] for k in where}
    if got != expected:
        raise SystemExit(f"index does not cover the experts exactly: missing {len(expected - got)}, extra {len(got - expected)}")
    for (l, e, p, _), fn in where.items():
        if where[(l, e, p, False)] != where[(l, e, p, True)]:
            raise SystemExit(f"weight and scale of layer {l} expert {e} {p} live in different shards")
    return {fn: sorted(keys) for fn, keys in sorted(plan.items())}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


class Filler:
    def __init__(self):
        cfg = json.load(open(SRC / "config.json"))
        self.n_expert = cfg["n_routed_experts"]
        self.moe_layers = [i for i, f in enumerate(cfg["moe_layer_freq"]) if f]
        self.slots = gguf_slots(ROOT / "gguf", self.n_expert)
        if set(self.slots) != {(l, p) for l in self.moe_layers for p in PROJ_TENSOR}:
            raise SystemExit("GGUF expert tensors do not match the config's MoE layers")
        self.plan = shard_plan(self.n_expert, self.moe_layers)
        self.manifest = json.load(open(MANIFEST))
        self.fds = {f: os.open(f, os.O_RDWR) for f in {s["file"] for s in self.slots.values()}}
        STATE.mkdir(parents=True, exist_ok=True)
        INCOMING.mkdir(parents=True, exist_ok=True)

    def done(self, shard: str) -> bool:
        return (STATE / f"{shard}.done").exists()

    def write_expert(self, st: Safetensors, key: tuple[int, int, str]) -> None:
        layer, expert, proj = key
        slot = self.slots[(layer, proj)]
        prefix = f"model.layers.{layer}.mlp.experts.{expert}.{proj}_proj"
        packed, scale = st.u8(prefix + ".weight"), st.u8(prefix + ".weight_scale")
        if packed.shape != (slot["rows"], slot["cols"] // 2) or scale.shape != (slot["rows"], slot["cols"] // 32):
            raise ValueError(f"{prefix}: shapes {packed.shape} / {scale.shape} do not fit {slot}")
        data = repack(np.asarray(packed), np.asarray(scale)).tobytes()
        assert len(data) == slot["per_expert"]
        off = slot["offset"] + expert * slot["per_expert"]
        view = memoryview(data)
        while view:
            n = os.pwrite(self.fds[slot["file"]], view, off)
            view, off = view[n:], off + n

    def check_sample(self, st: Safetensors, key: tuple[int, int, str]) -> None:
        layer, expert, proj = key
        slot = self.slots[(layer, proj)]
        prefix = f"model.layers.{layer}.mlp.experts.{expert}.{proj}_proj"
        packed, scale = np.asarray(st.u8(prefix + ".weight")), np.asarray(st.u8(prefix + ".weight_scale"))
        raw = os.pread(self.fds[slot["file"]], slot["per_expert"], slot["offset"] + expert * slot["per_expert"])
        if raw != repack(packed, scale).tobytes():
            raise RuntimeError(f"readback mismatch at {prefix}")
        got = gguf.quants.dequantize(np.frombuffer(raw, dtype=np.uint8).reshape(slot["rows"], -1),
                                     gguf.GGMLQuantizationType.MXFP4)
        if not np.array_equal(got.reshape(slot["rows"], slot["cols"]), reference_dequant(packed, scale)):
            raise RuntimeError(f"gguf-py MXFP4 dequant disagrees with the E2M1 reference at {prefix}")

    def process(self, shard: str, path: Path, got: str, pool: cf.ThreadPoolExecutor) -> None:
        # got = sha256 computed by the downloader thread, so hashing overlaps the previous shard's writes
        t0 = time.time()
        want = self.manifest["sha256"][shard]
        if got != want:
            raise RuntimeError(f"{shard}: sha256 {got} != manifest {want}")
        st = Safetensors(path)
        keys = self.plan[shard]
        list(pool.map(lambda k: self.write_expert(st, k), keys))
        for fd in self.fds.values():
            os.fdatasync(fd)
        self.check_sample(st, random.choice(keys))
        (STATE / f"{shard}.done").write_text(json.dumps({"sha256": got, "experts": len(keys), "t": time.time()}))
        for fd in self.fds.values():
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        del st
        log(f"{shard}: {len(keys)} expert tensors written + verified in {time.time() - t0:.1f}s")

    def run(self) -> None:
        from huggingface_hub import hf_hub_download

        todo = [s for s in self.plan if not self.done(s)]
        log(f"{len(self.plan) - len(todo)} of {len(self.plan)} shards already done; {len(todo)} to go")
        ready: dict[str, tuple[Path, str]] = {}
        cond = threading.Condition()
        errors: list[BaseException] = []
        queue = iter(todo)
        in_flight = 0

        def downloader() -> None:
            # DL_THREADS of these take shards in order; ready + in-flight never exceeds DOWNLOAD_AHEAD files on disk
            nonlocal in_flight
            while True:
                with cond:
                    cond.wait_for(lambda: len(ready) + in_flight < DOWNLOAD_AHEAD or errors)
                    shard = None if errors else next(queue, None)
                    if shard is None:
                        return
                    in_flight += 1
                try:
                    local = SRC / shard
                    if local.exists() and not local.is_symlink():
                        path = local
                    else:
                        path = Path(hf_hub_download(REPO, shard, revision=REV, local_dir=INCOMING))
                    digest = sha256_of(path)
                except BaseException as e:  # surface in the main thread
                    with cond:
                        errors.append(e)
                        cond.notify_all()
                    return
                with cond:
                    in_flight -= 1
                    ready[shard] = (path, digest)
                    cond.notify_all()

        for _ in range(DL_THREADS):
            threading.Thread(target=downloader, daemon=True).start()
        t_start, n_done = time.time(), 0
        with cf.ThreadPoolExecutor(WRITE_THREADS) as pool:
            for shard in todo:
                with cond:
                    cond.wait_for(lambda: shard in ready or errors)
                    if errors:
                        raise errors[0]
                    path, digest = ready[shard]
                self.process(shard, path, digest, pool)
                if path.parent == INCOMING:
                    path.unlink()
                with cond:
                    del ready[shard]
                    cond.notify_all()
                n_done += 1
                rate = (time.time() - t_start) / n_done
                log(f"progress {len(self.plan) - len(todo) + n_done}/{len(self.plan)}, ETA {rate * (len(todo) - n_done) / 60:.0f} min")
        log("ALL SHARDS FILLED")

    def verify(self) -> bool:
        ok = True
        missing = [s for s in self.plan if not self.done(s)]
        if missing:
            log(f"{len(missing)} shards not filled yet, e.g. {missing[:3]}")
            ok = False
        for f, fd in self.fds.items():
            size = os.fstat(fd).st_size
            pos, holes = 0, []
            while pos < size:
                try:
                    hole = os.lseek(fd, pos, os.SEEK_HOLE)
                except OSError:
                    break
                if hole >= size:
                    break
                try:
                    pos = os.lseek(fd, hole, os.SEEK_DATA)
                except OSError:  # hole runs to EOF
                    pos = size
                holes.append((hole, pos))
            log(f"{Path(f).name}: {size / 2**30:.2f} GiB, {len(holes)} holes"
                + (f", first at {holes[0]}" if holes else ""))
            ok &= not holes
        log("VERIFY " + ("PASS" if ok else "FAIL"))
        return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    filler = Filler()
    if args.verify:
        sys.exit(0 if filler.verify() else 1)
    filler.run()
    sys.exit(0 if filler.verify() else 1)
