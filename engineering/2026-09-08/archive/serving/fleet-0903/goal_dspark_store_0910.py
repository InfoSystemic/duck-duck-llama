"""Optional isolated MTP cache: real mtp.* names and a separate eviction journal.

No DSpark record enters the backbone cache. This module never constructs a store
or downloads data on import. Existing protected store implementations are intact.
"""
import json
import os
from pathlib import Path
import re
import types

from deepseek_v41_checkpoint_0910 import REVISION
from deepseek_v41_checkpoint_transport_0910b import CoalescedStore
from deepseek_v41_resident_store_0910 import ResidentStore
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import sha256

MARKER = 'goal-dspark-cache-0910.json'
EXPERT = re.compile(r'mtp\.[0-2]\.ffn\.experts\.(?:0|[1-9][0-9]*)\.w[123]\.(?:weight|scale)')


def initialize_mtp_cache(root, backbone_root):
    """Explicitly initialize a distinct, empty, task-owned directory; no network."""
    root, backbone_root = Path(root), Path(backbone_root).resolve()
    if root.resolve() == backbone_root:
        raise ValueError('DSpark cache must be separate from backbone cache')
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink() or root.stat().st_uid != os.getuid():
        raise ValueError('Expected an owned, nonsymlink DSpark cache directory')
    if any(root.iterdir()):
        raise ValueError('DSpark cache initialization requires an empty directory')
    atomic_json(root / 'owner.json', dict(task='deepseek-v41-native-0910', revision=REVISION))
    atomic_json(root / MARKER, dict(task='goal-dspark-native-0910', revision=REVISION,
                                   backbone_cache=str(backbone_root), namespace='mtp.*'))


def check_mtp_root(root):
    root = Path(root)
    assert root.is_dir() and not root.is_symlink() and root.stat().st_uid == os.getuid()
    marker = json.loads((root / MARKER).read_text())
    assert marker['task'] == 'goal-dspark-native-0910' and marker['revision'] == REVISION
    assert marker['namespace'] == 'mtp.*' and root.resolve() != Path(marker['backbone_cache']).resolve()
    assert json.loads((root / 'owner.json').read_text()) == dict(task='deepseek-v41-native-0910', revision=REVISION)
    assert not (root / 'eviction.json').exists(), 'Unexpected backbone eviction journal in MTP cache'


def finish_mtp_eviction(root):
    root = Path(root)
    check_mtp_root(root)
    journal = root / 'mtp-eviction.json'
    if not journal.exists():
        return
    victims = json.loads(journal.read_text())['victims']
    manifest = root / 'tensors.jsonl'
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    known = {item['name']: item for item in records}
    removed = set()
    for item in victims:
        name = item['name']
        assert EXPERT.fullmatch(name) and name not in removed
        removed.add(name)
        if name in known:
            assert known[name]['sha256'] == item['sha256'] and known[name]['bytes'] == item['bytes']
        path = root / (name + '.bin')
        if path.exists():
            stat = path.stat()
            assert path.is_file() and not path.is_symlink() and stat.st_uid == os.getuid() and stat.st_nlink == 1
            assert stat.st_size == item['bytes'] and sha256(path) == item['sha256']
            path.unlink()
    temporary = root / 'tensors.mtp-next'
    with temporary.open('w') as handle:
        for record in records:
            if record['name'] not in removed:
                handle.write(json.dumps(record, separators=(',', ':')) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(manifest)
    journal.unlink()


class MTPResidentStore(ResidentStore):
    def __init__(self, root, output, *, allow_download=False, cap_bytes=8 << 30, workers=4):
        check_mtp_root(root)
        finish_mtp_eviction(root)
        super().__init__(root, output, cap_bytes=cap_bytes, workers=workers)
        self.catalog = types.SimpleNamespace(tensors={name: meta for name, meta in self.catalog.tensors.items()
                                                     if name.startswith('mtp.')})
        assert all(name in self.catalog.tensors for name in self.records)
        self.existing_projection = {}
        self.allow_download = allow_download

    def ensure(self, names):
        names = list(dict.fromkeys(names))
        assert all(name in self.catalog.tensors and name.startswith('mtp.') for name in names)
        missing = [name for name in names if name not in self.records]
        if missing and not self.allow_download:
            raise RuntimeError(f'DSpark native tensors absent locally; downloads disabled ({len(missing)} tensors)')
        if names and all(name in self.resident_names for name in names):
            self.tick += 1
            self.touched.update({name: self.tick for name in names})
            return
        self.tick += 1
        amount = sum(self.catalog.tensors[name]['bytes'] for name in missing)
        needed = self.stored_bytes + amount - self.cap_bytes
        if needed > 0:
            protected = {name.rsplit('.', 2)[0] for name in names if EXPERT.fullmatch(name)}
            groups = {}
            for name, record in self.records.items():
                if EXPERT.fullmatch(name):
                    prefix = name.rsplit('.', 2)[0]
                    if prefix not in protected:
                        groups.setdefault(prefix, []).append(record)
            order = sorted(groups, key=lambda prefix: max(self.touched.get(row['name'], 0) for row in groups[prefix]))
            victims, reclaimed = [], 0
            for prefix in order:
                victims.extend(groups[prefix])
                reclaimed += sum(row['bytes'] for row in groups[prefix])
                if reclaimed >= needed:
                    break
            assert reclaimed >= needed, 'Insufficient evictable MTP cache; common tensors stay pinned'
            atomic_json(self.root / 'mtp-eviction.json', dict(victims=[
                {key: row[key] for key in ('name', 'bytes', 'sha256')} for row in victims]))
            try:
                finish_mtp_eviction(self.root)
                for row in victims:
                    name = row['name']
                    del self.records[name]
                    self.verified.discard(name)
                    self.touched.pop(name, None)
                self.stored_bytes -= reclaimed
                self.evicted_bytes += reclaimed
                for prefix in {row['name'].rsplit('.', 2)[0] + '.' for row in victims} & self.residents.keys():
                    self.release(prefix)
            except BaseException:
                self.release_all()
                raise
        # Bypass ServingStore.ensure: its journal accepts backbone names only.
        CoalescedStore.ensure(self, names)
        self.touched.update({name: self.tick for name in names})
