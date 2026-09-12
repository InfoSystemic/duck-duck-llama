#!/usr/bin/env python3
"""Explicitly selected exact16/native-sparse additions to the native CPU server."""
import json
import os
from pathlib import Path
import re
import threading

import torch

import deepseek_v41_server_0910 as server
from deepseek_v41_resident_store_0910 import ResidentStore, bind
from goal_runtime_0910 import Optimizations
from goal_exact16_runtime_0910 import install as install_exact16
from goal_sparse_native_0910 import NativeSparse
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import sha256


BASE = Path(__file__).resolve().parent
FULL = dict(quant=True, reuse=True, grouped=True, hc=True, sparse=True, native_workers=16)
EXACT_FLAG = 'DEEPSEEK_GOAL2_EXACT16'
SPARSE_FLAG = 'DEEPSEEK_GOAL2_NATIVE_SPARSE'
CAP_FLAG = 'DEEPSEEK_GOAL2_PACKED_CAP_GIB'
EXACT_ROOT = BASE / 'results/deepseek-v41-grouped-int16-exact-goal-0910'
SPARSE_ROOT = BASE / 'results/goal_sparse_native_0910'


def read_environment(environment=None):
    """Read only the three required, nonsecret Goal 2 settings."""
    environment = os.environ if environment is None else environment
    flags = {name: environment.get(name) for name in (EXACT_FLAG, SPARSE_FLAG, CAP_FLAG)}
    for name in (EXACT_FLAG, SPARSE_FLAG):
        if flags[name] not in ('0', '1'):
            raise ValueError(f'{name} must explicitly be the literal 0 or 1')
    cap = flags[CAP_FLAG]
    if not isinstance(cap, str) or re.fullmatch(r'0|[1-9][0-9]*', cap) is None or int(cap) > 128:
        raise ValueError(f'{CAP_FLAG} must explicitly be a decimal integer from 0 through 128')
    if flags[EXACT_FLAG] == flags[SPARSE_FLAG] == '0':
        raise ValueError('At least one Goal 2 candidate must be enabled')
    return flags


def validate_proof(path, hash_key):
    proof = json.loads(path.read_text())
    if not proof.get('passed') or not proof.get(hash_key):
        raise RuntimeError(f'Candidate proof did not pass: {path}')
    if not all(sha256(name) == digest for name, digest in proof[hash_key].items()):
        raise RuntimeError(f'Candidate proof source hash mismatch: {path}')


def source_hashes(exact16, native_sparse):
    names = [Path(__file__).name, 'deepseek_v41_server_0910.py',
             'deepseek_v41_resident_store_0910.py', 'deepseek_v41_serving_store_0910.py',
             'deepseek_v41_native_bridge_0910.py', 'goal_runtime_0910.py',
             'goal_grouped_moe_0910.py', 'deepseek_v41_native_grouped_goal_0910.py',
             'goal_native_quant_0910.py', 'goal_quant_reuse_0910.py',
             'goal_hc_0910.py', 'goal_sparse_script_0910.py',
             'results/deepseek-v41-native-gemm-0910b/libdeepseek-v41-native-gemm.so',
             'results/deepseek-v41-native-grouped-goal-0910/libdeepseek-v41-native-grouped.so',
             'results/goal_native_quant_0910/libgoal_native_quant_0910.so',
             'results/goal_hc_0910/libgoal_hc_0910.so']
    if exact16:
        names += ['goal_exact16_runtime_0910.py', 'deepseek_v41_grouped_int16_exact_goal_0910.py',
                  'deepseek-v41-grouped-int16-exact-goal-0910.cpp',
                  str(EXACT_ROOT / 'libdeepseek-v41-grouped-int16-exact.so'),
                  str(EXACT_ROOT / 'kernel-check.json'), str(EXACT_ROOT / 'runtime-check.json')]
    if native_sparse:
        names += ['goal_sparse_native_0910.py', 'goal_sparse_native_0910.cpp',
                  str(SPARSE_ROOT / 'libgoal_sparse_native_0910.so'),
                  str(SPARSE_ROOT / 'fixture-check.json')]
    return {str(BASE / name): sha256(BASE / name) for name in names}


class Goal2Runtime(server.Runtime):
    def __init__(self, args):
        self.goal2_environment = read_environment()
        self.use_exact16 = self.goal2_environment[EXACT_FLAG] == '1'
        self.use_native_sparse = self.goal2_environment[SPARSE_FLAG] == '1'
        self.packed_cap_bytes = int(self.goal2_environment[CAP_FLAG]) << 30
        if self.use_exact16:
            validate_proof(EXACT_ROOT / 'kernel-check.json', 'input_sha256')
            validate_proof(EXACT_ROOT / 'runtime-check.json', 'sha256')
        if self.use_native_sparse:
            validate_proof(SPARSE_ROOT / 'fixture-check.json', 'sha256')
        hashes = source_hashes(self.use_exact16, self.use_native_sparse)
        server.ServingStore, server.bind = ResidentStore, bind
        super().__init__(args)
        self.original_fp4, self.original_fp8 = self.module.fp4_gemm, self.module.fp8_gemm
        self.optimizations = Optimizations(self)
        self.optimizations.configure(FULL)
        self.original_grouped_backend = self.optimizations.grouped.backend
        self.exact16 = None
        if self.use_exact16:
            self.exact16 = install_exact16(self, self.optimizations, cap_bytes=self.packed_cap_bytes)
            self.exact16.configure(dict(exact16=True, timing=False))
        self.native_sparse = None
        if self.use_native_sparse:
            self.native_sparse = NativeSparse(SPARSE_ROOT / 'libgoal_sparse_native_0910.so')
            self.native_sparse.install(self.module)
        self._assert_configuration()
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output / 'goal2-config.json', dict(
            environment=self.goal2_environment, optimizations=dict(FULL),
            exact16=self.use_exact16, native_sparse=self.use_native_sparse,
            packed_cap_bytes=self.packed_cap_bytes, exact16_timing=False,
            torch_workers=torch.get_num_threads(), native_workers=self.native.workers,
            fp4_fp8_functions_preserved=True, sparse_fallback='SparseScript',
            source_sha256=hashes, pid=os.getpid()))

    def _assert_configuration(self):
        opt = self.optimizations
        assert torch.get_num_threads() == 16 and self.native.workers == 16
        assert opt.config == FULL and opt.reuse.enabled and opt.grouped.enabled and opt.sparse.enabled
        assert self.module.act_quant == opt.quant.act_quant
        assert self.module.hc_split_sinkhorn == opt.hc.hc_split_sinkhorn
        assert self.module.fp4_gemm == self.original_fp4 and self.module.fp8_gemm == self.original_fp8
        if self.use_exact16:
            assert self.exact16.enabled and not self.exact16.record_timing
            assert opt.grouped.backend is self.exact16.backend
            assert self.exact16.cap_bytes == self.packed_cap_bytes
        else:
            assert self.exact16 is None and opt.grouped.backend is self.original_grouped_backend
        if self.use_native_sparse:
            assert self.native_sparse.enabled and self.native_sparse.baseline is opt.sparse
            assert self.module.sparse_attn == self.native_sparse.sparse_attn
        else:
            assert self.native_sparse is None and self.module.sparse_attn is opt.sparse

    def _metrics(self):
        return dict(exact16=self.exact16.metrics() if self.exact16 is not None else {},
                    sparse=dict(native_calls=self.native_sparse.native_calls if self.native_sparse else 0,
                                fallback_calls=self.native_sparse.fallback_calls if self.native_sparse else 0))

    def generate(self, ids, count, emit):
        # ThreadingHTTPServer creates a fresh request thread. Set its ATen/OpenMP
        # thread count explicitly before entering any model or native operation.
        torch.set_num_threads(16)
        self._assert_configuration()
        before = self._metrics()
        result = super().generate(ids, count, emit)
        self._assert_configuration()
        after = self._metrics()
        exact_delta = {name: value - before['exact16'][name]
                       for name, value in after['exact16'].items()
                       if name not in ('packed_bytes', 'packed_entries', 'cap_bytes')}
        sparse_delta = {name: value - before['sparse'][name] for name, value in after['sparse'].items()}
        pack_count_delta = exact_delta.get('pack_count', 0)
        atomic_json(self.args.output / 'goal2-last-request.json', dict(
            completed=self.completed, environment=self.goal2_environment,
            request_thread_id=threading.get_native_id(), torch_workers=torch.get_num_threads(),
            native_workers=self.native.workers, flags_verified=True,
            pack_count_delta=pack_count_delta, sparse_native_calls_delta=sparse_delta['native_calls'],
            zero_packing_this_request=pack_count_delta == 0,
            exact16_metrics_delta=exact_delta, sparse_calls_delta=sparse_delta,
            packed_state={name: after['exact16'][name] for name in ('packed_bytes', 'packed_entries', 'cap_bytes')
                          if name in after['exact16']},
            sparse_fallback_status=dict(self.native_sparse.fallback_status) if self.native_sparse else {},
            downloaded_bytes=result['timings']['downloaded_bytes'],
            usage=result['usage'], timings=result['timings'], finish_reason=result['finish_reason']))
        return result


def main():
    # Fail on an absent/malformed selection before the ordinary server parser or
    # runtime can create native mappings. All normal endpoint lifecycle checks stay active.
    read_environment()
    server.ServingStore, server.bind, server.Runtime = ResidentStore, bind, Goal2Runtime
    server.main()


if __name__ == '__main__':
    main()
