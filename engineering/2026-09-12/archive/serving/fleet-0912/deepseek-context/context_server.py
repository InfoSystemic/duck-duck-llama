#!/usr/bin/env python3
"""Unpromoted context evaluation entry; selected fleet files remain unchanged."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1] / 'fleet-0903'
sys.path.insert(0, str(BASE))

import deepseek_v41_server_0910 as server
import deepseek_v41_goal2_server_0910 as goal2
from deepseek_v41_resident_store_0910 import ResidentStore
from qwen_high_quant_trial import atomic_json
from bounded_engram import BoundedEngram
from context_policy import BoundedPrefill, resize_context


class ContextRuntime(goal2.Goal2Runtime):
    def __init__(self, args):
        original_builder, original_store = server.meta_model, goal2.ResidentStore

        def builder():
            module, config, adapter, model = original_builder()
            resize_context(module, config, model, args.context)
            return module, config, adapter, model

        class ContextStore(ResidentStore):
            def __init__(self, *positional, **keywords):
                super().__init__(*positional, **keywords)
                self.context_rows = BoundedEngram(
                    self, args.engram_cache, args.row_cache_max, args.persisted_row_max)

            def rows(self, prefix, ids):
                return self.context_rows.rows(prefix, ids)

            def close(self):
                try:
                    super().close()
                finally:
                    cache = getattr(self, 'context_rows', None)
                    if cache is not None:
                        cache.close()

        server.meta_model, goal2.ResidentStore = builder, ContextStore
        try:
            super().__init__(args)
        finally:
            server.meta_model, goal2.ResidentStore = original_builder, original_store

    def prepare(self, body):
        try:
            return super().prepare(body)
        except AssertionError as error:
            if str(error) == 'This initial endpoint has a 256-token context limit':
                raise AssertionError(f'Prompt plus output exceeds {self.config.max_seq_len} tokens') from None
            raise

    def generate(self, ids, count, emit):
        original = self.model
        proxy = BoundedPrefill(original, self.args.initial_prefill)
        self.model = proxy
        try:
            result = super().generate(ids, count, emit)
            atomic_json(self.args.output / 'context-last-request.json', dict(
                context=self.config.max_seq_len, initial_prefill=self.args.initial_prefill,
                prefill_model_calls=proxy.prefill_calls, rows=self.store.context_rows.metrics(),
                usage=result['usage'], timings=result['timings']))
            return result
        finally:
            self.model = original


class ContextHandler(server.Handler):
    def do_GET(self):
        if self.path != '/health':
            return super().do_GET()
        runtime = self.server.runtime
        self.json(200, dict(status='ok', model=server.MODEL, busy=runtime.lock.locked(),
                            context=runtime.config.max_seq_len, native_precision=True,
                            initial_prefill=runtime.args.initial_prefill,
                            experimental_context=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--engram-cache', type=Path, required=True)
    parser.add_argument('--context', type=int, choices=(256, 4096, 16384), default=4096)
    parser.add_argument('--initial-prefill', type=int, choices=range(1, 257), default=256)
    parser.add_argument('--row-cache-max', type=int, default=8192)
    parser.add_argument('--persisted-row-max', type=int, default=8192)
    parser.add_argument('--port', type=int, default=18174)
    parser.add_argument('--lifecycle-lock-fd', type=int, required=True)
    args = parser.parse_args()
    goal2.read_environment()
    if not 128 <= args.row_cache_max <= 100000 or not 0 <= args.persisted_row_max <= 100000:
        parser.error('row-cache-max must be 128..100000 and persisted-row-max 0..100000')
    lock = BASE / 'results/qwen-q6-trial-0907/lifecycle.lock'
    if os.fstat(args.lifecycle_lock_fd).st_ino != lock.stat().st_ino:
        parser.error('The existing fleet lifecycle lock descriptor is required')
    args.output.mkdir(parents=True, exist_ok=True)
    runtime = ContextRuntime(args)
    httpd = server.ThreadingHTTPServer(('127.0.0.1', args.port), ContextHandler)
    httpd.daemon_threads = True
    httpd.runtime = runtime
    atomic_json(args.output / 'ready.json', dict(
        ready=True, pid=os.getpid(), port=args.port, model=server.MODEL, revision=server.REVISION,
        context=args.context, native_precision=True, experimental_context=True,
        initial_prefill=args.initial_prefill, vision=False, dspark=False,
        cache_limit_bytes=runtime.store.cap_bytes, rows=runtime.store.context_rows.metrics(),
        ready_at=time.time()))
    try:
        httpd.serve_forever(poll_interval=.25)
    finally:
        httpd.server_close()
        runtime.store.close()


if __name__ == '__main__':
    main()
