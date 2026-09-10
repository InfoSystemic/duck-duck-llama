#!/usr/bin/env python3
"""Finish runtime validation after the build guard mistook --version for inference."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from model_measurement_guard import ModelMeasurementGuard
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import inference_snapshot, runtime_environment, sha256
from select_flash_q4_0910c import Manager

BASE=Path(__file__).resolve().parent
OUT=BASE/'results/qwen-mtp-fresh-seed-runtime-0910'


def main():
    assert os.sched_getaffinity(0)=={127} and not OUT.exists()
    with (BASE/'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manager=Manager();peer=manager.validate_current()
        guard=ModelMeasurementGuard(peer['pid'],{peer['pid']:18131},inference_snapshot);guard.assert_idle()
        path=BASE/'results/qwen-mtp-fresh-seed-build-0910/result.json';build=json.loads(path.read_text())
        assert build['baseline_object_identical'] and build['baseline_library_identical']
        assert not build['passed'] and 'Model work is active or queued' in build['error']
        assert [row['label'] for row in build['steps']]==['baseline-compile','baseline-link','candidate-compile','candidate-link','loader']
        assert all(row['exit_code']==0 for row in build['steps'])
        assert all(sha256(p)==h for key in ['input_sha256','private_source_sha256'] for p,h in build[key].items())
        runtime=json.loads((BASE/'results/qwen-get-rows-runtime-0909/result.json').read_text())
        bundle=path.parent/'runtime';library=path.parent/'private-common/libllama-common.so.0.3.0'
        assert (bundle/'libllama-common.so.0').resolve()==library
        candidate_env=dict(runtime['runtime_env'],LD_LIBRARY_PATH=str(bundle))
        env={k:v for k,v in os.environ.items() if k not in runtime_environment(os.environ)};env.update(candidate_env)
        OUT.mkdir()
        with (OUT/'version.log').open('w') as log:
            # This exact command loads no model and exits. Guard the serving peer
            # before and after; its short-lived --version PID is owned here.
            done=subprocess.run([str(bundle/'llama-server'),'--version'],env=env,cwd=BASE,
                stdout=log,stderr=subprocess.STDOUT,timeout=15)
        assert done.returncode==0
        guard.assert_idle();manager.validate_current()
        assert set(inference_snapshot())=={str(peer['pid'])}
        assert all(sha256(p)==h for key in ['input_sha256','private_source_sha256'] for p,h in build[key].items())
        result=dict(finished=time.time(),passed=True,build_result=str(path),build_result_sha256=sha256(path),
            source_sha256=sha256(__file__),baseline_object_identical=True,baseline_library_identical=True,
            library=str(library),library_sha256=sha256(library),parent_common_sha256=build['parent_common_sha256'],
            private_source_sha256=build['private_source_sha256'],server=str(bundle/'llama-server'),
            server_sha256=runtime['server_sha256'],runtime_env=candidate_env,cpu=runtime['cpu'],base=runtime['base'],llama=runtime['llama'],
            version_exit=done.returncode,version_log_sha256=sha256(OUT/'version.log'),peer_preserved=True,
            scope='Build and dynamic-loader validation completed. Original failed build result retained: guard classified its owned --version process as competing inference. No model correctness or speed claim.')
        atomic_json(OUT/'result.json',result)
        print(json.dumps({k:result[k] for k in ['passed','baseline_object_identical','baseline_library_identical','library_sha256','peer_preserved']}))


if __name__=='__main__':
    os.umask(0o077);main()
