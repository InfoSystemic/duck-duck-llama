#!/usr/bin/env python3
"""Reproduce the pinned common library, then isolate a fresh-sequence MTP fix."""
import difflib
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

from model_measurement_guard import ModelMeasurementGuard
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import inference_snapshot, runtime_environment, sha256
from qwen_mtp_fresh_seed_transform_0910 import transform
from select_flash_q4_0910c import Manager

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parents[1] / 'engines/llama.cpp-q4e-goal-0904'
OUT = BASE / 'results/qwen-mtp-fresh-seed-build-0910'


def main():
    assert os.sched_getaffinity(0) == {127} and not OUT.exists()
    manager = Manager()
    peer = manager.validate_current()
    assert peer['quant'] == 'UD-Q4_K_XL' and set(inference_snapshot()) == {str(peer['pid'])}
    guard = ModelMeasurementGuard(peer['pid'], {peer['pid']:18131}, inference_snapshot)
    guard.assert_idle()
    runtime_path = BASE / 'results/qwen-get-rows-runtime-0909/result.json'
    runtime = json.loads(runtime_path.read_text())
    original_library = (Path(runtime['server']).parent / 'libllama-common.so.0.3.0').resolve()
    commands_path = ENGINE / 'build-goal/compile_commands.json'
    unit, = [r for r in json.loads(commands_path.read_text()) if r['file'].endswith('/common/speculative.cpp')]
    command = shlex.split(unit['command'])
    source, cwd = Path(unit['file']), Path(unit['directory'])
    object_argument = command[command.index('-o')+1]
    original_object = cwd / object_argument
    link_path = cwd / 'CMakeFiles/llama-common.dir/link.txt'
    link = shlex.split(link_path.read_text())
    assert link.count(object_argument) == 1
    paths = [Path(__file__),BASE/'qwen_mtp_fresh_seed_transform_0910.py',runtime_path,
        commands_path,source,original_object,original_library,link_path,BASE/'model_measurement_guard.py']
    output_index = link.index('-o')+1
    paths.extend((cwd/value).resolve() for i,value in enumerate(link)
        if i != output_index and not value.startswith('-') and (cwd/value).is_file())
    inputs = {str(p):sha256(p) for p in paths}
    with (BASE/'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        guard.assert_idle()
        OUT.mkdir();private=OUT/'private-common';private.mkdir()
        modified=private/'speculative.cpp'
        original=source.read_text();changed=transform(original);modified.write_text(changed)
        (private/'speculative.cpp.patch').write_text(''.join(difflib.unified_diff(
            original.splitlines(True),changed.splitlines(True),fromfile='common/speculative.cpp',tofile='common/speculative.cpp')))
        result=dict(started=time.time(),passed=False,controller_pid=os.getpid(),steps=[],
            input_sha256=inputs,private_source_sha256={str(modified):sha256(modified)},
            parent_common=str(original_library),parent_common_sha256=sha256(original_library),
            model_loaded=False,model_gain_established=False,
            scope='Reset MTP pending hidden carry for fresh position-zero sequences. Keep continuation carry. Private common library; original engine sources and selected Flash runtime unchanged. Model validation required.')
        child=None
        def save():atomic_json(OUT/'result.json',result)
        def cancel(*_):raise InterruptedError('Release only this owned private build')
        for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,cancel)
        def run(argv,label,env=None):
            nonlocal child
            guard.assert_idle();log=OUT/(label+'.log')
            with log.open('w') as handle:
                child=subprocess.Popen(argv,cwd=cwd,env=env,stdout=handle,stderr=subprocess.STDOUT)
                deadline=time.monotonic()+600
                while child.poll() is None:
                    guard.assert_idle();assert time.monotonic()<deadline;time.sleep(.25)
            result['steps'].append(dict(label=label,command=argv,exit_code=child.returncode,log_sha256=sha256(log)))
            save();assert child.returncode==0,label
            print(json.dumps(dict(completed=label)),flush=True)
            return log.read_text()
        def compile_to(src,obj,label):
            argv=list(command);argv[argv.index('-o')+1]=str(obj);argv[argv.index('-c')+1]=str(src)
            return run(argv,label)
        def link_to(obj,library,label):
            argv=[str(obj) if v==object_argument else v for v in link]
            argv[output_index]=str(library)
            return run(argv,label)
        save()
        try:
            baseline_object=OUT/'baseline-speculative.cpp.o'
            compile_to(source,baseline_object,'baseline-compile')
            assert sha256(baseline_object)==sha256(original_object),'Parent object is not reproduced'
            baseline_library=OUT/'baseline-libllama-common.so.0.3.0'
            link_to(baseline_object,baseline_library,'baseline-link')
            assert sha256(baseline_library)==sha256(original_library),'Pinned common library is not reproduced'
            result.update(baseline_object_identical=True,baseline_library_identical=True)
            obj=private/'speculative.cpp.o';library=private/'libllama-common.so.0.3.0'
            compile_to(modified,obj,'candidate-compile');link_to(obj,library,'candidate-link')
            bundle=OUT/'runtime';bundle.mkdir()
            for old in Path(runtime['server']).parent.iterdir():
                if old.is_file() or old.is_symlink():
                    target=library if old.name.startswith('libllama-common.so') else old.resolve()
                    (bundle/old.name).symlink_to(target)
            env={k:v for k,v in os.environ.items() if k not in runtime_environment(os.environ)}
            candidate_env=dict(runtime['runtime_env'],LD_LIBRARY_PATH=str(bundle));env.update(candidate_env)
            loader=run(['ldd',str(bundle/'llama-server')],'loader',env)
            assert 'not found' not in loader
            common_lines=[line for line in loader.splitlines() if 'libllama-common.so.0 =>' in line]
            assert len(common_lines)==1 and str(bundle/'libllama-common.so.0') in common_lines[0]
            run([str(bundle/'llama-server'),'--version'],'version',env)
            assert all(sha256(p)==h for p,h in inputs.items())
            manager.validate_current()
            result.update(passed=True,library=str(library),library_sha256=sha256(library),
                server=str(bundle/'llama-server'),server_sha256=runtime['server_sha256'],runtime_env=candidate_env,
                cpu=runtime['cpu'],base=runtime['base'],llama=runtime['llama'],peer_preserved=True)
        except BaseException as error:
            result['error']=repr(error);raise
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
            result['finished']=time.time();save()


if __name__=='__main__':
    os.umask(0o077);main()
