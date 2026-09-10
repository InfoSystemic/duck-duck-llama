#!/usr/bin/env python3
"""Reproduce the current Qwen CPU and isolate two exact R8 projection changes."""
import difflib
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from model_measurement_guard import ModelMeasurementGuard
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import inference_snapshot,sha256
from qwen_r8_projection_transform_0910 import transform
from select_flash_q4_0910c import Manager

BASE=Path(__file__).resolve().parent
ENGINE=BASE.parents[1]/'engines/llama.cpp-q4e-goal-0904'
OUT=BASE/'results/qwen-r8-projection-build-0910'


def main():
    assert os.sched_getaffinity(0)=={127} and not OUT.exists()
    manager=Manager();peer=manager.validate_current()
    guard=ModelMeasurementGuard(peer['pid'],{peer['pid']:18131},inference_snapshot);guard.assert_idle()
    parent_path=BASE/'results/qwen-q6-q8-wide-batch-0907/private-cpu/manifest.json'
    current_path=BASE/'results/qwen-get-rows-columns-build-0909/private-cpu/manifest.json'
    runtime_path=BASE/'results/qwen-get-rows-runtime-0909/result.json'
    fixed_path=BASE/'results/qwen-mtp-fresh-seed-runtime-0910/result.json'
    parent,current,runtime,fixed=[json.loads(p.read_text()) for p in [parent_path,current_path,runtime_path,fixed_path]]
    assert runtime['passed'] and fixed['passed'] and sha256(runtime['cpu'])==current['library_sha256']==runtime['cpu_sha256']
    assert sha256(fixed['library'])==fixed['library_sha256']
    compile_command=list(parent['compile_commands'][1]);source=Path(compile_command[compile_command.index('-c')+1])
    original_object=compile_command[compile_command.index('-o')+1]
    link_command=list(current['link_command']);assert link_command.count(original_object)==1
    paths=[Path(__file__),BASE/'qwen_r8_projection_transform_0910.py',BASE/'qwen-q8-r8-k160-0910.h',
        parent_path,current_path,runtime_path,fixed_path,source,Path(original_object),Path(fixed['library']),BASE/'model_measurement_guard.py']
    paths += [Path(v) for v in link_command if v.endswith(('.o','.a','.so.0.22.0')) and Path(v).is_file()]
    inputs={str(p):sha256(p) for p in paths}
    for p,h in parent['private_source_sha256'].items():assert sha256(p)==h;inputs[p]=h
    with (BASE/'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guard.assert_idle()
        OUT.mkdir();private=OUT/'private-cpu';private.mkdir()
        cpp=private/'repack-x86.cpp';original=source.read_text();changed=transform(original);cpp.write_text(changed)
        for name,origin in [('qwen-q8-batch.h',source.parent/'qwen-q8-batch.h'),('qwen-q8-r8-k160-0910.h',BASE/'qwen-q8-r8-k160-0910.h')]:
            (private/name).write_bytes(origin.read_bytes())
        (private/'repack-x86.cpp.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),changed.splitlines(True),fromfile='a/ggml/src/ggml-cpu/arch/x86/repack.cpp',tofile='b/ggml/src/ggml-cpu/arch/x86/repack.cpp')))
        private_sources={str(p):sha256(p) for p in [cpp,private/'qwen-q8-batch.h',private/'qwen-q8-r8-k160-0910.h']}
        result=dict(started=time.time(),passed=False,peer_pid=peer['pid'],input_sha256=inputs,private_source_sha256=private_sources,
            steps=[],parent_cpu_sha256=runtime['cpu_sha256'],model_loaded=False,
            scope='Opt-in K160/NR1 activation preparation retaining contiguous R8 weight reads; independently opt-in tile8 for K1536. Precision, layout and each output FMA order preserved by design; validation required.')
        owned=None
        def save():atomic_json(OUT/'result.json',result)
        def cancel(*_):raise InterruptedError('Stop only the owned private R8 build')
        for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,cancel)
        def run(command,label):
            nonlocal owned
            guard.assert_idle();log=OUT/(label+'.log')
            with log.open('w') as handle:
                owned=subprocess.Popen(command,cwd=ENGINE,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                deadline=time.monotonic()+900
                while owned.poll() is None:
                    guard.assert_idle();assert time.monotonic()<deadline,label;time.sleep(.25)
            result['steps'].append(dict(label=label,command=command,exit_code=owned.returncode,log_sha256=sha256(log)));save()
            assert owned.returncode==0,(label,owned.returncode)
            print(json.dumps(dict(completed=label)),flush=True)
        def compile_to(path,obj,label):
            command=list(compile_command);command[command.index('-c')+1]=str(path);command[command.index('-o')+1]=str(obj);run(command,label)
        def link_to(obj,library,label):
            command=[str(obj) if v==original_object else v for v in link_command];command[command.index('-o')+1]=str(library);run(command,label)
            return command
        save()
        try:
            baseline_obj=OUT/'baseline-repack-x86.cpp.o';baseline_lib=OUT/'baseline-libggml-cpu.so.0.22.0'
            compile_to(source,baseline_obj,'baseline-compile');assert sha256(baseline_obj)==sha256(original_object)
            link_to(baseline_obj,baseline_lib,'baseline-link');assert sha256(baseline_lib)==runtime['cpu_sha256']
            result.update(baseline_object_identical=True,baseline_library_identical=True);save()
            obj=private/'repack-x86.cpp.o';library=private/'libggml-cpu.so.0.22.0'
            compile_to(cpp,obj,'candidate-compile');link=link_to(obj,library,'candidate-link')
            bundle=OUT/'runtime';bundle.mkdir()
            for old in Path(fixed['server']).parent.iterdir():
                if old.is_file() or old.is_symlink():
                    target=library if old.name.startswith('libggml-cpu.so') else old.resolve()
                    (bundle/old.name).symlink_to(target)
            assert all(sha256(p)==h for p,h in {**inputs,**private_sources}.items())
            manager.validate_current();guard.assert_idle()
            result.update(passed=True,library=str(library),library_sha256=sha256(library),link_command=link,
                server=str(bundle/'llama-server'),runtime_env=dict(fixed['runtime_env'],LD_LIBRARY_PATH=str(bundle)),
                base=runtime['base'],llama=runtime['llama'],common=fixed['library'],common_sha256=fixed['library_sha256'],
                peer_preserved=True,model_gain_established=False)
        except BaseException as error:
            result['error']=repr(error);raise
        finally:
            if owned is not None and owned.poll() is None:
                os.killpg(owned.pid,signal.SIGTERM)
                try:owned.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(owned.pid,signal.SIGKILL);owned.wait(timeout=10)
            result['finished']=time.time();save()


if __name__=='__main__':
    os.umask(0o077);main()
