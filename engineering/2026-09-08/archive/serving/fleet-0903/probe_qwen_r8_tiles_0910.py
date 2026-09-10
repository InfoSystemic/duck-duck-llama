#!/usr/bin/env python3
"""Exercise existing Q8 R8 output tiles on current Qwen tensor geometries."""
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from benchmark_flash_q4_selected_0910 import background
from model_measurement_guard import ModelMeasurementGuard
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import inference_snapshot, runtime_environment, sha256
from select_flash_q4_0910c import Manager

BASE=Path(__file__).resolve().parent
ENGINE=BASE.parents[1]/'engines/llama.cpp-q4e-goal-0904'
OUT=BASE/'results/qwen-r8-tiles-probe-0910'


def fixture(source):
    source=source[:source.index('int main(int argc, char ** argv) {')]
    replacements={
        'const int experts = moe ? 12 : 1;':'const int experts = moe ? 512 : 1;',
        'const int used = moe ? 8 : 1;':'const int used = moe ? 10 : 1;',
        'ggml_set_name(w, "blk.0.ffn_gate_exps.weight");':
            'ggml_set_name(w, moe ? "blk.0.ffn_down_exps.weight" : k == 1536 ? "blk.0.ssm_out.weight" : "blk.0.hc_attn_down.weight");',
        '    auto compute = [&]() {':'''    uint64_t route_phase=0;
    auto compute = [&]() {
        if (ids) {
            if (iqk) std::abort();
            std::vector<int32_t> routes(used*tokens);
            for (int t=0;t<tokens;++t) for (int j=0;j<used;++j)
                routes[t*used+j]=(j+3*t+131*route_phase)%experts;
            ggml_backend_tensor_set(ids,routes.data(),0,ggml_nbytes(ids));
            ++route_phase;
        }'''}
    for old,new in replacements.items():
        assert source.count(old)==1,old
        source=source.replace(old,new,1)
    return source+r'''
int main(int argc,char ** argv) {
    if (argc!=2) return 2;
    setenv("GGML_CPU_Q8_0_REPACK","1",1);
    setenv("GGML_CPU_Q8_0_REPACK_FORCE","1",1);
    setenv("REPACK_TEST_DOWN","1",1);
    ggml_backend_load_all();
    Dl_info runtime{};
    if (!dladdr(reinterpret_cast<void *>(ggml_backend_cpu_init),&runtime)) std::abort();
    std::printf("CPU_LIBRARY %s\n",runtime.dli_fname);
    FILE * dump=std::fopen(argv[1],"wb");
    if (!dump) return 2;
    const int threads=std::getenv("REPACK_TEST_THREADS") ? std::atoi(std::getenv("REPACK_TEST_THREADS")) : 15;
    int failures=0,cases=0;
    for (auto shape:std::vector<std::pair<int,int>>{{160,2560},{1536,2560},{10240,64},{10240,96}})
    for (int tokens:{1,3,5}) {
        const int k=shape.first,rows=shape.second;
        const bool moe=k==160;
        auto ref=run(false,k,rows,tokens,threads,moe,false,GGML_TYPE_Q8_0);
        auto got=run(true,k,rows,tokens,threads,moe,false,GGML_TYPE_Q8_0);
        bool okay=ref.values.size()==got.values.size() && ref.weight_digest==got.weight_digest;
        float max_abs=0,max_rel=0;
        for (size_t i=0;i<got.values.size();++i) {
            const float delta=std::abs(ref.values[i]-got.values[i]);
            max_abs=std::max(max_abs,delta);
            max_rel=std::max(max_rel,delta/(1+std::abs(ref.values[i])));
            okay &= std::isfinite(got.values[i]) && delta<=2e-4f*(1+std::abs(ref.values[i]));
        }
        if (std::fwrite(got.values.data(),sizeof(float),got.values.size(),dump)!=got.values.size()) return 2;
        uint64_t hash=14695981039346656037ULL;
        const auto * bytes=reinterpret_cast<const uint8_t *>(got.values.data());
        for (size_t i=0;i<got.values.size()*sizeof(float);++i) hash=(hash^bytes[i])*1099511628211ULL;
        std::printf("%s k=%d rows=%d threads=%d tokens=%d moe=%d native_ms=%.6f packed_ms=%.6f max_abs=%.8g max_scaled=%.8g weights=%016llx hash=%016llx\n",
            okay?"PASS":"FAIL",k,rows,threads,tokens,moe,ref.ms,got.ms,max_abs,max_rel,
            (unsigned long long)got.weight_digest,(unsigned long long)hash);
        std::fflush(stdout);++cases;failures+=!okay;
    }
    if (std::fclose(dump)) return 2;
    std::printf("Q8_R8_TILES cases=%d failures=%d\n",cases,failures);
    return failures?1:0;
}
'''


def main():
    assert os.sched_getaffinity(0)=={127} and not OUT.exists()
    manager=Manager();peer=manager.validate_current()
    assert peer['quant']=='UD-Q4_K_XL' and set(inference_snapshot())=={str(peer['pid'])}
    guard=ModelMeasurementGuard(peer['pid'],{peer['pid']:18131},inference_snapshot)
    runtime_path=BASE/'results/qwen-get-rows-runtime-0909/result.json';runtime=json.loads(runtime_path.read_text())
    assert runtime['passed'] and sha256(runtime['cpu'])==runtime['cpu_sha256']
    source_path=BASE/'iq2-repack-check.cpp'
    parent_path=BASE/'results/qwen-q6-q8-wide-batch-0907/private-cpu/manifest.json';parent=json.loads(parent_path.read_text())
    inputs={str(p):sha256(p) for p in [Path(__file__),source_path,runtime_path,parent_path,Path(runtime['cpu']),Path(runtime['base']),BASE/'model_measurement_guard.py']}
    for p,h in parent['private_source_sha256'].items():assert sha256(p)==h;inputs[p]=h
    with (BASE/'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guard.assert_idle()
        OUT.mkdir();cpp=OUT/'q8-r8-tiles.cpp';cpp.write_text(fixture(source_path.read_text()))
        result=dict(started=time.time(),passed=False,peer_pid=peer['pid'],input_sha256=inputs,
            fixture_sha256=sha256(cpp),runtime_cpu_sha256=runtime['cpu_sha256'],steps=[],runs=[],
            scope='Existing R8 arithmetic, tile 1/2/4/8. 512 experts, 10 routes, rotating IDs; SSM and HC dense shapes. Exact packed outputs across tiles plus native tolerance. One-socket component timings under recorded host load, not model speed or whole-server bandwidth.')
        owned=None
        def save():atomic_json(OUT/'result.json',result)
        def cancel(*_):raise InterruptedError('Stop only the owned Q8 fixture')
        for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,cancel)
        def run(command,label,env=None):
            nonlocal owned
            guard.assert_idle();log=OUT/(label+'.log')
            with log.open('w') as handle:
                owned=subprocess.Popen(command,cwd=BASE,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                deadline=time.monotonic()+900
                while owned.poll() is None:
                    guard.assert_idle();assert time.monotonic()<deadline,label;time.sleep(.25)
            result['steps'].append(dict(label=label,command=command,exit_code=owned.returncode,log_sha256=sha256(log)));save()
            assert owned.returncode==0,(label,owned.returncode)
            return log.read_text()
        save()
        try:
            binary=OUT/'q8-r8-tiles'
            command=['c++','-O3','-std=c++17','-march=native','-fopenmp']
            command+=['-I'+str(ENGINE/p) for p in ['ggml/include','ggml/src','ggml/src/ggml-cpu']]
            command += [str(cpp),'-L'+str(Path(runtime['server']).parent),'-Wl,-rpath,'+str(Path(runtime['server']).parent),
                '-lggml','-lggml-cpu','-lggml-base','-ldl','-o',str(binary)]
            run(command,'compile');result['binary_sha256']=sha256(binary)
            environment={k:v for k,v in os.environ.items() if k not in runtime_environment(os.environ) and not k.startswith('REPACK_TEST_')}
            environment.update(runtime['runtime_env'])
            environment.update(GGML_CPU_NUMA_DEVICES='0',GGML_CPU_NUMA_SHARED_DISPATCH='0',GGML_CPU_Q8_0_REPACK_TRACE='1',
                REPACK_TEST_THREADS='15',REPACK_TEST_REPEATS='31',REPACK_TEST_TIMING_MEDIAN='1',
                REPACK_TEST_PERSISTENT_POOL='1',REPACK_TEST_PIN_POOL='1')
            expected=None
            for index,tile in enumerate([1,2,4,8,8,4,2,1]):
                label=f'run-{index:02d}-tile{tile}';output=OUT/(label+'.bin')
                before=background(peer['pid'])
                log=run(['taskset','-c','48-62',str(binary),str(output)],label,dict(environment,GGML_CPU_Q8_0_REPACK_X_TILE=str(tile)))
                rows=[dict(field.split('=',1) for field in line.split()[1:]) for line in log.splitlines() if line.startswith('PASS ')]
                assert len(rows)==12 and 'Q8_R8_TILES cases=12 failures=0' in log
                loaded,=re.findall(r'^CPU_LIBRARY (.+)$',log,re.M)
                assert Path(loaded).resolve()==Path(runtime['cpu']).resolve()
                traces=re.findall(r'q8_0_r8: AVX-512/VNNI GEMV n=(\d+) nr=(\d+) nc=(\d+) x_tile=(\d+)',log)
                assert traces and all(int(t[-1])==tile for t in traces)
                digest=sha256(output)
                if expected is None:expected=digest
                assert digest==expected,'Tiling changed packed output bits'
                entry=dict(index=index,tile=tile,rows=rows,output_sha256=digest,output_bytes=output.stat().st_size,
                    exact_packed_outputs=True,kernel_trace=traces,background_before=before,background_after=background(peer['pid']))
                result['runs'].append(entry);save()
                print(json.dumps(dict(index=index,tile=tile,passed=True,exact_packed_outputs=True,
                    moe_ms=[row['packed_ms'] for row in rows if row['moe']=='1'])),flush=True)
            assert all(sha256(p)==h for p,h in inputs.items()) and sha256(cpp)==result['fixture_sha256']
            manager.validate_current();result.update(passed=True,peer_preserved=True,model_loaded=False)
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
