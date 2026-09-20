#!/usr/bin/env python3
"""Stage opt-in exact-output Q4 token batching. Does not change production."""
import difflib,hashlib,json,shlex,subprocess
from pathlib import Path
ROOT=Path('/home/user/InfoSystemic/AI-Server')
P=ROOT/'serving/fleet-0903/results/glm-flash-q8-r8-ordered-k-0908/private-cpu'
C=ROOT/'serving/fleet-0912-ctx/glm-pool-copy-0919'
D=ROOT/'serving/fleet-0912-ctx/glm-q4-batch-cpu-0919'
M=ROOT/'serving/fleet-0912-ctx/glm-q4-batch-0919'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not D.exists(),'Refuse to overwrite candidate'
assert sha(P/'repack.cpp')=='34e3beff2fc1006832682b427fb1be5e425b788dc27c826637c8d761cec63c2f'
assert sha(C/'build/libggml-cpu.so.0.22.0')=='4793379e6894a9286168f79c4f323985388ec0b0ac65e54013dcd2957488af1f'
for path,digest in json.loads((C/'build-provenance.json').read_text()).items():
 if path.endswith('.o'):assert sha(path)==digest,path
check=[json.loads(l) for l in (M/'correctness-prod.jsonl').read_text().splitlines()]
assert check[-1]['bitwise_parity'] and check[-1]['cases']==1080
D.mkdir()
original=(P/'repack.cpp').read_text();source=original
anchor='static bool flash_q8_r8_ordered_k_enabled() {'
assert source.count(anchor)==1
instrument=r'''// Private 0919 Q4 batch candidate. Behavior and counters are opt-in.
extern "C" void ggml_gemv_q4_K_x16_batch2(int, float * const *, const void *, const void * const *, int);
extern "C" void ggml_gemv_q4_K_x16_batch3(int, float * const *, const void *, const void * const *, int);
static bool flash_q4_batch_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_CPU_X16_Q4_BATCH");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}
static bool flash_q4_batch_probe_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_CPU_X16_Q4_BATCH_PROBE");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}
static std::atomic<uint64_t> flash_q4_batch_calls[4]{};
extern "C" uint64_t ggml_cpu_q4_batch_count(int nr);
extern "C" uint64_t ggml_cpu_q4_batch_count(int nr) {
    return nr >= 2 && nr <= 3 ? flash_q4_batch_calls[nr].load(std::memory_order_relaxed) : 0;
}
static void flash_q4_batch(int k, float * const *out, const void *weights, const void * const *in, int nr, int rows) {
    GGML_ASSERT(nr == 2 || nr == 3);
    if (nr == 3) ggml_gemv_q4_K_x16_batch3(k, out, weights, in, rows);
    else ggml_gemv_q4_K_x16_batch2(k, out, weights, in, rows);
    if (flash_q4_batch_probe_enabled()) flash_q4_batch_calls[nr].fetch_add(1, std::memory_order_relaxed);
}

'''
source=source.replace(anchor,instrument+anchor)
old='''                for (int64_t i11 = 0; i11 < ne11;) {
                    const bool compact_batch = sp.gemv == ggml_gemv_q5_K_x16_compact_p4_q8_K;'''
new='''                for (int64_t i11 = 0; i11 < ne11;) {
                    if (flash_q4_batch_enabled() && sp.src_type == GGML_TYPE_Q4_K &&
                            sp.dst_type == GGML_TYPE_Q4_K && i11 + 1 < ne11) {
                        const int nr = (int) std::min<int64_t>(3, ne11 - i11);
                        const void * inputs[3]; float * outputs[3];
                        for (int r = 0; r < nr; ++r) {
                            inputs[r] = wdata + ((size_t) i12 * ne11 + i11 + r) * row_bytes;
                            outputs[r] = (float *) ((char *) dst->data + (i11 + r) * dst->nb[1] + i12 * dst->nb[2]) + r0;
                        }
                        flash_q4_batch((int) k, outputs, w, inputs, nr, (int) (r1 - r0));
                        i11 += nr;
                        continue;
                    }
                    const bool compact_batch = sp.gemv == ggml_gemv_q5_K_x16_compact_p4_q8_K;'''
assert source.count(old)==1;source=source.replace(old,new)
old='''            for (int64_t ir = 0; ir < cnt; ir++) {
                const row_map m = rows[e * ne12 + ir];
                const int64_t i11 = m.i1 % ne11;
                const int64_t i12 = m.i2;
                const char * q = wdata + i11 * row_bytes + i12 * nbw2;
                float * out = (float *) ((char *) dst->data + m.i1 * dst->nb[1] + m.i2 * dst->nb[2]) + t0;
                sp.gemv((int) k, out, 0, w, q, 1, (int) tr);
            }'''
new='''            for (int64_t ir = 0; ir < cnt;) {
                if (flash_q4_batch_enabled() && sp.src_type == GGML_TYPE_Q4_K &&
                        sp.dst_type == GGML_TYPE_Q4_K && ir + 1 < cnt) {
                    const int nr = (int) std::min<int64_t>(3, cnt - ir);
                    const void * inputs[3]; float * outputs[3];
                    for (int r = 0; r < nr; ++r) {
                        const row_map m = rows[e * ne12 + ir + r];
                        inputs[r] = wdata + (m.i1 % ne11) * row_bytes + m.i2 * nbw2;
                        outputs[r] = (float *) ((char *) dst->data + m.i1 * dst->nb[1] + m.i2 * dst->nb[2]) + t0;
                    }
                    flash_q4_batch((int) k, outputs, w, inputs, nr, (int) tr);
                    ir += nr;
                    continue;
                }
                const row_map m = rows[e * ne12 + ir];
                const int64_t i11 = m.i1 % ne11;
                const int64_t i12 = m.i2;
                const char * q = wdata + i11 * row_bytes + i12 * nbw2;
                float * out = (float *) ((char *) dst->data + m.i1 * dst->nb[1] + m.i2 * dst->nb[2]) + t0;
                sp.gemv((int) k, out, 0, w, q, 1, (int) tr);
                ++ir;
            }'''
assert source.count(old)==1;source=source.replace(old,new)
(D/'repack.cpp').write_text(source)
(D/'candidate.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),source.splitlines(True),fromfile=str(P/'repack.cpp'),tofile=str(D/'repack.cpp'))))
kernel=(M/'q4_batch.cpp').read_text().replace('q4_batch2(', 'ggml_gemv_q4_K_x16_batch2(').replace('q4_batch3(', 'ggml_gemv_q4_K_x16_batch3(')
(D/'q4_batch.cpp').write_text(kernel)
parent=json.loads((P/'manifest.json').read_text())
compile=list(parent['compile_command']);compile[compile.index('-o')+1]=str(D/'repack.cpp.o');compile[compile.index('-c')+1]=str(D/'repack.cpp')
kcompile=list(compile);kcompile[kcompile.index('-o')+1]=str(D/'q4_batch.cpp.o');kcompile[kcompile.index('-c')+1]=str(D/'q4_batch.cpp')
link_lines=[line for line in (C/'rebuild.sh').read_text().splitlines() if line.startswith('/usr/bin/c++ ') and ' -shared ' in line]
assert len(link_lines)==1
link=shlex.split(link_lines[0].replace('"$OUT"',shlex.quote(str(C/'build'))))
original_objects=[Path(a) for a in link if a.endswith('.o')];assert len(original_objects)==15
parent_link=list(link);parent_link[parent_link.index('-o')+1]=str(D/'parent-link.so')
subprocess.run(parent_link,check=True);assert sha(D/'parent-link.so')==sha(C/'build/libggml-cpu.so.0.22.0')
link[link.index('-o')+1]=str(D/'libggml-cpu.so.0.22.0');link[link.index(str(P/'repack.cpp.o'))]=str(D/'repack.cpp.o')
link.insert(link.index(str(D/'repack.cpp.o'))+1,str(D/'q4_batch.cpp.o'))
manifest={'status':'BUILDING','production_changed':False,'parent_library_sha256':sha(C/'build/libggml-cpu.so.0.22.0'),'parent_link_identical':True,'original_source':str(P/'repack.cpp'),'original_source_sha256':sha(P/'repack.cpp'),'candidate_source_sha256':sha(D/'repack.cpp'),'kernel_source_sha256':sha(D/'q4_batch.cpp'),'compile_command':compile,'kernel_compile_command':kcompile,'link_command':link,'frozen_objects':{str(p):sha(p) for p in original_objects},'activation':'GGML_CPU_X16_Q4_BATCH=1; default off','counter':'GGML_CPU_X16_Q4_BATCH_PROBE=1; default off; ggml_cpu_q4_batch_count(nr)','scope':['Q4_K x16 ordinary MUL_MAT','Q4_K x16 ordinary MUL_MAT_ID'],'excluded':['Q4/Q5 clamped fusion candidate','other quant types','single-token dispatch','already-fused gate/up paths']}
print('Parent relink identical. Building isolated Q4 batch candidate.',flush=True)
try:
 subprocess.run(compile,check=True);subprocess.run(kcompile,check=True);subprocess.run(link,check=True)
 (D/'libggml-cpu.so.0').symlink_to('libggml-cpu.so.0.22.0');(D/'libggml-cpu.so').symlink_to('libggml-cpu.so.0')
 manifest.update(status='BUILT, NOT VALIDATED OR DEPLOYED',library_sha256=sha(D/'libggml-cpu.so.0.22.0'))
 print('Built',manifest['library_sha256'],flush=True)
except BaseException as exc:manifest['error']=repr(exc);raise
finally:(D/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
