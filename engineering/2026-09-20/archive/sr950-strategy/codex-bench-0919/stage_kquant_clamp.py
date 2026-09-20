#!/usr/bin/env python3
"""Build an isolated opt-in x16 Q4/Q5 clamped-expert candidate; no service changes."""
import difflib, hashlib, json, shlex, subprocess
from pathlib import Path
ROOT=Path('/home/user/InfoSystemic/AI-Server')
P=ROOT/'serving/fleet-0903/results/glm-flash-q8-r8-ordered-k-0908/private-cpu'
C=ROOT/'serving/fleet-0912-ctx/glm-pool-copy-0919'
D=ROOT/'serving/fleet-0912-ctx/glm-kquant-clamp-0919'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not D.exists(), 'Refuse to overwrite a candidate'
assert sha(P/'repack.cpp')=='34e3beff2fc1006832682b427fb1be5e425b788dc27c826637c8d761cec63c2f'
assert sha(C/'build/libggml-cpu.so.0.22.0')=='4793379e6894a9286168f79c4f323985388ec0b0ac65e54013dcd2957488af1f'
provenance=json.loads((C/'build-provenance.json').read_text())
for path,digest in provenance.items():
 if path.endswith('.o'):assert sha(path)==digest, path
parent=json.loads((P/'manifest.json').read_text())
D.mkdir()
original=(P/'repack.cpp').read_text()
source=original
anchor='static bool flash_q8_r8_ordered_k_enabled() {'
assert source.count(anchor)==1
instrument=r'''// Private 0919 experiment. Both behavior and counters are off unless requested.
static bool flash_kquant_clamp_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_CPU_KQUANT_CLAMP_FUSION");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}
static bool flash_kquant_clamp_probe_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_CPU_KQUANT_CLAMP_PROBE");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}
static std::atomic<uint64_t> flash_kquant_clamp_calls[GGML_TYPE_COUNT]{};
extern "C" uint64_t ggml_cpu_kquant_clamp_fusion_count(int type);
extern "C" uint64_t ggml_cpu_kquant_clamp_fusion_count(int type) {
    return type >= 0 && type < GGML_TYPE_COUNT
        ? flash_kquant_clamp_calls[type].load(std::memory_order_relaxed) : 0;
}

'''
source=source.replace(anchor,instrument+anchor)
old=r'''        if (!enabled || sp.dst_type != GGML_TYPE_Q8_0 ||
                gate_clamp->op != GGML_OP_CLAMP || up_clamp->op != GGML_OP_CLAMP) {
            return false;
        }
        const bool okay = forward_x16_moe_swiglu(
            params, gate_clamp->src[0], up_clamp->src[0], dst, gate_clamp, up_clamp);
        static std::atomic<bool> logged{false};
        if (okay && params->ith == 0 && !logged.exchange(true, std::memory_order_relaxed)) {
            GGML_LOG_INFO("Q8_X16_CLAMP_FUSION_ACTIVE\n");
        }
        return okay;'''
new=r'''        const bool q8_enabled = enabled && sp.dst_type == GGML_TYPE_Q8_0;
        const bool kquant_enabled = flash_kquant_clamp_enabled() &&
            (sp.dst_type == GGML_TYPE_Q4_K || sp.dst_type == GGML_TYPE_Q5_K);
        if ((!q8_enabled && !kquant_enabled) ||
                gate_clamp->op != GGML_OP_CLAMP || up_clamp->op != GGML_OP_CLAMP) {
            return false;
        }
        if (kquant_enabled) {
            const ggml_tensor * gate = gate_clamp->src[0];
            const ggml_tensor * up = up_clamp->src[0];
            // Only unchanged Q4/Q5 quantization and the supported expert table.
            if (gate->op != GGML_OP_MUL_MAT_ID || up->op != GGML_OP_MUL_MAT_ID ||
                    gate->src[0]->type != sp.dst_type || up->src[0]->type != sp.dst_type ||
                    gate->src[0]->ne[2] > 512 || up->src[0]->ne[2] > 512) {
                return false;
            }
        }
        const bool okay = forward_x16_moe_swiglu(
            params, gate_clamp->src[0], up_clamp->src[0], dst, gate_clamp, up_clamp);
        static std::atomic<bool> logged{false};
        if (okay && q8_enabled && params->ith == 0 && !logged.exchange(true, std::memory_order_relaxed)) {
            GGML_LOG_INFO("Q8_X16_CLAMP_FUSION_ACTIVE\n");
        }
        if (okay && kquant_enabled && params->ith == 0 && flash_kquant_clamp_probe_enabled()) {
            flash_kquant_clamp_calls[sp.dst_type].fetch_add(1, std::memory_order_relaxed);
        }
        return okay;'''
assert source.count(old)==1
source=source.replace(old,new)
(D/'repack.cpp').write_text(source)
(D/'candidate.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),source.splitlines(True),fromfile=str(P/'repack.cpp'),tofile=str(D/'repack.cpp'))))
compile=list(parent['compile_command'])
compile[compile.index('-o')+1]=str(D/'repack.cpp.o')
compile[compile.index('-c')+1]=str(D/'repack.cpp')
link_lines=[line for line in (C/'rebuild.sh').read_text().splitlines() if line.startswith('/usr/bin/c++ ') and ' -shared ' in line]
assert len(link_lines)==1
link=shlex.split(link_lines[0].replace('"$OUT"',shlex.quote(str(C/'build'))))
original_objects=[Path(a) for a in link if a.endswith('.o')]
assert len(original_objects)==15
assert str(C/'build/ggml-cpu.c.o') in link and str(C/'build/ops.cpp.o') in link
assert str(P/'repack.cpp.o') in link
parent_link=list(link);parent_link[parent_link.index('-o')+1]=str(D/'parent-link.so')
subprocess.run(parent_link,check=True)
assert sha(D/'parent-link.so')==sha(C/'build/libggml-cpu.so.0.22.0'), 'Parent relink must be byte-identical'
link[link.index('-o')+1]=str(D/'libggml-cpu.so.0.22.0')
link[link.index(str(P/'repack.cpp.o'))]=str(D/'repack.cpp.o')
manifest={'status':'BUILDING','production_changed':False,'parent_library_sha256':sha(C/'build/libggml-cpu.so.0.22.0'),
 'parent_link_identical':True,'original_source':str(P/'repack.cpp'),'original_source_sha256':sha(P/'repack.cpp'),
 'candidate_source_sha256':sha(D/'repack.cpp'),'compile_command':compile,'link_command':link,
 'frozen_objects':{str(p):sha(p) for p in original_objects},
 'activation':'GGML_CPU_KQUANT_CLAMP_FUSION=1; default off',
 'counter':'GGML_CPU_KQUANT_CLAMP_PROBE=1; default off; exported ggml_cpu_kquant_clamp_fusion_count(type)'}
(D/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Parent relink is byte-identical. Compiling isolated Q4/Q5 x16 candidate.',flush=True)
try:
 subprocess.run(compile,check=True)
 subprocess.run(link,check=True)
 (D/'libggml-cpu.so.0').symlink_to('libggml-cpu.so.0.22.0')
 (D/'libggml-cpu.so').symlink_to('libggml-cpu.so.0')
 manifest.update(status='BUILT, NOT VALIDATED OR DEPLOYED',library_sha256=sha(D/'libggml-cpu.so.0.22.0'))
 print('Candidate built:',manifest['library_sha256'],flush=True)
except BaseException as exc:
 manifest['error']=repr(exc);raise
finally:(D/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
