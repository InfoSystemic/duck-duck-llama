#!/usr/bin/env python3
"""Check Q6/Q8 expert geometry and numerical outputs; make no speed claim."""
import array
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot, sha256

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/qwen-q6-expert-validation-0907'
ENGINE = BASE.parents[1] / 'engines/llama.cpp-q4e-goal-0904'
BIN = ENGINE / 'validated-iq-batch3-bin'


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--label', default='qwen-q6-expert-validation-0907')
    parser.add_argument('--experts', type=int, choices=(32, 512), default=32)
    parser.add_argument('--cpu-manifest', type=Path)
    parser.add_argument('--x16-q8-experts', action='store_true')
    parser.add_argument('--q8', action='store_true', help='Exercise Q8 gate/up and MTP geometry')
    args = parser.parse_args()
    assert '/' not in args.label and args.label not in ('.', '..')
    OUT = BASE / 'results' / args.label
    OUT.mkdir(exist_ok=args.resume)
    plan = json.loads((BASE / 'results/qwen-even-split-model-trial-0906-staging/plan.json').read_text())
    candidate = Path(plan['candidate_library'])
    assert sha256(candidate) == plan['candidate_library_sha256']
    inventory = json.loads((BASE / 'results/qwen-higher-quant-bandwidth-inventory-0907.json').read_text())
    q6 = next(model for model in inventory['models'] if model['quant'] == 'UD-Q6_K_XL')
    experts = [record for record in q6['records'] if record['routed']]
    assert len(experts) == 144
    q8_layers = []
    for layer in range(48):
        group = {kind: next(record for record in experts if record['name'] ==
                 f'blk.{layer}.ffn_{kind}_exps.weight') for kind in ('gate', 'up', 'down')}
        assert group['down']['type'] == 'Q8_0'
        assert group['gate']['type'] == group['up']['type']
        assert group['gate']['type'] in ('Q6_K', 'Q8_0')
        if group['gate']['type'] == 'Q8_0':
            q8_layers.append(layer)
    result = dict(started=time.time(), passed=False, candidate_sha256=sha256(candidate), experts=args.experts,
                  x16_q8_experts=args.x16_q8_experts, weight_type='q8_0' if args.q8 else 'q6_K',
                  q8_gate_up_layers=q8_layers, geometry=[], numeric=[],
                  scope='Functional component validation; no performance claim')
    def save():
        (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    trial = json.loads((BASE / 'results/qwen-q6-trial-0907/state.json').read_text())
    pid = trial['current']['pid']
    ports = {pid: 18095}
    if not trial['full_stopped']:
        ports[4005448] = 18091
    guard = ModelMeasurementGuard(pid, ports, inference_snapshot)
    guard.wait_idle(OUT / 'waiting-for-idle.json')
    environment = {k: v for k, v in os.environ.items() if not k.startswith(('GGML_', 'LLAMA_', 'OMP_', 'GOMP_'))}
    environment.update(plan['original_environment'])
    environment.update(OMP_NUM_THREADS='1', COLD_GRAPH_K_PER_SOCKET='2560',
                       COLD_GRAPH_ROWS='640', COLD_GRAPH_MATRICES='4')
    inputs = [BASE / 'qwen-expert-graph-check.cpp', BASE / 'qwen-expert-split-check.cpp', candidate]
    inputs += list({(BIN / name).resolve() for name in plan['baseline_binary_sha256']})
    cpu_prefix = ''
    if args.cpu_manifest:
        cpu = json.loads(args.cpu_manifest.read_text())
        assert sha256(cpu['library']) == cpu['library_sha256']
        cpu_prefix = str(Path(cpu['library']).parent) + ':'
        result.update(cpu_manifest=str(args.cpu_manifest), cpu_sha256=cpu['library_sha256'])
        inputs += [args.cpu_manifest, Path(cpu['library'])]
    if args.x16_q8_experts:
        assert args.cpu_manifest and cpu['experts']
        environment['GGML_CPU_X16_Q8_EXPERTS'] = '1'
    result['input_sha256'] = {str(path): sha256(path) for path in inputs}
    save()

    def run(command, label, env, input_text=None):
        guard.assert_idle()
        log_path = OUT / (label + '.log')
        with log_path.open('w') as log:
            proc = subprocess.Popen(command, env=env, stdin=subprocess.PIPE if input_text else subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                if input_text:
                    proc.stdin.write(input_text.encode())
                    proc.stdin.close()
                deadline = time.monotonic() + 300
                while proc.poll() is None:
                    guard.assert_idle()
                    assert time.monotonic() < deadline, label
                    time.sleep(0.5)
                assert proc.returncode == 0, (label, proc.returncode, str(log_path))
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=10)
        print(json.dumps({'completed': label}), flush=True)
        return [json.loads(line) for line in log_path.read_text().splitlines() if line.startswith('{')]

    def compile_source(source, output, definitions=()):
        command = ['g++', '-O2', '-std=c++17', '-pthread', *definitions, str(source)]
        command += ['-I' + str(ENGINE / path) for path in
                    ('include', 'src', 'ggml/include', 'ggml/src', 'ggml/src/ggml-cpu')]
        command += ['-L' + str(BIN), '-lllama', '-lggml', '-lggml-cpu', '-lggml-base',
                    '-ldl', '-o', str(output)]
        run(command, 'build-' + output.name, environment)

    try:
        geometry = (BASE / 'qwen-expert-split-check.cpp').read_text()
        old = 'down ? GGML_TYPE_IQ4_NL : (layer == 2 ? GGML_TYPE_IQ3_XXS : GGML_TYPE_IQ2_XS)'
        assert geometry.count(old) == 1
        condition = ' || '.join('layer == ' + str(layer) for layer in q8_layers) or 'false'
        geometry = geometry.replace(old, f'(down || {condition}) ? GGML_TYPE_Q8_0 : GGML_TYPE_Q6_K')
        source = OUT / 'geometry.cpp'
        source.write_text(geometry)
        compile_source(source, OUT / 'geometry')
        for enabled in (0, 1):
            env = dict(environment, LD_LIBRARY_PATH=cpu_prefix + str(candidate.parent) + ':' + str(BIN),
                       GGML_Q4E_EXPERT_EVEN_SPLIT=str(enabled))
            events = run([str(OUT / 'geometry')], 'geometry-' + str(enabled), env)
            splits = [event for event in events if event['event'] == 'split']
            assert len(splits) == 144
            assert all(sorted(item['slices']) == ([160] * 4 if enabled else [128, 128, 128, 256])
                       for item in splits)
            result['geometry'].append(dict(even_split=bool(enabled), splits=splits))
            save()

        graph = (BASE / 'qwen-expert-graph-check.cpp').read_text()
        old = '#if defined(QWEN_Q8_CHECK)\nusing qwen_weight_block = block_q8_0;'
        assert graph.count(old) == 1
        graph = graph.replace(old, '#if defined(QWEN_Q6_CHECK)\nusing qwen_weight_block = block_q6_K;\n'
                              'static constexpr ggml_type qwen_weight_type = GGML_TYPE_Q6_K;\n'
                              '#elif defined(QWEN_Q8_CHECK)\nusing qwen_weight_block = block_q8_0;')
        old = '#if defined(QWEN_Q8_CHECK)\nusing qwen_activation_block = block_q8_0;'
        assert graph.count(old) == 1
        graph = graph.replace(old, '#if defined(QWEN_Q6_CHECK)\nusing qwen_activation_block = block_q8_K;\n'
            'using qwen_down_block = block_q8_0;\nstatic constexpr ggml_type qwen_activation_type = GGML_TYPE_Q8_K;\n'
            'static constexpr ggml_type qwen_down_type = GGML_TYPE_Q8_0;\nstatic constexpr bool draft_geometry = false;\n'
            '#elif defined(QWEN_Q8_CHECK)\nusing qwen_activation_block = block_q8_0;')
        old = '#if defined(QWEN_Q8_CHECK)\n            down_templates[i].d'
        assert graph.count(old) == 1
        graph = graph.replace(old, '#if defined(QWEN_Q8_CHECK) || defined(QWEN_Q6_CHECK)\n            down_templates[i].d')
        assert graph.count('packed_block_bytes = 2208') == 1
        graph = graph.replace('packed_block_bytes = 2208', 'packed_block_bytes = 16 * sizeof(block_q6_K)')
        if args.experts == 512:
            assert graph.count('constexpr int experts = 32') == 1
            graph = graph.replace('constexpr int experts = 32', 'constexpr int experts = 512')
            assert graph.count('tokens > 3') == 1
            graph = graph.replace('tokens > 3', 'tokens > 64')
            old = '(u + 2 * t + 3 * m + probe) % experts'
            assert graph.count(old) == 1
            graph = graph.replace(old, '(u + 10 * t + (m % 2 ? 490 : 250) + probe) % experts')
        source = OUT / 'numeric.cpp'
        source.write_text(graph)
        compile_source(source, OUT / 'numeric', ['-DQWEN_MOE_CHECK', '-DQWEN_Q8_CHECK' if args.q8 else '-DQWEN_Q6_CHECK'])
        for tokens in ((1, 3, 64) if args.experts == 512 else (1, 3)):
            outputs = []
            for enabled in (0, 1):
                path = OUT / f'numeric-{tokens}-{enabled}.f32'
                env = dict(environment, LD_LIBRARY_PATH=cpu_prefix + str(candidate.parent) + ':' + str(BIN),
                    GGML_Q4E_EXPERT_EVEN_SPLIT=str(enabled), COLD_GRAPH_TOKENS=str(tokens),
                    COLD_GRAPH_OUTPUT_PATH=str(path))
                events = run(['taskset', '-c', '0-14,16-30,32-46,48-62', str(OUT / 'numeric'),
                              '32' if enabled else '128', '15', 'fixture', '3'],
                             f'numeric-{tokens}-{enabled}', env, 'exit\n')
                ready = next(event for event in events if event['event'] == 'ready')
                assert ready['weight_type'] == result['weight_type'] and ready['down_checked']
                values = array.array('f')
                values.frombytes(path.read_bytes())
                assert len(values) == 3 * 4 * 10 * tokens * 2560
                outputs.append(values)
                result['numeric'].append(dict(tokens=tokens, even_split=bool(enabled), reference=ready))
                save()
            errors = [abs(x - y) / (1 + abs(x)) for x, y in zip(*outputs)]
            assert all(math.isfinite(value) for value in errors)
            assert max(errors) <= 4e-5, max(errors)
            result.setdefault('cross_layout', []).append(dict(tokens=tokens, values=len(errors),
                                                               max_scaled_error=max(errors)))
            save()
        if args.experts == 512:
            path = OUT / 'numeric-64-unfused.f32'
            env = dict(environment, LD_LIBRARY_PATH=cpu_prefix + str(candidate.parent) + ':' + str(BIN),
                       GGML_Q4E_EXPERT_EVEN_SPLIT='1', GGML_CPU_MOE_GATE_UP_FUSION='0',
                       COLD_GRAPH_TOKENS='64', COLD_GRAPH_OUTPUT_PATH=str(path))
            events = run(['taskset', '-c', '0-14,16-30,32-46,48-62', str(OUT / 'numeric'),
                          '32', '15', 'fixture', '3'], 'numeric-64-unfused', env, 'exit\n')
            ready = next(event for event in events if event['event'] == 'ready')
            assert ready['experts'] == 512 and ready['down_checked']
            result['unfused_prefill'] = ready
            save()
        assert all(sha256(path) == digest for path, digest in result['input_sha256'].items())
        guard.assert_idle()
        result['passed'] = True
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        result['finished'] = time.time()
        save()


if __name__ == '__main__':
    main()
