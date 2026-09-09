#!/usr/bin/env python3
"""Record the measured narrow Q8 projection runtime and its RMS rollback."""
import fcntl
import json
import os
from pathlib import Path
import time

from glm_flash_q8_trial import BASE, Manager, PORT, memory_status
from model_measurement_guard import ModelMeasurementGuard, read_service
from qwen_split_trial import inference_snapshot, sha256


def main():
    os.umask(0o077)
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manager = Manager()
        current = manager.validate_current()
        guard = ModelMeasurementGuard(current['pid'], {current['pid']: PORT}, inference_snapshot)
        guard.assert_idle()
        configuration, _ = manager.candidate(True, 0, 15, q8_experts=True,
                                            q8_clamp=True, pooling=True, sum16=True, rms_guard=True, ordered_k=True)
        assert all(current[key] == value for key, value in configuration.items())
        assert not manager.qwen.state.get('current') and manager.qwen.state['full_stopped']
        assert any('UD-Q6_K_XL' in value for value in manager.state['qwen_rollback']['command'])
        assert '--no-cache-prompt' in current['command']
        assert not any('AUDIT' in key or 'PROFILE' in key for key in current['runtime_env'])
        preflight_path = BASE / 'results/glm-flash-q8-r8-ordered-k-control-0908/preflight.json'
        preflight = json.loads(preflight_path.read_text())
        assert preflight['passed'] and preflight['candidate'] == configuration
        assert preflight['controller_sha256'] == sha256(BASE / 'glm_flash_q8_trial.py')
        profile_path = BASE / 'results/glm-flash-q8-r8-ordered-k-profile-0908/analysis/result.json'
        profile = json.loads(profile_path.read_text())
        assert profile['passed'] and profile['pid'] == current['pid']
        assert profile['cpu_sha256'] == current['cpu_sha256'] and profile['lost_records'] == 0
        assert profile['sample_counts']['q8_r8_ordered_vector'] > 0
        assert all(sha256(path) == digest for path, digest in profile['source_sha256'].items())
        evidence = [Path(__file__), BASE / 'glm_flash_q8_trial.py', preflight_path, profile_path]
        rows = []
        for label in ('glm-flash-q8-r8-ordered-k-raw-0908', 'glm-flash-q8-r8-ordered-k-raw-repeat-0908'):
            controller_path = BASE / 'results' / (label + '-controller.json')
            comparison_path = BASE / 'results' / (label + '-comparison.json')
            controller = json.loads(controller_path.read_text())
            comparison = json.loads(comparison_path.read_text())
            assert controller['passed'] and controller['current'] == current
            assert sha256(controller['measurement']) == controller['measurement_sha256']
            assert comparison['input_comparison_passed'] and comparison['all_streams_equal']
            assert all(sha256(path) == digest for path, digest in comparison['source_sha256'].items())
            for row in comparison['rows']:
                assert all(sha256(path) == digest for path, digest in row['stream_sha256'].items())
                if 'glm-flash-q8-r8-ordered-k-control-raw-0908' in row['reference']:
                    rows.append(dict(label=label, **row))
            evidence += [controller_path, comparison_path, Path(controller['measurement'])]
        assert len(rows) == 4 and all(row['speed_change_percent'] > 0 for row in rows)
        mappings = Path(f"/proc/{current['pid']}/maps").read_text().splitlines()
        expected = {
            'libggml-cpu.so.': Path(current['cpu_library']),
            'libllama.so.': Path(current['pinned_directory']) / 'libllama.so.0.3.0',
            'libggml-base.so.': Path(current['pinned_directory']) / 'libggml-base.so.0.22.0',
        }
        mapped = {}
        for stem, wanted in expected.items():
            actual = {line.split()[-1] for line in mappings if '/' + stem in line}
            assert actual == {str(wanted.resolve())}
            mapped[stem] = dict(path=str(wanted.resolve()), sha256=sha256(wanted))
        command = ['python3', str(BASE / 'glm_flash_q8_trial.py'), 'launch', '--drafts', '0',
                   '--workers', '15', '--batching', '--q8-experts', '--q8-clamp', '--pooling', '--sum16', '--rms-guard']
        result = dict(time=time.time(), passed=True, decision='retain_for_raw_tuning', current=current,
                      source_sha256={str(path): sha256(path) for path in evidence},
                      fresh_control_comparisons=rows, mapped_libraries=mapped,
                      profile_sample_count=profile['sample_count'],
                      profile_period_percent=profile['period_percent'],
                      memory=memory_status(), service=read_service(PORT),
                      selected_qwen_q6_saved=True, qwen_unloaded=True, full_stopped=True,
                      launch_command=command + ['--ordered-k'], rollback_command=command,
                      target_gb_s=285, capacity_gb_s=380,
                      target_reached=all(row['candidate_adjusted_gb_s'] >= 285 for row in rows),
                      temporary_storage=True, reference_quality_qualified=False,
                      scope='Two isolated short-context decode pairs and bounded output equivalence. '
                            'A fresh preceding control is available; controls are not interleaved. '
                            'This is not a throughput ceiling or a source-checkpoint quality certification.')
        manager.validate_current()
        guard.assert_idle()
        output = BASE / 'results/glm-flash-q8-r8-ordered-k-post-model-0908.json'
        assert not output.exists()
        output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(passed=True, decision=result['decision'], pid=current['pid'],
                              memory_available_gb=result['memory']['MemAvailable'] / 1e9,
                              target_reached=result['target_reached'], output=str(output))), flush=True)


if __name__ == '__main__':
    main()
