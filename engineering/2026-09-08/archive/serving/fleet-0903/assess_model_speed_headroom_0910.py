#!/usr/bin/env python3
"""Record measured speed provenance and conditional budgets; no inference or tuning."""
import hashlib
import json
from pathlib import Path
import statistics
import time

BASE = Path('/home/user/InfoSystemic/AI-Server/serving/fleet-0903')
SOURCES = {}


def read(relative):
    path = BASE / relative
    assert '.private.' not in str(path)
    data = path.read_bytes()
    SOURCES[str(path)] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def measurement(relative):
    data = read(relative)
    assert data['finished'] and data['input_integrity_verified'] and not data.get('error')
    result = {}
    for row in data['measurements']:
        assert not row['abort'] and row['timings']['cache_n'] == 0
        assert all(row[k]['valid'] for k in ('decode', 'baseline_before', 'baseline_after'))
        result[row['kind']] = dict(tok_s=row['timings']['predicted_per_second'],
            adjusted_gb_s=row['background_subtracted_gb_s'],
            generated_tokens=row['timings']['predicted_n'],
            drafted_tokens=row['timings'].get('draft_n', 0),
            accepted_drafts=row['timings'].get('draft_n_accepted', 0),
            source=relative)
    return result


def main():
    destination = BASE / 'results/model-speed-headroom-0910.json'
    note = BASE / 'MODEL-SPEED-HEADROOM-20260910.md'
    assert not destination.exists() and not note.exists()
    q4 = read('results/glm-flash-q4-measurement-audit-0910.json')
    assert q4['passed'] and q4['all_48_counters_reparsed']
    scheduling = read('results/qwen-decode-scheduling-model-0909b/result.json')
    assert scheduling['passed'] and scheduling['all_outputs_match']
    peak_qwen = measurement('results/qwen-private-decode-scheduling-250-0909b-2-candidate-decode/result.json')
    full = measurement('results/glm53-full-current-bandwidth-0906/result.json')
    q8 = [measurement('results/flash-bandwidth250-comparison-0909-' + label + '/result.json')
          for label in ('04-mtp-control', '05-mtp-candidate', '06-mtp-candidate', '07-mtp-control')]
    earlier_full = read('results/glm53-full-bandwidth-baseline-0905c/result.json')
    earlier_full_prose = next(row for row in earlier_full['measurements']
        if row['kind'] == 'prose' and row['draft_n'] == 2)
    assert not earlier_full_prose['abort'] and earlier_full_prose['timings']['predicted_n'] == 512
    full_edit = read('results/glm53-full-replay-bandwidth-0906/result.json')
    passing_edit_rates = [row['timings']['predicted_per_second'] for row in full_edit['measurements']
                         if row['quality']['passed'] and not row['abort']]
    old_flash = read('results/glm5n-goal-x16-chunk16-1024-t15/result.json')
    old_flash_long = [row['response']['timings']['predicted_per_second'] for row in old_flash['checks']
                     if row['response']['timings']['predicted_n'] >= 128 and not row['throughput_contended']]
    ops = read('results/qwen-shared-dispatch-ops-analysis-0909.json')
    quant = read('results/q4-switch-assessment-0910.json')
    assert quant['passed']
    bases = {}
    for kind in ('prose', 'code'):
        bases.setdefault('GLM Flash Q4 MTP2', {})[kind] = dict(
            tok_s=q4['summaries'][kind]['mean_tok_s'],
            adjusted_gb_s=statistics.mean(run['rows'][kind]['adjusted_gb_s'] for run in q4['runs']))
        summary = next(row for row in scheduling['summaries'] if row['workload'] == kind)
        bases.setdefault('Qwen Flash-Next Q6 MTP4', {})[kind] = dict(
            tok_s=summary['parent_mean_tok_s'], adjusted_gb_s=summary['parent_mean_gb_s'])
        bases.setdefault('GLM Full mixed Q4 MTP2', {})[kind] = full[kind]
    projections = {}
    for model, rows in bases.items():
        projections[model] = {}
        for kind, row in rows.items():
            effective_bytes = row['adjusted_gb_s'] / row['tok_s']
            projections[model][kind] = dict(basis=row, approximate_gb_per_generated_token=effective_bytes,
                tok_s_if_all_costs_scale={str(bandwidth): bandwidth / effective_bytes
                    for bandwidth in (250, 323, 353.4, 380)})
    qwen_budget = {}
    for kind, row in bases['Qwen Flash-Next Q6 MTP4'].items():
        peak = peak_qwen[kind]['tok_s']
        qwen_budget[kind] = dict(target_tok_s=40, target_ms_per_token=25,
            repeated_parent_ms_per_token=1000 / row['tok_s'],
            required_speedup_percent_from_repeated_parent=(40 / row['tok_s'] - 1) * 100,
            required_elapsed_time_reduction_percent_from_repeated_parent=(1 - row['tok_s'] / 40) * 100,
            required_speedup_percent_from_peak=(40 / peak - 1) * 100,
            conditional_gb_s_at_40=row['adjusted_gb_s'] * 40 / row['tok_s'])
    peaks = dict(qwen_q6=peak_qwen,
        flash_q4={kind: max(run['rows'][kind]['tok_s'] for run in q4['runs']) for kind in ('prose', 'code')},
        flash_q8={kind: max(run[kind]['tok_s'] for run in q8) for kind in ('prose', 'code')},
        flash_historical_iq2_completed_or_long=max(old_flash_long),
        full_latest=full,
        full_earlier_single_prose=earlier_full_prose['timings']['predicted_per_second'],
        full_specialized_passing_edit=max(passing_edit_rates))
    result = dict(passed=True, time=time.time(), capacity_assumed_gb_s=380,
        scope='Assessment of saved evidence, not new inference, a measured hardware ceiling, or an all-history search of short checks.',
        measured=peaks, conditional_projections=projections, qwen_40_budget=qwen_budget,
        qwen_target_precision='UD-Q6_K_XL', qwen_40_target_reached=False,
        whole_model_250_goal_complete=False,
        projection_limitations=[
            'Requires similar physical traffic per generated token, draft acceptance, and output workload.',
            'Assumes other execution costs fall enough to sustain the stated average DRAM rate.',
            'Uses stable-window DRAM averages and overall reported decode rates; the inferred bytes/token are approximate.',
            'Flash Q4 was measured with 15-18 background CPU cores; this is not an uncontended ceiling.',
            'Systemwide counters minus adjacent idle traffic estimate model attribution.',
            'The 380 GB/s denominator is the stated planning capacity, not a new capacity measurement.'],
        qwen_profile=ops['rows'], source_sha256=SOURCES)
    result['source_sha256'][str(Path(__file__))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    destination.write_text(json.dumps(result, indent=2) + '\n')
    lines = [
        '# Model speed and remaining headroom, September 10, 2026 UTC', '',
        'The user now expects Qwen3.8-Flash-Next to exceed 40 generated tok/s. '
        'Keep the selected Q6 precision as the engineering target. The existing all-model '
        '250 GB/s request remains incomplete. These are single-conversation decode rates, '
        'including reasoning; prefill, aggregate concurrency, and short arithmetic checks are excluded.', '',
        '| Model/configuration | Highest saved prose tok/s | Highest saved code tok/s | Qualification |',
        '| --- | ---: | ---: | --- |',
        f"| Qwen Flash-Next Q6, experimental MTP4 | {peak_qwen['prose']['tok_s']:.2f} | {peak_qwen['code']['tok_s']:.2f} | One best candidate run; repeated parent means 24.52/31.80, candidate means 24.37/31.84; no consistent scheduling gain or promotion |",
        f"| Flash Q4, selected Q8 MTP2 | {peaks['flash_q4']['prose']:.2f} | {peaks['flash_q4']['code']:.2f} | Two matching-output repeats under recorded background CPU load; means 15.85/15.81 |",
        f"| Flash Q8, historical MTP2 | {peaks['flash_q8']['prose']:.2f} | {peaks['flash_q8']['code']:.2f} | Maxima across control/candidate arms, not one superior configuration |",
        f"| Full mixed Q4, latest MTP2 evidence | {full['prose']['tok_s']:.2f} | {full['code']['tok_s']:.2f} | Latest fully recorded fresh prose/code pair; earlier single prose run 8.50 |", '',
        f"Older Flash IQ2 long/direct outputs reached {max(old_flash_long):.2f} tok/s. "
        f"Full reached {max(passing_edit_rates):.2f} tok/s on a passing specialized copy/edit replay with deeper speculation. "
        'Those are different quantizations or workloads and do not establish the selected configurations\' general speed.', '',
        'Current Flash Q4 uses 178.5-184.0 GB/s after adjacent-idle subtraction, around '
        '47-48% of the stated capacity. Qwen experimental MTP4 runs use roughly 139-153 GB/s, '
        '37-40%; Full latest MTP2 uses 197-199 GB/s, about 52%. Historical Flash Q8 raw '
        'decode reached 241.37 GB/s (63.5%) while emitting fewer tokens than MTP2. '
        'All four sockets contribute similar traffic in the examined counter windows. '
        'A missing memory socket does not explain these gaps.', '',
        'Qwen 40 tok/s means 25 ms per generated token. Compared with repeated parent '
        'means, prose must fall from about 40.8 ms to 25 ms (39% less time), and code '
        'from 31.4 ms to 25 ms (20% less time). At the measured MTP4 traffic ratio, '
        '40 tok/s corresponds to approximately 231 GB/s for prose and 189 GB/s for code. '
        'That fits inside the stated DRAM budget. It does not prove that every execution '
        'stage can meet the 25 ms budget. An ordinary raw Q6 weight-only estimate at '
        '40 tok/s is about 279 GB/s, before state and activation traffic; speculation '
        'changes bytes per generated token.', '',
        'The following arithmetic scales measured rates to higher sustained bandwidth '
        'at unchanged approximate physical traffic per generated token. It also assumes '
        'the compute, dependency, scheduling, and draft costs fall enough to permit that '
        'bandwidth. These are conditional planning figures, not achieved speeds, promises, '
        'or proven maxima. Each cell is prose/code.', '',
        '| Measured configuration used as basis | At 250 GB/s | At 323 GB/s (85%) | At 353.4 GB/s (93%) |',
        '| --- | ---: | ---: | ---: |']
    for model, rows in projections.items():
        cells = [f"{rows['prose']['tok_s_if_all_costs_scale'][str(b)]:.1f}/{rows['code']['tok_s_if_all_costs_scale'][str(b)]:.1f}" for b in (250, 323, 353.4)]
        lines.append('| ' + model + ' | ' + ' | '.join(cells) + ' |')
    lines += ['',
        'The next Qwen engineering priority is target matrix execution and how work '
        'passes between the four sockets, followed by draft output projection and '
        'verification scheduling. Existing traces assign about 58% of summed target '
        'socket-stage time to matrix operations and 11-12% to cross-socket reductions. '
        'The output projection is about 49% of draft graph time. These are instrumented '
        'stage proportions with following barriers included; concurrent socket times '
        'are summed. They are not removable fractions of request time. A fresh wall-time '
        'critical-path profile of the corrected Q6 stack is required before budgeting '
        'a large new kernel or scheduling change.', '',
        'The corrected gather already produced repeated gains of about 3.9% prose '
        'and 2.5% code. Broader barrier/spin changes, huge pages, expanded Q6 layouts, '
        'and the latest ten-route tile change did not establish a consistent model '
        'gain. Repeating those settings alone is not evidence of a route to 40. '
        'The latest scheduling change was -0.60% on prose and +0.10% on code. '
        'Qwen Q4 removes only 8.69% of estimated active target bytes (9.51% equal-bandwidth '
        'weight-only speed headroom), which does not close the observed prose gap by itself.', '',
        'Flash Q4 first needs an uncontended repeated baseline and profiling of its '
        'mixed Q4/Q5/Q6 expert kernels alongside the retained dense Q8 kernels. About '
        '8.97 of its estimated 14.25 GB of raw active target weights remain dense/shared/other. '
        'Full needs fresh model profiling and validation of any transplanted optimizations; '
        'the available evidence does not identify its largest current bottleneck.', '',
        'Large streaming bandwidth is a hardware budget. Layer dependencies, unpacking '
        'and arithmetic, small operations, state traffic, and synchronization can stop '
        'decode from issuing enough useful memory requests to spend that budget. '
        'This interpretation follows [Intel\'s CPU roofline guidance](https://www.intel.com/content/www/us/en/docs/advisor/get-started-guide/2023-0/identify-bottlenecks-using-cpu-roofline.html). '
        'Better speculation or less redundant traffic can improve tok/s while reducing GB/s.', '',
        '- [Machine-readable budgets and source hashes](results/model-speed-headroom-0910.json)',
        '- [Completed Flash Q4 switch](FLASH-Q4-SWITCH-20260910.md)',
        '- [Qwen scheduling comparison](QWEN-DECODE-SCHEDULING-20260909.md)',
        '- [Qwen operation attribution](results/qwen-shared-dispatch-ops-analysis-0909.json)',
        '- [Earlier all-model bandwidth assessment](MODEL-250GBPS-20260909.md)', '']
    note.write_text('\n'.join(lines))
    print(json.dumps(dict(passed=True, report=str(note), qwen_40_budget=qwen_budget), indent=2))


if __name__ == '__main__':
    main()
