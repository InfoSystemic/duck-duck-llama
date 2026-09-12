"""Scheduled diagnostic helper; importing this file never loads/runs a model.

After an exclusive model handoff, call ``trace_runtime(runtime, optimizations,
output_dir)`` on the already loaded runtime. It replays the same prompt and first
decode token in fresh request threads at Torch 16, 8, then 16 workers. Native
workers and optimization choices stay fixed. This is a parity trace, not a speed
benchmark, and must not overlap another model trial.
"""
import hashlib
import json
from pathlib import Path
import threading
import types

import torch


def _tensors(value, path='output'):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _tensors(item, f'{path}.{index}')
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _tensors(item, f'{path}.{key}')


def _digest(value):
    raw = value.detach().contiguous().view(torch.uint8).reshape(-1).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def trace_runtime(runtime, optimizations, output_dir, threads=(16, 8, 16), steps=2):
    """Trace common submodules, HC methods, F.linear and einsum without rewriting math."""
    if steps < 2 or steps > 10:
        raise ValueError('Replay 2–10 fixed-token steps, including prefill')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / 'thread-parity.json').exists():
        raise FileExistsError('Use a fresh trace output directory')
    if not runtime.lock.acquire(blocking=False):
        raise RuntimeError('The supplied runtime is handling a request')
    old_threads = torch.get_num_threads()
    old_config = dict(optimizations.config)
    m = runtime.module
    old_linear, old_einsum = m.F.linear, m.torch.einsum
    hooks, methods = [], []
    active = {}
    reference = []
    result = dict(passed=False, mode='diagnostic fixed-token replay, not a speed benchmark',
                  torch_threads=list(threads), native_workers=16, steps=steps, runs=[])

    def record(label, value, inputs=None):
        for path, tensor in _tensors(value):
            key = f'{active["step"]}:{label}:{path}'
            row = dict(key=key, shape=list(tensor.shape), dtype=str(tensor.dtype), sha256=_digest(tensor))
            if inputs is not None:
                row['input_tensors'] = [dict(path=p, shape=list(x.shape), dtype=str(x.dtype), sha256=_digest(x))
                                        for p, x in _tensors(inputs, 'input')]
            position = len(active['events'])
            if active['baseline']:
                reference.append((row, tensor.detach().clone()))
            else:
                if position >= len(reference) or reference[position][0]['key'] != key:
                    row['event_order_matches'] = False
                else:
                    prior, expected = reference[position]
                    row['event_order_matches'] = True
                    row['exact'] = row['sha256'] == prior['sha256']
                    if not row['exact']:
                        row['baseline_sha256'] = prior['sha256']
                        if 'input_tensors' in row:
                            row['input_tensors_match'] = row['input_tensors'] == prior.get('input_tensors')
                        if tensor.shape == expected.shape:
                            delta = (tensor.detach().float() - expected.float()).abs()
                            row['max_abs'] = float(delta.max()) if delta.numel() else 0.0
                            row['unequal_elements'] = int((tensor != expected).sum())
                        if active['first_difference'] is None:
                            active['first_difference'] = row
                            stem = output_dir / f'first-difference-run-{active["run"]}'
                            torch.save(dict(event=row, baseline=expected, actual=tensor.detach().clone()),
                                       stem.with_suffix('.pt'))
            active['events'].append(row)

    def linear(x, weight, bias=None):
        output = old_linear(x, weight, bias)
        # Input activations are small; checkpoint weight matrices can exceed
        # 2 GB and are intentionally not copied or hashed on every invocation.
        record(f'F.linear:{list(weight.shape)}:{weight.dtype}', output, x)
        return output

    def einsum(equation, *operands):
        output = old_einsum(equation, *operands)
        # Hash activations only; the second wo_a operand is a large fixed weight.
        record(f'einsum:{equation}', output, operands[0] if operands else None)
        return output

    try:
        ids, _ = runtime.prepare(dict(messages=[dict(role='user', content='Hi.')], max_tokens=16))
        golden_path = Path(__file__).resolve().parent / 'results/deepseek-v41-checkpoint-run-0910b/generation.json'
        golden = json.loads(golden_path.read_text())['runs'][0]
        fixed_ids = golden['token_ids'][:steps - 1]
        result['fixed_decode_inputs'] = fixed_ids
        result['prompt_ids'] = ids
        # Keep every optimization and native-worker count constant. Only the
        # Torch/MKL thread setting changes between the three request threads.
        config = dict(old_config, native_workers=16)
        config.pop('torch_workers', None)
        result['optimization_config'] = config
        optimizations.configure(config)
        for name, module in runtime.model.named_modules():
            if not name or '.ffn.experts.' in name or '.shared_experts.' in name:
                continue
            if hasattr(module, 'register_forward_hook'):
                def hook(_module, inputs, output, _name=name):
                    record('module:' + _name, output)
                hooks.append(module.register_forward_hook(hook))
        for layer in runtime.model.layers:
            for method_name in ['hc_mixes', 'hc_pre', 'hc_post']:
                original = getattr(layer, method_name)
                label = f'layers.{layer.layer_id}.{method_name}'

                def wrapped(this, *args, _original=original, _label=label, **kwargs):
                    output = _original(*args, **kwargs)
                    record(_label, output)
                    return output

                methods.append((layer, method_name, original))
                setattr(layer, method_name, types.MethodType(wrapped, layer))
        m.F.linear, m.torch.einsum = linear, einsum
        for run, workers in enumerate(threads):
            active.clear()
            active.update(run=run, baseline=run == 0, events=[], first_difference=None, step=0)
            failures, outputs = [], []
            before = runtime.store.downloaded_bytes

            def request_thread():
                try:
                    torch.set_num_threads(workers)
                    with torch.inference_mode():
                        outputs.append(runtime.model(torch.tensor([ids], dtype=torch.int64), 0))
                        for step, token in enumerate(fixed_ids, 1):
                            active['step'] = step
                            outputs.append(runtime.model(torch.tensor([[token]], dtype=torch.int64),
                                                         len(ids) + step - 1))
                except BaseException as error:
                    failures.append(error)

            thread = threading.Thread(target=request_thread)
            thread.start()
            thread.join()
            if failures:
                raise failures[0]
            trace_file = output_dir / f'events-run-{run}-torch-{workers}.json'
            trace_file.write_text(json.dumps(active['events'], indent=2) + '\n')
            summary = dict(torch_workers=workers, events=len(active['events']),
                           first_difference=active['first_difference'],
                           downloaded_bytes=runtime.store.downloaded_bytes - before,
                           output_ids=[int(x[0].item()) for x in outputs], trace=str(trace_file))
            if run:
                summary['complete_event_order_matches'] = (len(active['events']) == len(reference) and
                    all(row.get('event_order_matches') for row in active['events']))
                summary['all_events_exact'] = summary['complete_event_order_matches'] and all(
                    row.get('exact') for row in active['events'])
            result['runs'].append(summary)
            (output_dir / 'thread-parity.json').write_text(json.dumps(result, indent=2) + '\n')
        result['passed'] = True
        return result
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        m.F.linear, m.torch.einsum = old_linear, old_einsum
        for handle in hooks:
            handle.remove()
        for layer, name, original in methods:
            setattr(layer, name, original)
        optimizations.configure(old_config)
        torch.set_num_threads(old_threads)
        runtime.lock.release()
        (output_dir / 'thread-parity.json').write_text(json.dumps(result, indent=2) + '\n')
