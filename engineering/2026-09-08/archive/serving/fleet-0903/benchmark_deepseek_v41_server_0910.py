#!/usr/bin/env python3
"""Repeat a complete known greeting on the running CPU endpoint; no IMC claim."""
import json
import os
from pathlib import Path
import time
import urllib.request
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import process_info,sha256,inference_snapshot
from select_flash_q4_0910c import Manager
from model_measurement_guard import ModelMeasurementGuard

BASE=Path(__file__).resolve().parent
OUT=BASE/'results/deepseek-v41-server-greeting-0910'


def main():
    assert os.sched_getaffinity(0)=={127} and not OUT.exists()
    selected=json.loads((BASE/'deepseek-v41-selected.json').read_text())
    pid=selected['pid'];info=process_info(pid);assert info['start']==selected['start']
    assert info['affinity']==list(range(48,64))
    assert all(sha256(p)==h for p,h in selected['source_sha256'].items())
    prior=json.loads(Path(selected['evidence']).read_text());assert prior['passed']
    peer=Manager().validate_current();guard=ModelMeasurementGuard(peer['pid'],{peer['pid']:18131},inference_snapshot)
    record=Path(selected['evidence']).parent/'last-request.json'
    previous=json.loads(record.read_text())['completed']
    result=dict(passed=False,started=time.time(),pid=pid,start=info['start'],affinity=info['affinity'],runs=[],
        source_sha256=sha256(__file__),selected_sha256=sha256(BASE/'deepseek-v41-selected.json'),imc_bandwidth_measured=False)
    guard.assert_idle();OUT.mkdir();atomic_json(OUT/'result.json',result)
    try:
        for i in range(2):
            guard.assert_idle()
            with urllib.request.urlopen(selected['endpoint'].replace('/v1','/health'),timeout=5) as response:assert not json.load(response)['busy']
            body=dict(model='DeepSeek-V4.1-Flash',messages=[dict(role='user',content='Hi.')],temperature=0,max_tokens=16,stream=False)
            request=urllib.request.Request(selected['endpoint']+'/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            print(json.dumps(dict(full_greeting_request=i,pid=pid)),flush=True)
            began=time.monotonic()
            with urllib.request.urlopen(request,timeout=600) as response:
                assert response.status==200;data=json.load(response)
            elapsed=time.monotonic()-began;guard.assert_idle()
            assert data['choices'][0]['message']['content']=='Hello! How can I help you today?'
            assert data['choices'][0]['finish_reason']=='stop' and data['usage']['completion_tokens']==10
            last=json.loads(record.read_text());assert last['completed']==previous+1;previous=last['completed']
            assert last['timings']==data['timings']
            row=dict(repetition=i,response=data,wall_seconds=elapsed,
                warm=data['timings']['downloaded_bytes']==0,decode_tok_s=9/data['timings']['decode_seconds'])
            result['runs'].append(row);atomic_json(OUT/'result.json',result)
            print(json.dumps(dict(repetition=i,warm=row['warm'],decode_tok_s=row['decode_tok_s'],downloaded_bytes=data['timings']['downloaded_bytes'])),flush=True)
        assert result['runs'][1]['warm']
        assert process_info(pid)['start']==info['start'] and all(sha256(p)==h for p,h in selected['source_sha256'].items())
        result.update(passed=True,server_preserved=True,completed_greetings=True,
            warm_decode_tok_s=result['runs'][1]['decode_tok_s'],
            scope='Two complete identical greedy greetings on the initial 16-core CPU endpoint. First fills remaining Engram row cache; second has zero weight or row downloads. '
                  'Nine decode tokens after one prefill-produced token; includes EOS. This short bring-up result is not a representative long-context benchmark or whole-server maximum.')
    except BaseException as e:result['error']=repr(e);raise
    finally:
        result['finished']=time.time();atomic_json(OUT/'result.json',result)


if __name__=='__main__':main()
