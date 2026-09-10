#!/usr/bin/env python3
"""Check cache eviction ownership/recovery and incremental tokenizer parity."""
import hashlib
import json
from pathlib import Path
import tempfile
import sys
from tokenizers.decoders import DecodeStream
from check_deepseek_v41_engram_hash_0910 import TokenizerAdapter
from deepseek_v41_serving_store_0910 import finish_eviction
from qwen_split_trial import sha256


def main():
    cases=[]
    with tempfile.TemporaryDirectory(prefix='deepseek-v41-cache-check-') as temporary:
        root=Path(temporary)
        (root/'owner.json').write_text(json.dumps(dict(task='deepseek-v41-native-0910')))
        names=['layers.0.ffn.experts.3.w1.weight','layers.1.ffn.experts.9.w2.scale','head.weight']
        records=[]
        for i,name in enumerate(names):
            raw=bytes([i+1])*100;path=root/(name+'.bin');path.write_bytes(raw)
            records.append(dict(name=name,bytes=len(raw),sha256=sha256(path)))
        manifest=root/'tensors.jsonl'
        manifest.write_text(''.join(json.dumps(r)+'\n' for r in records))
        (root/'eviction.json').write_text(json.dumps(dict(victims=[records[0]])))
        finish_eviction(root)
        assert not (root/(names[0]+'.bin')).exists()
        assert [json.loads(l)['name'] for l in manifest.read_text().splitlines()]==names[1:]
        assert sha256(root/(names[2]+'.bin'))==records[2]['sha256'];cases.append('owned_expert_only')
        finish_eviction(root);cases.append('completed_eviction_idempotent')
        (root/'eviction.json').write_text(json.dumps(dict(victims=[records[1]])))
        (root/(names[1]+'.bin')).unlink();finish_eviction(root)
        assert [json.loads(l)['name'] for l in manifest.read_text().splitlines()]==names[2:];cases.append('recover_after_unlink')
        (root/'eviction.json').write_text(json.dumps(dict(victims=[records[2]])))
        before=manifest.read_bytes()
        try:finish_eviction(root)
        except AssertionError:pass
        else:raise AssertionError('Common weight eviction accepted')
        assert manifest.read_bytes()==before and sha256(root/(names[2]+'.bin'))==records[2]['sha256'];cases.append('common_weights_protected')
        (root/'eviction.json').write_text(json.dumps(dict(victims=[records[0]])))
        (root/(names[0]+'.bin')).symlink_to(root/(names[2]+'.bin'))
        try:finish_eviction(root)
        except AssertionError:pass
        else:raise AssertionError('Symlink eviction accepted')
        assert sha256(root/(names[2]+'.bin'))==records[2]['sha256'];cases.append('symlink_protected')
    tokenizer=TokenizerAdapter().backend_tokenizer
    streams=0
    for text in ['Hello! How can I help you today?','你好，世界。','🦆 Café naïve résumé','English 中文 العربية']:
        ids=tokenizer.encode(text,add_special_tokens=False).ids
        for count in range(1,len(ids)+1):
            decoder=DecodeStream(skip_special_tokens=True);pieces=[]
            for token in ids[:count]:
                piece=decoder.step(tokenizer,token)
                if piece:pieces.append(piece)
            emitted=''.join(pieces);full=tokenizer.decode(ids[:count],skip_special_tokens=True)
            assert full.startswith(emitted)
            assert emitted+full[len(emitted):]==full;streams+=1
    result=dict(passed=True,eviction_cases=cases,incremental_decode_prefixes=streams,
        input_sha256={str(p):sha256(p) for p in [Path(__file__),Path(__file__).with_name('deepseek_v41_serving_store_0910.py')]})
    Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
