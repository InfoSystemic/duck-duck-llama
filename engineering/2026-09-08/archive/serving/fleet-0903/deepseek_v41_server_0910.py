#!/usr/bin/env python3
"""Local OpenAI-compatible text endpoint for the native DeepSeek CPU bring-up."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import torch
from tokenizers.decoders import DecodeStream
from deepseek_v41_checkpoint_0910 import BASE,meta_model,REVISION
from deepseek_v41_native_bridge_0910 import NativeGemm
from deepseek_v41_serving_store_0910 import ServingStore
from run_deepseek_v41_checkpoint_0910b import bind
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import sha256

MODEL='DeepSeek-V4.1-Flash'


class Runtime:
    def __init__(self,args):
        self.args=args;self.lock=threading.Lock();self.completed=0
        torch.set_num_threads(16);torch.set_num_interop_threads(1);torch.set_default_dtype(torch.bfloat16)
        torch.set_default_device('cpu')
        lib=BASE/'results/deepseek-v41-native-gemm-0910b/libdeepseek-v41-native-gemm.so'
        proof=json.loads((lib.parent/'kernel-check.json').read_text())
        assert proof['passed'] and sha256(lib)==proof['input_sha256'][str(lib)]
        self.native=NativeGemm(lib,16);self.native.install()
        self.module,self.config,self.adapter,self.model=meta_model()
        self.tokenizer=self.adapter.backend_tokenizer
        self.store=ServingStore(args.cache,args.output)
        self.backbone=bind(self.model,self.store)
        ep=BASE/'results/deepseek-v41-cpu-source-0910/encoding/encoding.py'
        spec=importlib.util.spec_from_file_location('deepseek_v41_server_encoding',ep)
        self.encoder=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.encoder)
        self.eos=self.tokenizer.token_to_id(self.encoder.eos_token)

    def prepare(self,body):
        assert isinstance(body,dict),'Expected a JSON object'
        assert body.get('model',MODEL) in [MODEL,'deepseek-v4.1-flash','deepseek-v41-flash'],'Unknown model'
        messages=body.get('messages');assert isinstance(messages,list) and 1<=len(messages)<=32,'Expected 1–32 messages'
        assert all(isinstance(m,dict) and m.get('role') in ['system','user','assistant'] and isinstance(m.get('content'),str) for m in messages),'Text messages with system, user, or assistant roles are supported'
        assert body.get('temperature',0)==0,'This initial endpoint supports temperature=0'
        assert not body.get('tools') and not body.get('tool_choice'),'Tool execution is not enabled in this endpoint'
        assert body.get('n',1)==1 and not body.get('logprobs') and not body.get('stop'),'n=1 without logprobs or custom stop sequences is supported'
        count=body.get('max_completion_tokens',body.get('max_tokens',64))
        assert type(count) is int and 1<=count<=128,'Output limit must be between 1 and 128 tokens'
        prompt=self.encoder.encode_messages(messages,thinking_mode='chat')
        ids=self.tokenizer.encode(prompt,add_special_tokens=False).ids
        assert len(ids)+count<=self.config.max_seq_len,'This initial endpoint has a 256-token context limit'
        return ids,count

    def generate(self,ids,count,emit):
        x=torch.tensor([ids],dtype=torch.int64);position=0;tokens=[];pieces=[];stream=DecodeStream(skip_special_tokens=True)
        start=time.perf_counter();before=self.store.downloaded_bytes;first=None;decode_seconds=0
        for step in range(count):
            began=time.perf_counter();output,logits,_=self.model(x,position);elapsed=time.perf_counter()-began
            assert torch.isfinite(logits).all(),'Nonfinite logits'
            if first is None:first=elapsed
            else:decode_seconds+=elapsed
            token=int(output.item());tokens.append(token)
            if token==self.eos:break
            delta=stream.step(self.tokenizer,token)
            if delta:emit(delta);pieces.append(delta)
            position+=x.shape[1];x=torch.tensor([[token]],dtype=torch.int64)
        content=self.tokenizer.decode(tokens,skip_special_tokens=True);emitted=''.join(pieces)
        assert content.startswith(emitted),'Streaming decoder changed its emitted prefix'
        if len(content)>len(emitted):emit(content[len(emitted):])
        result=dict(content=content,finish_reason='stop' if tokens[-1]==self.eos else 'length',
            usage=dict(prompt_tokens=len(ids),completion_tokens=len(tokens),total_tokens=len(ids)+len(tokens)),
            timings=dict(total_seconds=time.perf_counter()-start,prefill_seconds=first,decode_seconds=decode_seconds,
                downloaded_bytes=self.store.downloaded_bytes-before,cache_bytes=self.store.stored_bytes),token_ids=tokens)
        self.completed+=1
        atomic_json(self.args.output/'last-request.json',dict(completed=self.completed,usage=result['usage'],timings=result['timings'],finish_reason=result['finish_reason']))
        return result


class Handler(BaseHTTPRequestHandler):
    protocol_version='HTTP/1.1'

    def log_message(self,*_):pass

    def json(self,status,value):
        data=json.dumps(value,ensure_ascii=False).encode()
        self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)))
        self.end_headers();self.wfile.write(data)

    def do_GET(self):
        runtime=self.server.runtime
        if self.path=='/health':self.json(200,dict(status='ok',model=MODEL,busy=runtime.lock.locked(),context=256,native_precision=True))
        elif self.path=='/v1/models':self.json(200,dict(object='list',data=[dict(id=MODEL,object='model',owned_by='deepseek-ai')]))
        else:self.json(404,dict(error='Unknown endpoint'))

    def do_POST(self):
        if self.path!='/v1/chat/completions':self.json(404,dict(error='Unknown endpoint'));return
        runtime=self.server.runtime
        try:
            size=int(self.headers.get('Content-Length','0'));assert 0<size<=131072,'Request size must be 1–131072 bytes'
            body=json.loads(self.rfile.read(size));ids,count=runtime.prepare(body)
        except (ValueError,AssertionError,TypeError) as e:self.json(400,dict(error=dict(message=str(e),type='invalid_request_error')));return
        if not runtime.lock.acquire(blocking=False):self.json(409,dict(error=dict(message='Model is handling another request',type='model_busy')));return
        streaming=bool(body.get('stream',False));identifier='chatcmpl-'+uuid.uuid4().hex;created=int(time.time())
        stop=threading.Event();write_lock=threading.Lock();disconnected=threading.Event()
        def event(value):
            with write_lock:
                if disconnected.is_set():raise ConnectionError('Client disconnected')
                self.wfile.write(b'data: '+json.dumps(value,ensure_ascii=False).encode()+b'\n\n');self.wfile.flush()
        def chunk(delta,finish=None):return dict(id=identifier,object='chat.completion.chunk',created=created,model=MODEL,
            choices=[dict(index=0,delta=delta,finish_reason=finish)])
        def heartbeat():
            while not stop.wait(10):
                try:
                    with write_lock:self.wfile.write(b': processing\n\n');self.wfile.flush()
                except OSError:disconnected.set();return
        try:
            if streaming:
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.send_header('Cache-Control','no-cache');self.send_header('Connection','close');self.end_headers()
                self.close_connection=True;event(chunk(dict(role='assistant')))
                threading.Thread(target=heartbeat,daemon=True).start()
            def emit(delta):
                if disconnected.is_set():raise ConnectionError('Client disconnected')
                if streaming:event(chunk(dict(content=delta)))
            result=runtime.generate(ids,count,emit)
            if streaming:
                event(chunk({},result['finish_reason']))
                with write_lock:self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush()
            else:
                self.json(200,dict(id=identifier,object='chat.completion',created=created,model=MODEL,
                    choices=[dict(index=0,message=dict(role='assistant',content=result['content']),finish_reason=result['finish_reason'])],
                    usage=result['usage'],timings=result['timings']))
        except (BrokenPipeError,ConnectionError,ConnectionResetError):pass
        except Exception as e:
            error=dict(error=dict(type='inference_error',message=type(e).__name__))
            if streaming:
                try:event(error)
                except OSError:pass
            else:self.json(500,error)
            print(json.dumps(dict(request_failed=type(e).__name__)),flush=True)
        finally:stop.set();runtime.lock.release()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=18170)
    parser.add_argument('--cache',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--lifecycle-lock-fd',type=int,required=True);args=parser.parse_args()
    assert os.fstat(args.lifecycle_lock_fd).st_ino==(BASE/'results/qwen-q6-trial-0907/lifecycle.lock').stat().st_ino
    runtime=Runtime(args)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);server.daemon_threads=True;server.runtime=runtime
    atomic_json(args.output/'ready.json',dict(ready=True,pid=os.getpid(),port=args.port,model=MODEL,revision=REVISION,
        context=256,native_precision=True,vision=False,dspark=False,cache_limit_bytes=runtime.store.cap_bytes,
        lifecycle_lock_held=True,source_sha256=sha256(__file__),ready_at=time.time()))
    try:server.serve_forever(poll_interval=.25)
    finally:server.server_close();runtime.store.close()


if __name__=='__main__':main()
