#!/usr/bin/env python3
"""Capture synthetic Codex requests offline to attribute prompt overhead; no inference."""
import hashlib,http.server,json,os,queue,re,subprocess,threading,time
from pathlib import Path
from test_flash_catalog_capture import tool_labels
from appserver_bench import PROMPT
H=Path(__file__).resolve().parent
O=H/'prompt-overhead-0919';CAT=Path('/home/user/.codex-glm/model-catalogs/glm-5.3-flash.json')
WRAPPER='/home/user/.local/libexec/paseo/codex-smeagol'

class Recorder(http.server.BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_GET(self):
  b=b'{"status":"ok"}';self.send_response(200);self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
 def do_POST(self):
  body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  # Deliberately exclude HTTP headers, credentials, and arbitrary metadata.
  allowed={k:body.get(k) for k in ['model','instructions','input','tools','reasoning','parallel_tool_calls','text','stream']}
  self.server.captures.put(allowed)
  b=b'{"error":{"message":"Offline capture complete; no inference","type":"invalid_request_error","code":"capture_only"}}'
  self.send_response(400);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)

def run_case(server,name,disabled):
 cmd=[WRAPPER,'-c','model_catalog_json='+json.dumps(str(CAT)),
  '-c','model_providers.smeagol.base_url='+json.dumps(f'http://127.0.0.1:{server.server_port}/v1'),
  '-c','model_providers.smeagol.request_max_retries=0','-c','model_providers.smeagol.stream_max_retries=0','app-server']
 for feature in disabled:cmd.extend(['--disable',feature])
 if 'goals' not in disabled:cmd.extend(['--enable','goals'])
 msgs=queue.Queue()
 with (O/(name+'.stderr.log')).open('w') as err:
  p=subprocess.Popen(cmd,cwd=H,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1)
  def reader():
   for line in p.stdout:
    try:msgs.put(json.loads(line))
    except json.JSONDecodeError:pass
   msgs.put({'_exit':True})
  threading.Thread(target=reader,daemon=True).start();seq=0
  def send(method,params,notification=False):
   nonlocal seq
   seq+=1;frame={'method':method,'params':params}
   if not notification:frame['id']=seq
   p.stdin.write(json.dumps(frame)+'\n');p.stdin.flush()
   if notification:return
   end=time.monotonic()+30
   while time.monotonic()<end:
    msg=msgs.get(timeout=max(.1,end-time.monotonic()))
    if msg.get('_exit'):raise RuntimeError('app-server exited')
    if msg.get('id')==seq:
     if 'error' in msg:raise RuntimeError(msg['error'])
     return msg.get('result')
   raise TimeoutError(method)
  try:
   send('initialize',{'clientInfo':{'name':'glm-prompt-capture','version':'1'},'capabilities':{'experimentalApi':True}})
   send('initialized',{},True)
   r=send('thread/start',{'model':'glm-5.3-flash','cwd':str(H),'ephemeral':True,'approvalPolicy':'on-request','sandbox':'workspace-write'})
   send('turn/start',{'threadId':r['thread']['id'],'input':[{'type':'text','text':PROMPT}],
    'model':'glm-5.3-flash','effort':'low','collaborationMode':{'mode':'default','settings':{'model':'glm-5.3-flash'}}})
   body=server.captures.get(timeout=30)
  finally:
   p.terminate()
   try:p.wait(timeout=5)
   except subprocess.TimeoutExpired:p.kill();p.wait(timeout=5)
   p.stdin.close();p.stdout.close()
 path=O/(name+'.request.json')
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'w') as f:json.dump(body,f,indent=2);f.write('\n')
 def texts(item):
  c=item.get('content',[])
  return c if isinstance(c,str) else '\n'.join(x.get('text','') for x in c if isinstance(x,dict))
 items=[]
 for i,item in enumerate(body['input']):
  text=texts(item)
  # Store only headings and sizes in the public summary; raw generated instructions stay local.
  headings=re.findall(r'(?m)^#+[^\n]+|^<[A-Za-z][^\n>]*>',text)
  items.append({'index':i,'role':item.get('role'),'chars':len(text),'headings':headings,'sha256':hashlib.sha256(text.encode()).hexdigest()})
 tools=[]
 for tool in body.get('tools',[]):
  tools.append({'names':tool_labels([tool]),'chars':len(json.dumps(tool))})
 return {'case':name,'disabled_features':disabled,'instructions_chars':len(body.get('instructions','')),
  'input_chars':len(json.dumps(body['input'])),'input_items':items,'tool_names':tool_labels(body['tools']),
  'tool_chars':len(json.dumps(body['tools'])),'tools':tools,'reasoning':body.get('reasoning'),
  'capture':str(path),'inference_requests':0}

def main():
 assert not O.exists();O.mkdir(mode=0o700)
 server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Recorder);server.captures=queue.Queue()
 threading.Thread(target=server.serve_forever,daemon=True).start()
 report={'inference_requests':0,'production_config_changed':False,'cases':[]}
 try:
  for name,disabled in [('baseline',[]),('no-multi-agent',['multi_agent']),('no-goals',['goals']),('no-multi-agent-or-goals',['multi_agent','goals'])]:
   r=run_case(server,name,disabled);report['cases'].append(r)
   (O/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(r),flush=True)
 finally:server.shutdown();server.server_close()
 report['completed']=True;(O/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
