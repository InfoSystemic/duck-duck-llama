#!/usr/bin/env python3
"""Offline capture of the model catalog's skill-usage-instruction switch."""
import copy,http.server,json,threading,urllib.request
from pathlib import Path
import capture_prompt_breakdown as cap
H=Path(__file__).resolve().parent
O=H/'prompt-skill-variants-0919'
def main():
 assert not O.exists();O.mkdir(mode=0o700);cap.O=O
 original=json.loads(Path('/home/user/.codex-glm/model-catalogs/glm-5.3-flash.json').read_text())
 server=http.server.ThreadingHTTPServer(('127.0.0.1',0),cap.Recorder);server.captures=cap.queue.Queue()
 threading.Thread(target=server.serve_forever,daemon=True).start();rows=[]
 try:
  for name,usage in [('skill-usage-on',True),('skill-usage-off',False)]:
   catalog=copy.deepcopy(original)
   for model in catalog['models']:model['include_skills_usage_instructions']=usage
   path=O/(name+'.catalog.json');path.write_text(json.dumps(catalog,indent=2)+'\n');cap.CAT=path
   row=cap.run_case(server,name,[])
   body=json.loads(Path(row['capture']).read_text());body={k:v for k,v in body.items() if v is not None}
   req=urllib.request.Request('http://127.0.0.1:18131/v1/responses/input_tokens',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
   row['native_token_count']=json.load(urllib.request.urlopen(req,timeout=30))
   rows.append(row);print(json.dumps(row),flush=True)
 finally:server.shutdown();server.server_close()
 (O/'summary.json').write_text(json.dumps({'inference_requests':0,'production_changed':False,'completed':True,'cases':rows},indent=2)+'\n')
if __name__=='__main__':main()
