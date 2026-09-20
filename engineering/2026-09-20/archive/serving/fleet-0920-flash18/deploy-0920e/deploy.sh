#!/bin/bash
# deploy.sh -- install the f18 revision e drop-in on glm53-flash-production.service, restart, verify; roll back to revision d on failure.
set -uo pipefail
D=/home/user/InfoSystemic/AI-Server/serving/fleet-0920-flash18/deploy-0920e
C=/home/user/InfoSystemic/AI-Server/serving/fleet-0920-flash18/deploy-0920d
DROP=$HOME/.config/systemd/user/glm53-flash-production.service.d/95-f18-0920.conf
cd $D && sha256sum -c SHA256SUMS || { echo "library hash mismatch"; exit 1; }
curl -s -m 5 http://127.0.0.1:18131/slots | python3 -c "import json,sys; s=json.load(sys.stdin); sys.exit(1 if any(x.get('is_processing') for x in s) else 0)" || { echo "production busy or down; not touching it"; exit 1; }
rollback() { echo "ROLLBACK to revision d: $1"; cp $C/95-f18-0920.conf.proposed $DROP; systemctl --user daemon-reload; systemctl --user restart glm53-flash-production.service; exit 1; }
cp $D/95-f18-0920.conf.proposed $DROP && systemctl --user daemon-reload && systemctl --user restart glm53-flash-production.service || rollback "restart failed"
for i in $(seq 900); do
  s=$(curl -s -m 2 http://127.0.0.1:18131/health 2>/dev/null || true); [[ "$s" == *'"ok"'* ]] && break
  systemctl --user is-active -q glm53-flash-production.service || rollback "service not active while loading"
  sleep 1
done
[[ "$(curl -s -m 2 http://127.0.0.1:18131/health)" == *'"ok"'* ]] || rollback "not healthy after 900 s"
PID=$(systemctl --user show -p MainPID --value glm53-flash-production.service)
echo "healthy after ${i}s, pid $PID, threads $(ls /proc/$PID/task | wc -l)"
grep -oE '/home/user[^ ]*(libggml-base|libggml-cpu|libllama-common|libllama)\.so[^ ]*' /proc/$PID/maps | sort -u
for lib in libggml-base.so.0.22.0 libggml-cpu.so.0.22.0 libllama-common.so.0.3.0 libllama.so.0.3.0; do grep -q "$D/$lib" /proc/$PID/maps || rollback "$lib is not the revision e copy"; done
tr '\0' '\n' < /proc/$PID/environ | grep -E '^GGML_F18|^GGML_CPU_GLM_POOL_CACHE|^GOMP|^GGML_CPU_NUMA_SHARED|^LLAMA_F18|^GGML_META_F18'
R=$(curl -s -m 120 http://127.0.0.1:18131/completion -H 'Content-Type: application/json' -d '{"prompt":"The capital of France is","n_predict":8,"temperature":0,"stream":false}' | python3 -c "import json,sys; print(json.load(sys.stdin)['content'])")
echo "sanity completion: $R"
[[ "$R" == *Paris* ]] || rollback "sanity completion did not mention Paris"
echo DEPLOYED
