#!/bin/bash
# setmode.sh <f18 mask> [poolcache 0|1] -- flip the in-process feature switches (only while the server is idle)
python3 - "$1" "${2:-}" <<'PY'
import struct,sys
open('/dev/shm/f18-control.u32','r+b').write(struct.pack('<I',int(sys.argv[1],0)))
if sys.argv[2] != '': open('/dev/shm/f18-poolcache.u32','r+b').write(struct.pack('<I',int(sys.argv[2])))
PY
