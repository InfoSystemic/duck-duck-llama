from pathlib import Path
import subprocess
here=Path(__file__).resolve().parent
for script,log in [('build_candidate.py','build-controller.log'),('validate.py','validation-controller.log'),('run_numa.py','numa-controller.log')]:
    with (here/log).open('x') as out:
        subprocess.run(['python3','-u',str(here/script)],cwd=here,stdout=out,stderr=subprocess.STDOUT,check=True)
    print('completed',script,flush=True)
