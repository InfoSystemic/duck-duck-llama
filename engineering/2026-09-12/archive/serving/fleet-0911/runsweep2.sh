#!/bin/bash
D=~/InfoSystemic/AI-Server/serving/fleet-0911
while pgrep -f 'bash ./sweep.sh' >/dev/null || pgrep -f 'bash ./runpar.sh' >/dev/null || pgrep -f 'bash ./runprof.sh' >/dev/null; do sleep 20; done
cd $D && ./sweep.sh sweep2.txt
