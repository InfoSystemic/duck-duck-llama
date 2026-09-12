#!/usr/bin/env python3
"""Serve the exact-output grouped CPU candidate using the native released weights."""
import deepseek_v41_server_0910 as server
from deepseek_v41_resident_store_0910 import ResidentStore, bind
from goal_runtime_0910 import Optimizations

class GoalRuntime(server.Runtime):
    def __init__(self,args):
        super().__init__(args)
        self.optimizations=Optimizations(self)
        self.optimizations.configure(dict(quant=True,reuse=True,grouped=True,hc=True,sparse=True))

if __name__=='__main__':
    server.ServingStore=ResidentStore
    server.bind=bind
    server.Runtime=GoalRuntime
    server.main()
