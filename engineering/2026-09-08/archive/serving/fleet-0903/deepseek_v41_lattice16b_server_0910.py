#!/usr/bin/env python3
"""Serve grouped lattice16 decode with the exact vector FP32 fallback."""
import deepseek_v41_server_0910 as server
from deepseek_v41_goal_server_0910 import GoalRuntime
from deepseek_v41_resident_store_0910 import ResidentStore, bind
from deepseek_v41_lattice16b_runtime_0910 import Lattice16BRuntime


class LatticeRuntime(GoalRuntime):
    def __init__(self, args):
        super().__init__(args)
        self.lattice16 = Lattice16BRuntime(self)
        self.lattice16.configure(True)


if __name__ == '__main__':
    server.ServingStore = ResidentStore
    server.bind = bind
    server.Runtime = LatticeRuntime
    server.main()
