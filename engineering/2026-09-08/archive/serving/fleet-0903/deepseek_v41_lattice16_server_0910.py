#!/usr/bin/env python3
"""Serve exact grouped lattice16 decode over the native DeepSeek text checkpoint."""
import deepseek_v41_server_0910 as server
from deepseek_v41_goal_server_0910 import GoalRuntime
from deepseek_v41_resident_store_0910 import ResidentStore, bind
from deepseek_v41_lattice16_runtime_0910 import Lattice16Runtime


class LatticeRuntime(GoalRuntime):
    def __init__(self, args):
        super().__init__(args)
        self.lattice16 = Lattice16Runtime(self)
        self.lattice16.configure(True)


if __name__ == '__main__':
    server.ServingStore = ResidentStore
    server.bind = bind
    server.Runtime = LatticeRuntime
    server.main()
