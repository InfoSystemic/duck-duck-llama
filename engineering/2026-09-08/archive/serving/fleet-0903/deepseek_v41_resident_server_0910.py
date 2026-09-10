#!/usr/bin/env python3
"""Serve the same native CPU text graph with bounded retained expert mappings."""
import deepseek_v41_server_0910 as server
from deepseek_v41_resident_store_0910 import ResidentStore, bind

server.ServingStore = ResidentStore
server.bind = bind

if __name__ == '__main__':
    server.main()
