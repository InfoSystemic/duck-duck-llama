"""Optional, independently selectable CPU optimizations for the native text runtime."""
from pathlib import Path
from goal_native_quant_0910 import NativeQuant
from goal_quant_reuse_0910 import install as install_reuse
from goal_hc_0910 import NativeHC
from goal_sparse_script_0910 import install as install_sparse
from goal_grouped_moe_0910 import install as install_grouped

BASE=Path(__file__).resolve().parent

class Optimizations:
    def __init__(self,runtime):
        self.runtime=runtime
        self.quant=NativeQuant(BASE/'results/goal_native_quant_0910/libgoal_native_quant_0910.so')
        self.original_quant=runtime.module.act_quant
        self.hc=NativeHC(BASE/'results/goal_hc_0910/libgoal_hc_0910.so')
        self.original_hc=runtime.module.hc_split_sinkhorn
        self.reuse=install_reuse(runtime.module)
        self.sparse=install_sparse(runtime.module)
        self.grouped=install_grouped(runtime)
        self.configure({})

    def configure(self,config):
        self.config=dict(config)
        self.runtime.native.workers=config.get('native_workers',16)
        self.runtime.module.act_quant=self.quant.act_quant if config.get('quant') else self.original_quant
        self.runtime.module.hc_split_sinkhorn=self.hc.hc_split_sinkhorn if config.get('hc') else self.original_hc
        self.reuse.enabled=bool(config.get('reuse'))
        self.sparse.enabled=bool(config.get('sparse'))
        self.grouped.enabled=bool(config.get('grouped'))
