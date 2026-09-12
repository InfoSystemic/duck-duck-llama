"""Exact mixed FP4/FP8 grouped bridge sharing the bounded resident pack provider."""
import ctypes

from deepseek_v41_native_grouped_vnni_goal_0910 import Descriptor, GroupedVnniGemm


class GroupedLattice16(GroupedVnniGemm):
    def __init__(self, library, pack_provider, workers=16, fallback=None):
        super().__init__(library, pack_provider, workers, fallback)
        self.fn = self.library.ds41_grouped_lattice16
        self.fn.argtypes = [ctypes.POINTER(Descriptor), ctypes.c_int, ctypes.c_int]
        self.fn.restype = ctypes.c_int
        self.eligibility_fn = self.library.ds41_lattice16_blocks
        self.eligibility_fn.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.eligibility_fn.restype = ctypes.c_int

    def eligible_blocks(self, activation):
        assert activation.device.type == 'cpu' and activation.is_contiguous()
        assert activation.numel() == activation.shape[-1]
        return self.eligibility_fn(activation.data_ptr(), activation.shape[-1])
