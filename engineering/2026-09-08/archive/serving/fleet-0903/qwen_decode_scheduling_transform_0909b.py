"""Correct the experimental Q6 tile predicate to the model's ten expert routes."""
from qwen_decode_scheduling_transform_0909 import transform as prototype_transform


def transform(source):
    source = prototype_transform(source)
    old = 'experts != 512 || used != 8 || tokens < 1 || tokens > 8'
    assert source.count(old) == 1
    source = source.replace(old, 'experts != 512 || used != 10 || tokens < 1 || tokens > 8')
    old = 'QWEN_Q6_MOE_TILE rows=%lld\\n'
    assert source.count(old) == 1
    return source.replace(old, 'QWEN_Q6_MOE_TILE rows=%lld routes=10 k=2560 nc=160\\n')
