"""Context setup and bounded prefill; no checkpoint load at import time."""
import torch


def resize_context(module, config, model, context):
    if context not in (256, 4096, 16384):
        raise ValueError('Supported evaluation contexts: 256, 4096, 16384')
    config.max_seq_len = model.max_seq_len = context
    if model.engram_hash is not None:
        cache = model.engram_hash.cache
        model.engram_hash.cache = torch.zeros((cache.shape[0], context), dtype=cache.dtype, device='cpu')
    module.precompute_freqs_cis.cache_clear()
    for layer in model.layers:
        attn = layer.attn
        attn.freqs_cis = module.precompute_freqs_cis(
            config.rope_head_dim, context,
            config.original_seq_len if attn.compress_ratio else 0,
            config.compress_rope_theta if attn.compress_ratio else config.rope_theta,
            config.rope_factor, config.beta_fast, config.beta_slow)
        attn.window_kv_cache.zero_()
        if attn.is_kv_source:
            cache = attn.compress_kv_cache
            attn.compress_kv_cache = torch.zeros(
                (cache.shape[0], context // attn.compress_ratio, cache.shape[2]),
                dtype=cache.dtype, device='cpu')
            if attn.compress_ratio > 1:
                attn.compressor.kv_state.zero_()
                attn.compressor.score_state.fill_(-torch.inf)
        if attn.indexer is not None:
            attn.indexer.freqs_cis = None
            if attn.indexer.owns_k:
                cache = attn.indexer.k_cache
                attn.indexer.k_cache = torch.zeros(
                    (cache.shape[0], context // attn.compress_ratio, cache.shape[2]),
                    dtype=cache.dtype, device='cpu')
    module.shared_attn.compress_kv = None
    module.shared_attn.index_k = None
    module.shared_attn.topk_idxs = None
    module.shared_attn.candidates = None


class BoundedPrefill:
    """The official nonzero-position path only accepts one token at a time."""
    def __init__(self, model, initial_tokens):
        if type(initial_tokens) is not int or not 1 <= initial_tokens <= 256:
            raise ValueError('Initial prefill must be between 1 and 256 tokens')
        self.model, self.initial_tokens = model, initial_tokens
        self.prefill_calls = 0

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, ids, start_pos):
        length = ids.shape[1]
        if ids.ndim != 2 or ids.shape[0] != 1 or length < 1:
            raise ValueError('Expected nonempty batch-one token input')
        if start_pos < 0 or start_pos + length > self.model.max_seq_len:
            raise ValueError('Input positions exceed the configured context')
        if start_pos:
            if length != 1:
                raise ValueError('Nonzero-position chunks must contain exactly one token')
            return self.model(ids, start_pos)
        first = min(length, self.initial_tokens)
        result = self.model(ids[:, :first], 0)
        self.prefill_calls = 1
        for position in range(first, length):
            result = self.model(ids[:, position:position + 1], position)
            self.prefill_calls += 1
        return result
