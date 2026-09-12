"""Opt-in causal text chunk verification for the unmodified native checkpoint.

This is experimental verifier plumbing, not a speculative generation server.
Native projections can process several tokens per layer. Cache updates and
compressed index selection keep the single-token order, and attention keeps the
same ring-slot order and per-query sparse reduction. No draft token is trusted.
"""
import types

import torch
import torch.nn.functional as F


class CacheSnapshot:
    """Bounded rollback of mutable inference buffers and shared attention state."""
    MUTABLE = ('window_kv_cache', 'compress_kv_cache', 'k_cache', 'kv_state', 'score_state', 'cache')

    def __init__(self, model, module, cap_bytes=256 << 20):
        items = [(b, b.numel() * b.element_size()) for name, b in model.named_buffers()
                 if name.rsplit('.', 1)[-1] in self.MUTABLE]
        self.bytes = sum(size for _, size in items)
        if self.bytes > cap_bytes:
            raise ValueError(('Verifier snapshot exceeds memory cap', self.bytes, cap_bytes))
        self.buffers = [(b, b.clone()) for b, _ in items]
        self.shared = module.shared_attn
        self.shared_values = dict(vars(self.shared))
        self.restored = False

    @torch.inference_mode()
    def restore(self):
        for destination, value in self.buffers:
            destination.copy_(value)
        vars(self.shared).clear()
        vars(self.shared).update(self.shared_values)
        self.restored = True


class ChunkVerifier:
    def __init__(self, model, module, max_chunk=8, strict_linear=True):
        if (not 2 <= max_chunk <= 8 or getattr(module, 'world_size', 1) != 1
                or getattr(model, 'temperature', None) != 0):
            raise ValueError('Only a single-rank greedy CPU verifier with chunks 2..8 is supported')
        self.model, self.module = model, module
        self.max_chunk, self.strict_linear = max_chunk, strict_linear
        self.active = False
        self.start = self.length = 0
        self.candidates = []
        self.compressed_indices = None
        self.bindings = []
        self.batch_calls = self.serial_fallback_calls = 0
        self.last_mode = None
        self.original_head = model.head.forward
        self.original_linear = F.linear
        for layer in model.layers:
            attn = layer.attn
            original = attn.forward
            def forward(this, x, start_pos, _original=original):
                if not self.active:
                    return _original(x, start_pos)
                return self.attention(this, x, start_pos)
            self.bindings.append((attn, original))
            attn.forward = types.MethodType(forward, attn)

        def head(this, x, full_logits=False):
            if not self.active:
                return self.original_head(x, full_logits)
            # Each row uses the same original head GEMV and output dtype.
            return torch.stack([self.original_head(x[:, i:i+1]) for i in range(x.shape[1])], 1)
        model.head.forward = types.MethodType(head, model.head)

    def linear(self, x, weight, bias=None):
        if not self.active or not self.strict_linear or x.numel() == x.shape[-1]:
            return self.original_linear(x, weight, bias)
        flat = x.reshape(-1, x.shape[-1])
        return torch.cat([self.original_linear(flat[i:i+1], weight, bias)
                          for i in range(flat.shape[0])], 0).reshape(*x.shape[:-1], weight.shape[0])

    def snapshot(self):
        return CacheSnapshot(self.model, self.module)

    @torch.inference_mode()
    def verify(self, input_ids, start_pos, *, rollback=False):
        if (self.active or input_ids.device.type != 'cpu' or input_ids.dtype != torch.int64
                or input_ids.ndim != 2 or input_ids.shape[0] != 1
                or not 2 <= input_ids.shape[1] <= self.max_chunk
                or not 0 < start_pos < start_pos + input_ids.shape[1] <= self.model.max_seq_len):
            raise ValueError('Expected a bounded nonzero-position CPU text chunk, batch one')
        if F.linear is not self.original_linear:
            raise RuntimeError('F.linear changed after verifier construction')
        snapshot = self.snapshot()
        # The original indexer republishes its shared key pointer only when a
        # compressed group completes. Between completions, serial decoding can
        # observe a later layer's keys from the previous token. Preserve that
        # cross-layer dependency rather than silently changing the CPU model.
        # It cannot affect selection when every visible position fits in top-k.
        if any(attn.indexer is not None and
               (start_pos + input_ids.shape[1]) // attn.compress_ratio > attn.indexer.index_topk
               for attn, _ in self.bindings if attn.compress_ratio):
            self.serial_fallback_calls += 1
            self.last_mode = 'serial_preserve_shared_index_dependency'
            try:
                rows = [self.model(input_ids[:, i:i+1], start_pos + i)
                        for i in range(input_ids.shape[1])]
                result = (torch.stack([r[0] for r in rows], 1),
                          torch.stack([r[1] for r in rows], 1),
                          torch.cat([r[2] for r in rows], 1) if rows[0][2] is not None else None)
                if rollback:
                    snapshot.restore()
                return result
            except BaseException:
                snapshot.restore()
                raise
        self.batch_calls += 1
        self.last_mode = 'causal_batch'
        self.active, self.start, self.length = True, start_pos, input_ids.shape[1]
        self.candidates = [None] * self.length
        self.compressed_indices = None
        F.linear = self.linear
        try:
            result = self.model(input_ids, start_pos)
            if rollback:
                snapshot.restore()
            return result
        except BaseException:
            snapshot.restore()
            raise
        finally:
            F.linear = self.original_linear
            self.active = False
            self.candidates = []
            self.compressed_indices = None

    def attention(self, attn, x, start_pos):
        m = self.module
        batch, length, _ = x.shape
        assert batch == 1 and length == self.length and start_pos == self.start
        win, rd = attn.window_size, attn.rope_head_dim
        freqs = attn.freqs_cis[start_pos:start_pos + length]
        qr = attn.q_norm(attn.wq_a(x))
        q = attn.wq_b(qr).unflatten(-1, (attn.n_local_heads, attn.head_dim))
        m.apply_rotary_emb(q[..., -rd:], freqs)
        new_kv = attn.kv_norm(attn.wkv(x))
        m.apply_rotary_emb(new_kv[..., -rd:], freqs)
        m.act_quant(new_kv, m.fp8_block_size, m.scale_fmt, m.scale_dtype, True)
        # Copy the old ring before any proposed token overwrites one of its slots.
        window = torch.cat([attn.window_kv_cache[:batch], new_kv], dim=1)
        window_ids = []
        for offset in range(length):
            position = start_pos + offset
            slots = m.get_window_topk_idxs(win, batch, 1, position).clone()
            newest = slots.long() + ((position - slots.long()) // win) * win
            mapped = torch.where(newest >= start_pos, win + newest - start_pos, slots)
            window_ids.append(torch.where(slots >= 0, mapped, -1).int())
            attn.window_kv_cache[:batch, position % win] = new_kv[:, offset]

        compressed_ids = []
        compressed = None
        if attn.compress_ratio:
            if attn.is_index_source:
                for offset in range(length):
                    position = start_pos + offset
                    latent = None
                    if attn.is_kv_source:
                        latent = attn.compressor(x[:, offset:offset+1], position)
                        m.shared_attn.compress_kv = attn.compress_kv_cache
                    end = position + 1
                    if end // attn.compress_ratio == 0:
                        ids = torch.empty(batch, 1, 0, dtype=torch.int32, device='cpu')
                    else:
                        if attn.indexer.freqs_cis is None:
                            attn.indexer.freqs_cis = attn.freqs_cis
                        if attn.indexer.uses_candidates:
                            m.shared_attn.candidates = self.candidates[offset]
                        ids = attn.indexer(x[:, offset:offset+1], qr[:, offset:offset+1],
                                           latent, position, win + length)
                        if attn.indexer.is_candidate_source:
                            self.candidates[offset] = m.shared_attn.candidates
                    compressed_ids.append(ids)
                    if latent is not None:
                        latent_freqs = attn.freqs_cis[position + 1 - attn.compress_ratio].unsqueeze(0)
                        m.apply_rotary_emb(latent[..., -rd:], latent_freqs)
                        m.fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)
                        at = position // attn.compress_ratio
                        attn.compress_kv_cache[:batch, at:at + latent.shape[1]] = latent
                # Later layers consume each query's exact variable-length indices.
                self.compressed_indices = compressed_ids
                last = compressed_ids[-1]
                m.shared_attn.topk_idxs = torch.where(last >= 0, last - length, -1).int()
            else:
                if attn.is_kv_source:
                    raise NotImplementedError('A KV source without an index source needs separate chunk handling')
                compressed_ids = self.compressed_indices
            compressed = m.shared_attn.compress_kv[:batch, :(start_pos + length) // attn.compress_ratio]
            kv = torch.cat([window, compressed], dim=1)
        else:
            kv = window

        outputs = []
        for offset in range(length):
            ids = window_ids[offset]
            if attn.compress_ratio:
                ids = torch.cat([ids, compressed_ids[offset]], dim=-1)
            outputs.append(m.sparse_attn(q[:, offset:offset+1], kv, attn.attn_sink,
                                         ids, attn.softmax_scale))
        o = torch.cat(outputs, dim=1)
        m.apply_rotary_emb(o[..., -rd:], freqs, True)
        o = o.view(batch, length, attn.n_local_groups, -1)
        wo_a = attn.wo_a.weight.view(attn.n_local_groups, attn.o_lora_rank, -1)
        projected = torch.cat([torch.einsum('bsgd,grd->bsgr', o[:, i:i+1], wo_a)
                               for i in range(length)], dim=1)
        return attn.wo_b(projected.flatten(2))

    def uninstall(self):
        if self.active:
            raise RuntimeError('Cannot uninstall an active chunk verifier')
        for attn, original in self.bindings:
            attn.forward = original
        self.model.head.forward = self.original_head
        self.bindings.clear()
