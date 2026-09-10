"""Reset MTP cross-batch carry only when a sequence starts at position zero."""


def transform(source):
    marker = 'struct common_speculative_impl_draft_mtp : public common_speculative_impl {'
    assert source.count(marker) == 1
    before, mtp = source.split(marker, 1)
    old = '''                set_h(i_batch_beg[seq_id], pending_h[seq_id].data());'''
    new = '''                // Position zero has no predecessor hidden row. A previous request's
                // carry must not seed a fresh sequence, even if its KV was erased.
                if (batch_in.pos[i_batch_beg[seq_id]] == 0) {
                    std::fill(pending_h[seq_id].begin(), pending_h[seq_id].end(), 0.0f);
                }
                set_h(i_batch_beg[seq_id], pending_h[seq_id].data());'''
    assert mtp.count(old) == 1 and 'A previous request\'s' not in mtp
    return before + marker + mtp.replace(old, new, 1)
