"""Bounded native-format bridge; no requantization of DeepSeek's FP4 experts.

The producer stores adjacent FP4 elements in a byte. ggml block_mxfp4 stores
elements j and j+16 together after the original E8M0 scale byte.
"""


def repack_mxfp4_rows(packed, scales, rows, cols):
    if rows <= 0 or cols <= 0 or cols % 32:
        raise ValueError('FP4 rows must have a positive multiple of 32 columns')
    blocks = rows * (cols // 32)
    if len(packed) != blocks * 16 or len(scales) != blocks:
        raise ValueError('FP4 source or scale size does not match its shape')
    output = bytearray(blocks * 17)
    for block in range(blocks):
        output[block * 17] = scales[block]
        offset = block * 16
        for j in range(16):
            lo = (packed[offset + j // 2] >> ((j & 1) * 4)) & 15
            hi = (packed[offset + 8 + j // 2] >> ((j & 1) * 4)) & 15
            output[block * 17 + 1 + j] = lo | (hi << 4)
    return bytes(output)


def unpack_mxfp4_rows(raw, rows, cols):
    if rows <= 0 or cols <= 0 or cols % 32:
        raise ValueError('FP4 rows must have a positive multiple of 32 columns')
    blocks = rows * (cols // 32)
    if len(raw) != blocks * 17:
        raise ValueError('MXFP4 input size does not match its shape')
    packed, scales = bytearray(blocks * 16), bytearray(blocks)
    for block in range(blocks):
        scales[block] = raw[block * 17]
        for j in range(32):
            code = (raw[block * 17 + 1 + (j % 16)] >> (4 * (j // 16))) & 15
            packed[block * 16 + j // 2] |= code << (4 * (j & 1))
    return bytes(packed), bytes(scales)
