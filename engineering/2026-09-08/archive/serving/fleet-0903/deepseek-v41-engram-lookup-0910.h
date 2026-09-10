// SPDX-License-Identifier: MIT
// Standalone native V4.1 Engram lookup. Does not implement projections or a model graph.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const uint32_t * token_map;
    int64_t vocabulary;
    int64_t compressed_vocabulary;
    int64_t pad_token;
    const uint64_t * moduli;       // [2][24]
    const uint64_t * multipliers;  // [2][4]
    const uint64_t * table_rows;   // [2], corresponding to model layers 1 and 14
} deepseek_v41_lookup_config;

typedef struct {
    int32_t layer;                 // Hash layer index 0 or 1, NOT model layer ID.
    uint64_t first_row;
    uint64_t rows;
    const uint8_t * weight;        // Native E4M3, [rows][256]
    uint64_t weight_bytes;
    const uint8_t * scale;         // Native E8M0, [rows][8]
    uint64_t scale_bytes;
} deepseek_v41_lookup_shard;

// The metadata and shard descriptors are copied. Weight/scale memory is borrowed,
// immutable, and must remain readable for the lifetime of this handle and its clones.
// Each layer must be covered exactly once, without gaps, overlaps, or empty shards.
// Forward performs no heap allocation. Scratch space is 48 uint64 IDs per chunk token.
void * deepseek_v41_lookup_create(const deepseek_v41_lookup_config * config,
    const deepseek_v41_lookup_shard * shards, int64_t shard_count,
    int64_t max_sequence, int64_t max_chunk, int reciprocal, int vectorize);
void deepseek_v41_lookup_destroy(void * handle);
void * deepseek_v41_lookup_clone(const void * handle);
int64_t deepseek_v41_lookup_length(const void * handle);
int deepseek_v41_lookup_rewind(void * handle, int64_t position);

// Output shape [tokens][2][24][256], BF16 bits. Optional IDs: [tokens][2][24].
// All input errors leave history and caller outputs unchanged. An empty call does
// not truncate history. Mask 0 stops n-gram lookback; lookup still returns pad-hashed
// rows, as in the official reference. The later residual gate must apply the mask.
// Buffers must be valid, disjoint from each other, and disjoint from borrowed tables.
// Serialize access to each mutable handle. Independent clones may run concurrently.
int deepseek_v41_lookup_forward(void * handle, const int64_t * tokens,
    const uint8_t * mask, int64_t count, int64_t start,
    uint16_t * output, uint64_t output_elements,
    uint64_t * ids_output, uint64_t ids_elements);

#ifdef __cplusplus
}
#endif
