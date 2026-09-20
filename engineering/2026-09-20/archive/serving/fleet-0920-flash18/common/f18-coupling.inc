#pragma once
// Coupled sampling for speculative decoding (Gumbel-max with counter-based noise).
//
// The verifier picks   argmax_i( log p_i + G(salt, n, id_i) )   over the candidates its sampler chain left, where G is
// standard Gumbel noise that is a pure function of (salt, n, token id): salt is fresh per request, n counts the tokens the
// sampler has accepted. By the Gumbel-max theorem that is an exact sample from p, and nothing the drafter does can change it.
// A drafter that knows (salt, n) adds THE SAME noise to its own log q and takes the argmax, so both sides tend to land on the
// same token whenever q is close to p -- including at high-entropy positions, where a greedy draft is accepted only with
// probability p(argmax q).
//
// Included by sampling.cpp (verifier) and speculative.cpp (drafter); both live in libllama-common.

#include <cmath>
#include <cstdint>

static inline uint64_t f18_mix64(uint64_t x) { // splitmix64 finalizer: a bijection with full avalanche
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Gumbel(0,1) for (salt, position, token id)
static inline double f18_gumbel(uint64_t salt, uint64_t n, uint32_t id) {
    const uint64_t h = f18_mix64(f18_mix64(f18_mix64(salt) + n) + id);
    const double   u = ((double) (h >> 11) + 0.5) * (1.0 / 9007199254740992.0); // (0, 1), 53 bits
    return -std::log(-std::log(u));
}

struct f18_coupling {
    uint64_t salt  = 0;
    uint64_t n_acc = 0;   // tokens accepted by the verifier's sampler so far = noise index of the next token
    float    temp  = 1.0f;
    float    top_p = 1.0f;
    float    min_p = 0.0f;
    int32_t  top_k = 0;
};

// sampling.cpp: finds the one live verifier sampler whose accepted-token tail equals `tail` (oldest first). False when none or several match.
bool common_sampler_f18_coupling_find(const int32_t * tail, size_t n_tail, f18_coupling & out);
