// topk_radix_check.cpp -- the top-k select path above F18_TOPK_MAX_N (radix select) against std::partial_sort, on rows with ties
// and -inf entries. The selected SET must be identical whenever f18_top_k_row returns true; when it returns false the row must
// really have a tie across the cut (or a NaN). Also times both.
//
// build: c++ -std=c++17 -O2 -march=native -I<engine>/ggml/src/ggml-cpu topk_radix_check.cpp -o topk_radix_check
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <set>
#include <vector>
#include "../topk-fast.inc"

static std::vector<int32_t> reference(const std::vector<float> & x, int64_t k) {
    std::vector<int32_t> idx(x.size()); std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int32_t a, int32_t b) { return x[a] > x[b]; });
    idx.resize(k); return idx;
}

int main() {
    std::mt19937_64 rng(7);
    int failures = 0;
    for (int64_t n : {20000L, 34178L, 137000L}) {
        for (int mode = 0; mode < 3; ++mode) {
            const int64_t k = 2051;
            std::vector<float> x(n);
            std::normal_distribution<float> nd(0.0f, 1.0f); std::uniform_int_distribution<int> coin(0, 9);
            for (auto & v : x) {
                v = nd(rng);
                if (mode >= 1 && coin(rng) == 0) v = -INFINITY;          // masked pools
                if (mode == 2 && coin(rng) < 5) v = 0.0f;                // relu zeros: heavy ties
                if (mode == 2 && coin(rng) == 0) v = 0.25f;              // a tie band near the cut
            }
            std::vector<int32_t> out(n); std::vector<float> tmp(n);
            auto t0 = std::chrono::steady_clock::now();
            const bool ok = f18_top_k_row(x.data(), n, k, out.data(), tmp.data());
            const double ms_sel = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            t0 = std::chrono::steady_clock::now();
            const auto ref = reference(x, k);
            const double ms_ref = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::set<int32_t> sref(ref.begin(), ref.end());
            // the k-th largest value and whether it is tied across the cut
            std::vector<float> sorted(x); std::nth_element(sorted.begin(), sorted.begin() + (k - 1), sorted.end(), std::greater<float>());
            const float thr = sorted[k - 1];
            int64_t n_ge = 0; for (float v : x) n_ge += v >= thr;
            const bool tie = n_ge != k;
            bool pass;
            if (ok) {
                std::set<int32_t> ssel(out.begin(), out.begin() + k);
                pass = !tie && ssel == sref;
            } else {
                pass = tie;
            }
            failures += !pass;
            printf("n %6ld mode %d: select %s (%s) %.3f ms | partial_sort %.3f ms | k-th value %g, tie across cut %s -> %s\n",
                   (long) n, mode, ok ? "returned set" : "fell back", ok ? "set identical" : "tie present", ms_sel, ms_ref, thr,
                   tie ? "yes" : "no", pass ? "PASS" : "FAIL");
        }
    }
    printf("%s\n", failures ? "FAILURES" : "all passed");
    return failures ? 1 : 0;
}
