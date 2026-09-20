// topk_scan_check.cpp -- f18_topk_scan against std::partial_sort as used by llama.cpp's top-k sampler (same candidates, same order)
// build (from the repository root): g++ -O2 -std=c++17 -Iengineering/2026-09-20/archive/serving/fleet-0920-flash18/common tools/topk_scan_check.cpp -o topk_scan_check
#include "f18-topk-scan.inc"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
struct td { int32_t id; float logit; float p; };
int main() {
    std::mt19937 rng(7); std::normal_distribution<float> nd(0.0f, 3.0f);
    const int n = 154880; int n_bad = 0; double t_scan = 0, t_sort = 0; double tc[5] = {0}, ts[5] = {0};
    for (int trial = 0; trial < 300; ++trial) {
        const int k = trial % 3 == 0 ? 40 : (trial % 3 == 1 ? 1 + (int) (rng() % 128) : 10);
        std::vector<float> x(n);
        for (auto & v : x) v = nd(rng);
        if (trial % 5 == 1) for (int i = 0; i < 2000; ++i) x[rng() % n] = -INFINITY;
        if (trial % 5 == 2) for (int i = 0; i < 50; ++i)   x[rng() % n] = NAN;
        if (trial % 5 == 3) { for (auto & v : x) v = std::round(v*4)/4; }       // massive ties
        if (trial % 5 == 4) { for (int i = 0; i < n; ++i) x[i] = (float) i; }   // ascending: worst case for insertion
        float best[F18_FAST_K_MAX]; int32_t best_id[F18_FAST_K_MAX];
        auto t0 = std::chrono::steady_clock::now();
        const int m = f18_topk_scan(x.data(), n, k, best, best_id);
        auto t1 = std::chrono::steady_clock::now();
        std::vector<td> cur(n);
        for (int i = 0; i < n; ++i) cur[i] = { i, x[i], 0.0f };
        const bool has_nan = trial % 5 == 2;
        if (has_nan) { cur.erase(std::remove_if(cur.begin(), cur.end(), [](const td & a) { return a.logit != a.logit; }), cur.end()); }
        auto t2 = std::chrono::steady_clock::now();
        std::partial_sort(cur.begin(), cur.begin() + k, cur.end(), [](const td & a, const td & b) { return a.logit > b.logit; });
        auto t3 = std::chrono::steady_clock::now();
        tc[trial % 5] += std::chrono::duration<double, std::micro>(t1 - t0).count(); ts[trial % 5] += std::chrono::duration<double, std::micro>(t3 - t2).count(); t_scan += std::chrono::duration<double, std::micro>(t1 - t0).count(); t_sort += std::chrono::duration<double, std::micro>(t3 - t2).count();
        const bool asc = trial % 5 == 4;
        bool ok = asc ? m == -1 : m == k;   // the ascending row must be declined, everything else answered
        if (asc) { if (!ok) { n_bad++; printf("trial %d: ascending row not declined\n", trial); } continue; }
        for (int i = 0; i < k && ok; ++i) ok = best[i] == cur[i].logit;                 // same values in the same order
        const bool ties = trial % 5 == 3;
        if (!ties) for (int i = 0; i < k && ok; ++i) ok = best_id[i] == cur[i].id;       // and the same tokens when values are distinct
        if (ties)  for (int i = 0; i < k && ok; ++i) ok = x[best_id[i]] == best[i] && (i == 0 || best[i] < best[i-1] || best_id[i] > best_id[i-1]);
        if (!ok) { n_bad++; printf("trial %d k=%d MISMATCH\n", trial, k); }
    }
    printf("300 trials (k 1..128, -inf, NaN, ties, ascending): %d mismatches; scan %.1f us, partial_sort %.1f us per row\n", n_bad, t_scan/300, t_sort/300);
    const char * nm[5] = {"gaussian", "with -inf", "with NaN", "ties (quantised)", "ascending"};
    for (int c = 0; c < 5; ++c) printf("  %-18s scan %8.1f us   partial_sort %8.1f us\n", nm[c], tc[c]/60, ts[c]/60);
    return n_bad != 0;
}
