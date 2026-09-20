// topk_select_check.cpp -- the selection top-k with tie fallback against std::partial_sort, as SETS, on masked scores.
// Build: g++ -O3 -march=native -std=c++17 topk_select_check.cpp -o topk_select_check
// The earlier histogram selection (patches/topk-linear-selection.patch) changed the selected set once the finite
// count fell below k, because the tail among equal -inf scores is arbitrary. This variant returns false in exactly
// those cases (a tie across the cut) and the caller keeps partial_sort, so the set can never differ.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

static bool f18_top_k_row(const float * x, int64_t n, int64_t k, int32_t * out, float * tmp) {
    if (k >= n) { for (int64_t i = 0; i < n; ++i) out[i] = (int32_t) i; return k == n; }
    if (n > 16384) return false; // F18_TOPK_MAX_N: partial_sort wins on long rows, see the table this program prints
    for (int64_t i = 0; i < n; ++i) if (x[i] != x[i]) return false;
    memcpy(tmp, x, (size_t) n*sizeof(float));
    std::nth_element(tmp, tmp + (k - 1), tmp + n, std::greater<float>());
    const float thr = tmp[k - 1];
    int64_t n_ge = 0;
    for (int64_t i = 0; i < n; ++i) n_ge += x[i] >= thr;
    if (n_ge != k) return false;
    int64_t j = 0;
    for (int64_t i = 0; i < n; ++i) if (x[i] >= thr) out[j++] = (int32_t) i;
    return true;
}
struct cmp_top_k { const float * data; bool operator()(int32_t a, int32_t b) const { return data[a] > data[b]; } };

int main() {
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 3.0f);
    const int64_t k = 512;
    printf("%9s %8s %10s | %12s %12s | %8s %s\n", "n", "masked", "case", "partial ms", "select ms", "fallback", "set");
    for (int64_t n : {600, 1026, 2048, 7554, 25002, 65536, 262144}) {
        for (int mode = 0; mode < 4; ++mode) {
            // 0: 25% masked  1: finite count < k (cut inside -inf)  2: exact finite ties across the cut  3: no mask
            std::vector<float> x(n); for (auto & v : x) v = nd(rng);
            const char * name = "25% -inf";
            if (mode == 0) { for (int64_t i = 0; i < n; i += 4) x[i] = -INFINITY; }
            if (mode == 1) { for (int64_t i = 0; i < n; ++i) if (i % n >= k/2) x[i] = -INFINITY; name = "finite<k"; }
            if (mode == 2) { for (int64_t i = 0; i < n; ++i) x[i] = (float) (i % 7); name = "ties@cut"; }
            if (mode == 3) name = "no mask";
            std::vector<int32_t> a(n), b(k), idx(n); std::vector<float> tmp(n);
            double tp = 1e9, ts = 1e9; bool ok = false;
            for (int rep = 0; rep < 20; ++rep) {
                auto t0 = std::chrono::steady_clock::now();
                for (int64_t j = 0; j < n; ++j) a[j] = (int32_t) j;
                std::partial_sort(a.begin(), a.begin() + k, a.end(), cmp_top_k{x.data()});
                auto t1 = std::chrono::steady_clock::now();
                ok = f18_top_k_row(x.data(), n, k, b.data(), tmp.data());
                auto t2 = std::chrono::steady_clock::now();
                tp = std::min(tp, std::chrono::duration<double, std::milli>(t1 - t0).count());
                ts = std::min(ts, std::chrono::duration<double, std::milli>(t2 - t1).count());
            }
            const char * verdict = "n/a (caller sorts)";
            if (ok) {
                std::vector<int32_t> sa(a.begin(), a.begin() + k), sb(b.begin(), b.end());
                std::sort(sa.begin(), sa.end()); std::sort(sb.begin(), sb.end());
                verdict = sa == sb ? "IDENTICAL" : "DIFFERENT";
            }
            printf("%9lld %8s %10s | %12.4f %12.4f | %8s %s\n", (long long) n, mode == 3 ? "0%" : "yes", name, tp, ts, ok ? "no" : "YES", verdict);
        }
    }
    return 0;
}
