// coupled_sampling_check.cpp -- is argmax(logit + G(salt, n, id)) an exact sampler, and how much does coupling buy?
//
//   1. marginal: chi-square of pick frequencies against softmax(logits), over fresh salts and over consecutive counters
//   2. independence: consecutive positions under one salt (pair table against the product law)
//   3. coupling: P(draft == verifier) for a drafter distribution q near p, against a greedy drafter and the optimum 1 - TV
//
// build: g++ -O2 -std=c++17 -I.. coupled_sampling_check.cpp -o coupled_sampling_check
#include "f18-coupling.inc"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

bool common_sampler_f18_coupling_find(const int32_t *, size_t, f18_coupling &) { return false; }

static int pick(const std::vector<double> & logit, const std::vector<uint32_t> & id, uint64_t salt, uint64_t n) {
    double best = -INFINITY; int bi = -1;
    for (size_t i = 0; i < logit.size(); ++i) {
        const double v = logit[i] + f18_gumbel(salt, n, id[i]);
        if (v > best) { best = v; bi = (int) i; }
    }
    return bi;
}

static std::vector<double> softmax(const std::vector<double> & l, double temp = 1.0) {
    std::vector<double> p(l.size()); double m = *std::max_element(l.begin(), l.end()), s = 0;
    for (size_t i = 0; i < l.size(); ++i) { p[i] = exp((l[i] - m)/temp); s += p[i]; }
    for (auto & x : p) x /= s;
    return p;
}

// chi-square survival function via the Wilson-Hilferty normal approximation (df >= 3 here)
static double chi2_p(double x, int df) {
    const double z = (pow(x/df, 1.0/3.0) - (1.0 - 2.0/(9.0*df))) / sqrt(2.0/(9.0*df));
    return 0.5*erfc(z/sqrt(2.0));
}

int main() {
    std::mt19937_64 rng(12345);
    int n_fail = 0;

    struct dist_case { const char * name; std::vector<double> logit; };
    std::vector<dist_case> cases;
    { dist_case c{"peaked 0.90", {}}; c.logit = { log(0.90), log(0.05), log(0.03), log(0.015), log(0.005) }; cases.push_back(c); }
    { dist_case c{"uniform 5", {0, 0, 0, 0, 0}}; cases.push_back(c); }
    { dist_case c{"zipf 40", {}}; for (int i = 0; i < 40; ++i) c.logit.push_back(-1.1*log(1.0 + i)); cases.push_back(c); }
    { dist_case c{"two-way 0.55/0.45", { log(0.55), log(0.45) }}; cases.push_back(c); }
    { dist_case c{"tail 1e-4", { 0.0, log(1e-4), log(1e-4), log(1e-4) }}; cases.push_back(c); }

    // 1a. fresh salt per draw, realistic token ids and counters
    // 1b. ONE salt, consecutive counters (what a single request does)
    for (int mode = 0; mode < 2; ++mode) {
        for (const auto & c : cases) {
            const size_t k = c.logit.size();
            std::vector<uint32_t> id(k);
            for (auto & x : id) x = (uint32_t) (rng() % 154880);
            std::sort(id.begin(), id.end()); id.erase(std::unique(id.begin(), id.end()), id.end());
            if (id.size() != k) { for (size_t i = 0; i < k; ++i) id.push_back(1000 + (uint32_t) i); id.resize(k); }
            const auto p = softmax(c.logit);
            const int64_t N = 4000000;
            std::vector<int64_t> cnt(k, 0);
            const uint64_t salt0 = rng();
            for (int64_t t = 0; t < N; ++t) {
                const uint64_t salt = mode == 0 ? rng() : salt0;
                const uint64_t n    = mode == 0 ? 3000 + (rng() % 100000) : 4000 + (uint64_t) t;
                cnt[pick(c.logit, id, salt, n)]++;
            }
            double chi = 0; int df = -1;
            for (size_t i = 0; i < k; ++i) {
                const double e = p[i]*N;
                if (e < 5) continue;
                chi += (cnt[i] - e)*(cnt[i] - e)/e; df++;
            }
            const double pv = df >= 1 ? chi2_p(chi, df) : 1.0;
            const bool ok = pv > 1e-4;
            n_fail += !ok;
            printf("marginal %-10s %-18s N=%lld chi2=%8.2f df=%2d p=%.4f  top: want %.6f got %.6f  %s\n", mode == 0 ? "fresh-salt" : "one-salt", c.name,
                   (long long) N, chi, df, pv, p[0], (double) cnt[0]/N, ok ? "ok" : "FAIL");
        }
    }

    // 2. independence of consecutive positions under one salt
    {
        const std::vector<double> l = { log(0.5), log(0.3), log(0.2) };
        const std::vector<uint32_t> id = { 11, 4242, 150001 };
        const auto p = softmax(l);
        const int64_t N = 3000000;
        std::vector<int64_t> pair(9, 0);
        for (int64_t t = 0; t < N; ++t) {
            const uint64_t salt = rng();
            const uint64_t n = 5000 + (rng() % 1000);
            pair[pick(l, id, salt, n)*3 + pick(l, id, salt, n + 1)]++;
        }
        double chi = 0;
        for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) {
            const double e = p[a]*p[b]*N;
            chi += (pair[a*3 + b] - e)*(pair[a*3 + b] - e)/e;
        }
        const double pv = chi2_p(chi, 8);
        const bool ok = pv > 1e-4;
        n_fail += !ok;
        printf("independence of positions n, n+1: chi2=%.2f df=8 p=%.4f  %s\n", chi, pv, ok ? "ok" : "FAIL");
    }

    // 3. what coupling buys: the verifier draws from p, the drafter only knows q = softmax(logit_p + noise*sigma)
    printf("\n%-8s %-10s %10s %10s %10s %10s\n", "entropy", "q-noise", "greedy", "coupled", "1-TV", "agree-max");
    for (double scale : { 0.5, 1.5, 3.0 }) {          // logit spread: small = high entropy
        for (double sigma : { 0.0, 0.3, 0.7, 1.2 }) {
            const int K = 12; const int64_t N = 200000;
            double acc_greedy = 0, acc_coupled = 0, one_minus_tv = 0, agree = 0;
            std::normal_distribution<double> nd(0.0, 1.0);
            for (int64_t t = 0; t < N; ++t) {
                std::vector<double> lp(K), lq(K); std::vector<uint32_t> id(K);
                for (int i = 0; i < K; ++i) { lp[i] = scale*nd(rng); lq[i] = lp[i] + sigma*nd(rng); id[i] = 100 + 7*i; }
                const auto p = softmax(lp); const auto q = softmax(lq);
                const int am_q = (int) (std::max_element(lq.begin(), lq.end()) - lq.begin());
                const int am_p = (int) (std::max_element(lp.begin(), lp.end()) - lp.begin());
                const uint64_t salt = rng(), n = 4000 + t;
                const int x = pick(lp, id, salt, n);          // the verifier's exact sample
                acc_greedy  += x == am_q;
                acc_coupled += x == pick(lq, id, salt, n);
                agree       += am_p == am_q;
                double tv = 0; for (int i = 0; i < K; ++i) tv += fabs(p[i] - q[i]);
                one_minus_tv += 1.0 - 0.5*tv;
            }
            printf("%-8.1f %-10.1f %10.4f %10.4f %10.4f %10.4f\n", scale, sigma, acc_greedy/N, acc_coupled/N, one_minus_tv/N, agree/N);
        }
    }

    printf("\n%s\n", n_fail ? "FAILED" : "ALL EXACTNESS CHECKS PASSED");
    return n_fail != 0;
}
