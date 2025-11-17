// hls/kf1d_float/tb_kf1d.cpp
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <algorithm>

// kernel prototype
extern "C" void kf1d_float(
    const float *u_gyro,
    const float *z_accel,
    float dt, float Q, float R, float x0, float P0, int N,
    float *x_out, float *P_out
);

// simple CSV loader (one-column with header)
static bool load_csv_1col(const std::string &path, std::vector<float> &out) {
    FILE *f = fopen(path.c_str(), "r");
    if (!f) return false;

    char line[512];
    bool first = true;
    while (fgets(line, sizeof(line), f)) {
        if (first) { first = false; continue; }  // skip header
        float v;
        if (sscanf(line, "%f", &v) == 1)
            out.push_back(v);
    }
    fclose(f);
    return !out.empty();
}

struct Metrics {
    double mae;
    double rmse;
    double maxabs;
    int    first_bad;
    int    n;
};

static Metrics compare_vecs(const std::vector<float> &ref,
                            const std::vector<float> &got,
                            double eps_abs) {
    Metrics m{0, 0, 0, -1, (int)std::min(ref.size(), got.size())};
    double se   = 0.0;
    double ae   = 0.0;
    double maxa = 0.0;

    for (int i = 0; i < m.n; ++i) {
        double d  = (double)got[i] - (double)ref[i];
        double ad = std::fabs(d);
        ae += ad;
        se += d * d;
        if (ad > maxa) maxa = ad;
        if (m.first_bad < 0 && ad > eps_abs) m.first_bad = i;
    }

    int denom = std::max(1, m.n);
    m.mae     = ae / denom;
    m.rmse    = std::sqrt(se / denom);
    m.maxabs  = maxa;
    return m;
}

int main() {
    // NOTE: this base path is relative to the csim build directory
    // (kf1d_float_prj/sol1/csim/build)
    std::string base = "../../../../../data/sine";

    std::vector<float> t, u, z, truth;

    bool ok_t = load_csv_1col(base + "_t.csv",       t);
    bool ok_u = load_csv_1col(base + "_u_gyro.csv",  u);
    bool ok_z = load_csv_1col(base + "_z_accel.csv", z);
    bool ok_x = load_csv_1col(base + "_x_true.csv",  truth); // optional

    int N = 0;
    if (ok_u && ok_z) {
        N = std::min((int)u.size(), (int)z.size());
        u.resize(N);
        z.resize(N);
        if (ok_t) t.resize(N);
        if (ok_x) truth.resize(N);
        std::cout << "[tb] Using CSVs from data/sine_* (N=" << N << ")\n";
    } else {
        // fallback synthetic mini-test
        N = 64;
        float dt_syn = 0.01f;
        u.assign(N, 0.0f);
        z.assign(N, 0.0f);
        t.resize(N);
        for (int i = 0; i < N; ++i) {
            t[i] = i * dt_syn;
            z[i] = (i < 32 ? 0.0f : 0.5f); // step angle
            u[i] = 0.0f;                  // zero rate input
        }
        std::cout << "[tb] No input CSVs found; using synthetic N=64 step test.\n";
    }

    // dt: use median dt from t if available; else 0.01
    float dt = 0.01f;
    if (!t.empty()) {
        std::vector<float> diffs;
        diffs.reserve(t.size());
        for (size_t i = 1; i < t.size(); ++i)
            diffs.push_back(t[i] - t[i - 1]);
        std::sort(diffs.begin(), diffs.end());
        dt = diffs[diffs.size() / 2];
    }

    // KF params
    float Q  = 1e-4f;
    float R  = 1e-2f;
    float x0 = (z.empty() ? 0.0f : z[0]);
    float P0 = 1.0f;

    std::vector<float> x_out(N), P_out(N);

    // ----------------------------------------------------------------------
    // 1) RUN THE KERNEL
    // ----------------------------------------------------------------------
    kf1d_float(u.data(), z.data(), dt, Q, R, x0, P0, N,
               x_out.data(), P_out.data());

    std::cout << "[tb] Ran kf1d_float: N=" << N
              << " dt=" << dt << " Q=" << Q << " R=" << R << "\n";

    // ----------------------------------------------------------------------
    // 2) LOAD GOLDEN OUTPUTS AND COMPARE
    // ----------------------------------------------------------------------
    std::vector<float> ref_x, ref_P;
    bool ok_ref_x = load_csv_1col(base + "_x_hat.csv", ref_x);
    bool ok_ref_P = load_csv_1col(base + "_P.csv",     ref_P);

    if (!ok_ref_x || !ok_ref_P) {
        std::cerr << "[tb] WARNING: could not open golden CSVs "
                     "(sine_x_hat.csv or sine_P.csv). Skipping PASS/FAIL.\n";

        // Still print a little bit to see *some* output.
        std::cout << "[tb] First 5 x_out samples:\n";
        for (int i = 0; i < std::min(5, N); ++i)
            std::cout << "  x_out[" << i << "] = " << x_out[i] << "\n";
        return 0;
    }

    const double EPS = 1e-6;
    std::cout << "[tb] Expectation: kernel outputs must match golden "
              << "within eps=" << EPS << " (dataset=sine)\n";

    Metrics mx = compare_vecs(ref_x, x_out, EPS);
    Metrics mP = compare_vecs(ref_P, P_out, EPS);

    std::cout << "[tb] x_hat: MAE="  << mx.mae
              << " RMSE=" << mx.rmse
              << " MAX="  << mx.maxabs
              << "  -> "  << (mx.first_bad < 0 ? "PASS" : "FAIL") << "\n";
    if (mx.first_bad >= 0) {
        int i = mx.first_bad;
        std::cout << "      first mismatch @i=" << i
                  << " got=" << x_out[i]
                  << " ref=" << ref_x[i]
                  << " |diff|=" << std::fabs(x_out[i] - ref_x[i]) << "\n";
    }

    std::cout << "[tb] P    : MAE="  << mP.mae
              << " RMSE=" << mP.rmse
              << " MAX="  << mP.maxabs
              << "  -> "  << (mP.first_bad < 0 ? "PASS" : "FAIL") << "\n";
    if (mP.first_bad >= 0) {
        int i = mP.first_bad;
        std::cout << "      first mismatch @i=" << i
                  << " got=" << P_out[i]
                  << " ref=" << ref_P[i]
                  << " |diff|=" << std::fabs(P_out[i] - ref_P[i]) << "\n";
    }

    bool ok = (mx.first_bad < 0) && (mP.first_bad < 0);
    std::cout << "[tb] SUMMARY: " << (ok ? "PASS" : "FAIL") << std::endl;

    // Optional: RMSE vs ground-truth x_true, if available
    if (!truth.empty()) {
        int M = std::min((int)truth.size(), N);
        double se = 0.0;
        for (int i = 0; i < M; ++i) {
            double d = (double)x_out[i] - (double)truth[i];
            se += d * d;
        }
        double rmse_truth = std::sqrt(se / std::max(1, M));
        std::cout << "[tb] RMSE(KF vs x_true) ≈ " << rmse_truth << "\n";
    }

    return ok ? 0 : 1;
}
