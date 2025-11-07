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
static bool load_csv_1col(const std::string &path, std::vector<float>& out) {
    FILE *f = fopen(path.c_str(), "r");
    if (!f) return false;
    char line[512];
    bool first = true;
    while (fgets(line, sizeof(line), f)) {
        if (first) { first = false; continue; } // skip header
        float v;
        if (sscanf(line, "%f", &v) == 1) out.push_back(v);
    }
    fclose(f);
    return !out.empty();
}

int main() {
    std::string base = "../../../../data/sine"; // relative to vitis_hls csim build dir
    std::vector<float> t, u, z, truth;

    bool ok_t  = load_csv_1col(base + "_t.csv",      t);
    bool ok_u  = load_csv_1col(base + "_u_gyro.csv", u);
    bool ok_z  = load_csv_1col(base + "_z_accel.csv", z);
    bool ok_x  = load_csv_1col(base + "_x_true.csv", truth); // optional

    int N = 0;
    if (ok_u && ok_z) {
        N = std::min((int)u.size(), (int)z.size());
        u.resize(N); z.resize(N);
        if (ok_t) t.resize(N);
        if (ok_x) truth.resize(N);
        std::cout << "[tb] Using CSVs from data/sine_* (N=" << N << ")\n";
    } else {
        // fallback synthetic mini-test
        N = 64;
        float dt_syn = 0.01f;
        u.resize(N, 0.0f);
        z.resize(N, 0.0f);
        t.resize(N);
        for (int i=0; i<N; ++i) {
            t[i] = i*dt_syn;
            z[i] = (i<32 ? 0.0f : 0.5f); // step angle
            u[i] = 0.0f;                 // zero rate input
        }
        std::cout << "[tb] No CSVs found; using synthetic N=64 step test.\n";
    }

    // dt: use median dt from t if available; else 0.01
    float dt = 0.01f;
    if (!t.empty()) {
        std::vector<float> diffs;
        for (int i=1; i<(int)t.size(); ++i) diffs.push_back(t[i]-t[i-1]);
        std::sort(diffs.begin(), diffs.end());
        dt = diffs[diffs.size()/2];
    }

    // KF params
    float Q  = 1e-4f;
    float R  = 1e-2f;
    float x0 = (z.empty()? 0.0f : z[0]);
    float P0 = 1.0f;

    std::vector<float> x_out(N), P_out(N);

    // call kernel like a function (C-sim)
    kf1d_float(u.data(), z.data(), dt, Q, R, x0, P0, N, x_out.data(), P_out.data());

    // quick print
    std::cout << "[tb] Ran kf1d_float: N=" << N << " dt=" << dt << " Q=" << Q << " R=" << R << "\n";

    // If truth is available, print simple RMSE
    if (!truth.empty()) {
        double se=0.0;
        for (int i=0; i<N; ++i) {
            double d = (double)x_out[i] - (double)truth[i];
            se += d*d;
        }
        double rmse = std::sqrt(se / N);
        std::cout << "[tb] RMSE(KF vs truth) ≈ " << rmse << "\n";
    }

    // success
    return 0;
}
