# golden/golden_kf.py
import argparse, os
import numpy as np
import matplotlib.pyplot as plt

def kalman_1d(z, a=1.0, u=0.0, Q=1e-4, R=1e-2, x0=0.0, P0=1.0):
    """Scalar (1-D) Kalman filter."""
    x, P = x0, P0
    xs, Ps, Ks = np.zeros_like(z), np.zeros_like(z), np.zeros_like(z)
    for i, zi in enumerate(z):
        # predict
        x_pred = a * x + u
        P_pred = (a * a) * P + Q
        # update
        K = P_pred / (P_pred + R)
        x = x_pred + K * (zi - x_pred)
        P = (1.0 - K) * P_pred
        xs[i], Ps[i], Ks[i] = x, P, K
    return xs, Ps, Ks

def synth_const(N=400, true_val=5.0, noise_std=0.1, seed=0):
    rng = np.random.default_rng(seed)
    z = true_val + rng.normal(0.0, noise_std, N).astype(np.float32)
    return z

def synth_ramp(N=400, slope=0.01, intercept=0.0, noise_std=0.1, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(N, dtype=np.float32)
    truth = slope * t + intercept
    z = truth + rng.normal(0.0, noise_std, N).astype(np.float32)
    return z, truth

def write_csv(path, arr):
    np.savetxt(path, np.asarray(arr, dtype=np.float32), delimiter=",", fmt="%.6f")

def plot_series(path, z, xhat, truth=None, title=""):
    plt.figure(figsize=(8,3))
    plt.plot(z, label="measurement z", linewidth=1)
    plt.plot(xhat, label="KF estimate x̂", linewidth=1.5)
    if truth is not None:
        plt.plot(truth, label="ground truth", linewidth=1.5)
    plt.title(title); plt.xlabel("sample"); plt.ylabel("value"); plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="../data", help="output directory")
    parser.add_argument("--Q", type=float, default=1e-4)
    parser.add_argument("--R", type=float, default=1e-2)
    parser.add_argument("--x0", type=float, default=0.0)
    parser.add_argument("--P0", type=float, default=1.0)
    args = parser.parse_args()

    outdir = args.outdir
    plots_dir = os.path.join(outdir, "plots")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # --- Case 1: constant true value ---
    z_const = synth_const(N=400, true_val=5.0, noise_std=0.1, seed=0)
    x_const, P_const, _ = kalman_1d(z_const, a=1.0, Q=args.Q, R=args.R, x0=args.x0, P0=args.P0)
    write_csv(os.path.join(outdir, "z_const.csv"), z_const)
    write_csv(os.path.join(outdir, "x_out_const.csv"), x_const)
    write_csv(os.path.join(outdir, "P_out_const.csv"), P_const)
    plot_series(os.path.join(plots_dir, "const_plot.png"), z_const, x_const, truth=np.full_like(z_const, 5.0), title="Constant Signal")

    # --- Case 2: ramp true value ---
    z_ramp, truth_ramp = synth_ramp(N=400, slope=0.01, intercept=0.0, noise_std=0.1, seed=1)
    x_ramp, P_ramp, _ = kalman_1d(z_ramp, a=1.0, Q=args.Q, R=args.R, x0=args.x0, P0=args.P0)
    write_csv(os.path.join(outdir, "z_ramp.csv"), z_ramp)
    write_csv(os.path.join(outdir, "x_out_ramp.csv"), x_ramp)
    write_csv(os.path.join(outdir, "P_out_ramp.csv"), P_ramp)
    plot_series(os.path.join(plots_dir, "ramp_plot.png"), z_ramp, x_ramp, truth=truth_ramp, title="Ramp Signal")

    print("[OK] Wrote CSVs to:", outdir)
    print("[OK] Wrote plots to:", plots_dir)

if __name__ == "__main__":
    main()
