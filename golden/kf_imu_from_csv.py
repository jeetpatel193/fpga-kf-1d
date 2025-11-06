# golden/kf_imu_from_csv.py
# Runs a 1-D IMU Kalman filter using:
#   u_k  = gyro rate (rad/s)        -> input to PREDICT
#   z_k  = accel-derived angle (rad) -> measurement in UPDATE
#
# Reads CSVs produced by golden/gen_imu.py and writes:
#   data/<scenario>_x_hat.csv
#   data/<scenario>_P.csv
#   plots of angle overlay + covariance

import numpy as np
import argparse, os

def repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def load_series(basepath):
    """Load t, x_true (optional), u_gyro, z_accel from data/<scenario>_*.csv"""
    def load(name, required=True):
        p = basepath + f"_{name}.csv"
        if not os.path.exists(p) and not required:
            return None
        arr = np.loadtxt(p, delimiter=",", skiprows=1)  # skip header written by gen_imu.py
        return np.asarray(arr, dtype=float).reshape(-1)
    t       = load("t")
    x_true  = load("x_true", required=False)
    u_gyro  = load("u_gyro")
    z_accel = load("z_accel")
    n = min(t.size, u_gyro.size, z_accel.size)
    if x_true is not None:
        n = min(n, x_true.size)
    t, u_gyro, z_accel = t[:n], u_gyro[:n], z_accel[:n]
    x_true = x_true[:n] if x_true is not None else None
    return t, x_true, u_gyro, z_accel

def kalman_imu_1d(u_gyro, z_accel, dt, Q, R, x0=None, P0=1.0):
    """
    1-D KF with A=1, B=dt, C=1
      Predict: x^- = x + dt*u,   P^- = P + Q
      Update:  K = P^-/(P^-+R),  x = x^- + K*(z - x^-),  P = (1-K)*P^-
    """
    N = len(z_accel)
    x_hat = np.zeros(N)
    P_arr = np.zeros(N)
    K_arr = np.zeros(N)

    # sensible default: start from first measurement if x0 not provided
    x = float(z_accel[0] if x0 is None else x0)
    P = float(P0)

    for k in range(N):
        # PREDICT with gyro input
        x_pred = x + dt * float(u_gyro[k])
        P_pred = P + Q

        # UPDATE with accel measurement
        denom = P_pred + R
        K = P_pred / denom
        x = x_pred + K * (float(z_accel[k]) - x_pred)
        P = (1.0 - K) * P_pred

        x_hat[k] = x
        P_arr[k] = P
        K_arr[k] = K

    return x_hat, P_arr, K_arr

def main():
    ap = argparse.ArgumentParser(description="Run 1-D IMU Kalman filter on generated CSVs.")
    ap.add_argument("--scenario", choices=["sine","step"], default="sine")
    ap.add_argument("--Q", type=float, default=1e-4, help="process noise variance")
    ap.add_argument("--R", type=float, default=1e-2, help="measurement noise variance")
    ap.add_argument("--x0", type=float, default=None, help="initial angle (rad). default: first measurement")
    ap.add_argument("--P0", type=float, default=1.0, help="initial variance")
    ap.add_argument("--dt", type=float, default=None, help="override dt (s). default: use median diff from t.csv")
    ap.add_argument("--outdir", default=None, help="override output folder (default: repo/data)")
    ap.add_argument("--plots", action="store_true", help="save plots")
    args = ap.parse_args()

    root = repo_root()
    data_dir  = os.path.join(root, "data")
    plots_dir = os.path.join(data_dir, "plots")
    ensure_dirs(data_dir, plots_dir)
    if args.outdir:
        data_dir = args.outdir
        plots_dir = os.path.join(data_dir, "plots")
        ensure_dirs(data_dir, plots_dir)

    base = os.path.join(data_dir, args.scenario)
    t, x_true, u_gyro, z_accel = load_series(base)
    if args.dt is None:
        dt = float(np.median(np.diff(t)))  # robust if t has tiny jitter
    else:
        dt = float(args.dt)

    # Run the filter
    x_hat, P_arr, K_arr = kalman_imu_1d(u_gyro, z_accel, dt, args.Q, args.R, x0=args.x0, P0=args.P0)

    # Save outputs
    np.savetxt(base + "_x_hat.csv", x_hat, delimiter=",", header="x_hat", comments="")
    np.savetxt(base + "_P.csv",     P_arr, delimiter=",", header="P",     comments="")
    np.savetxt(base + "_K.csv",     K_arr, delimiter=",", header="K",     comments="")
    print(f"[OK] Wrote: {os.path.relpath(base + '_x_hat.csv', root)}, _P.csv, _K.csv")

    # Optional plots
    if args.plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Angle overlay: truth (if available) vs accel vs KF
        plt.figure()
        if x_true is not None:
            plt.plot(t, x_true, label="x_true (rad)")
        plt.plot(t, z_accel, label="z_accel (rad)", alpha=0.7)
        plt.plot(t, x_hat,   label="x_hat (KF)", linewidth=1.7)
        plt.xlabel("t (s)"); plt.ylabel("angle (rad)"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{args.scenario}_kf_angle.png")); plt.close()

        # Covariance over time
        plt.figure()
        plt.plot(t, P_arr, label="P (variance)")
        plt.xlabel("t (s)"); plt.ylabel("variance"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{args.scenario}_kf_P.png")); plt.close()

        # Gain (nice for intuition)
        plt.figure()
        plt.plot(t, K_arr, label="K (Kalman gain)")
        plt.xlabel("t (s)"); plt.ylabel("K"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{args.scenario}_kf_K.png")); plt.close()

        print(f"[OK] Plots: data/plots/{args.scenario}_kf_angle.png, _kf_P.png, _kf_K.png")

    # Simple RMSE if truth available
    if x_true is not None:
        rmse_meas = np.sqrt(np.mean((z_accel - x_true)**2))
        rmse_kf   = np.sqrt(np.mean((x_hat   - x_true)**2))
        print(f"[RMSE] accel vs truth: {rmse_meas:.5f}  |  KF vs truth: {rmse_kf:.5f}")

if __name__ == "__main__":
    main()
