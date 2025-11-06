# golden/gen_imu.py
import numpy as np
import argparse, os

def repo_root():
    # Resolve repo root regardless of current working dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, ".."))

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def make_signals(scenario, dt, T, amp, freq, step_time, step_angle):
    t = np.arange(0.0, T, dt)
    if scenario == "sine":
        x_true = amp * np.sin(2*np.pi*freq*t)          # rad
    elif scenario == "step":
        x_true = np.zeros_like(t)
        x_true[t >= step_time] = step_angle            # rad
    else:
        raise ValueError("scenario must be 'sine' or 'step'")
    omega_true = np.gradient(x_true, dt)               # rad/s
    return t, x_true, omega_true

def main():
    ap = argparse.ArgumentParser(description="Generate synthetic IMU data (gyro rate input + accel angle measurement).")
    ap.add_argument("--scenario", choices=["sine","step"], default="sine")
    ap.add_argument("--dt", type=float, default=0.01)          # 100 Hz
    ap.add_argument("--T", type=float, default=10.0)           # seconds
    ap.add_argument("--amp", type=float, default=0.3)          # rad (sine)
    ap.add_argument("--freq", type=float, default=0.5)         # Hz (sine)
    ap.add_argument("--step_time", type=float, default=1.0)    # s (step)
    ap.add_argument("--step_angle", type=float, default=0.5)   # rad (step)
    ap.add_argument("--sigma_g", type=float, default=0.02)     # rad/s gyro 1σ
    ap.add_argument("--sigma_a", type=float, default=0.05)     # rad accel angle 1σ
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--outdir", default=None, help="override output folder")
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()

    root = repo_root()
    outdir = args.outdir or os.path.join(root, "data")
    plots_dir = os.path.join(outdir, "plots")
    ensure_dirs(outdir, plots_dir)

    rng = np.random.default_rng(args.seed)

    # Ground truth angle + rate
    t, x_true, omega_true = make_signals(
        args.scenario, args.dt, args.T, args.amp, args.freq, args.step_time, args.step_angle
    )

    # IMU measurements
    u_gyro  = omega_true + rng.normal(0.0, args.sigma_g, size=t.size)  # input (rad/s)
    z_accel = x_true     + rng.normal(0.0, args.sigma_a, size=t.size)  # measurement (rad)

    # Save CSVs (with headers)
    base = os.path.join(outdir, f"{args.scenario}")
    np.savetxt(base + "_t.csv",       t,        delimiter=",", header="t",       comments="")
    np.savetxt(base + "_x_true.csv",  x_true,   delimiter=",", header="x_true",  comments="")
    np.savetxt(base + "_u_gyro.csv",  u_gyro,   delimiter=",", header="u_gyro",  comments="")
    np.savetxt(base + "_z_accel.csv", z_accel,  delimiter=",", header="z_accel", comments="")

    print(f"[OK] Wrote IMU CSVs: {os.path.relpath(outdir, root)}  (prefix: {args.scenario}_)")

    if args.plots:
        import matplotlib
        matplotlib.use("Agg")   # headless
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(t, x_true, label="x_true (rad)")
        plt.plot(t, z_accel, label="z_accel (rad)", alpha=0.7)
        plt.xlabel("t (s)"); plt.ylabel("angle (rad)")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{args.scenario}_angle.png")); plt.close()

        plt.figure()
        plt.plot(t, omega_true, label="omega_true (rad/s)")
        plt.plot(t, u_gyro,     label="u_gyro (rad/s)", alpha=0.7)
        plt.xlabel("t (s)"); plt.ylabel("rate (rad/s)")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{args.scenario}_rate.png")); plt.close()

        print(f"[OK] Saved plots: data/plots/{args.scenario}_angle.png, _rate.png")

if __name__ == "__main__":
    main()
