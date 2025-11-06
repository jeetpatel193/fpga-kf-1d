# golden/compare_golden.py
# Compare candidate CSVs against golden CSVs with a tolerance.
# Works for any 1-D array CSVs (one column, header in row 1).

import argparse, os
import numpy as np

def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    return np.asarray(arr, dtype=float).reshape(-1)

def stats(ref, cand):
    err = cand - ref
    mae  = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    maxe = float(np.max(np.abs(err)))
    idx  = int(np.argmax(np.abs(err)))
    return mae, rmse, maxe, idx

def main():
    ap = argparse.ArgumentParser(description="Compare candidate CSVs against golden CSVs.")
    ap.add_argument("--ref",   required=True, help="reference (golden) CSV")
    ap.add_argument("--cand",  required=True, help="candidate CSV (e.g., HLS output)")
    ap.add_argument("--eps",   type=float, default=1e-6, help="tolerance for PASS")
    ap.add_argument("--label", default="", help="optional label to print (x_hat, P, etc.)")
    args = ap.parse_args()

    ref  = load_csv(args.ref)
    cand = load_csv(args.cand)
    if ref.shape != cand.shape:
        print(f"[FAIL] shape mismatch: ref {ref.shape} vs cand {cand.shape}")
        return 2

    mae, rmse, maxe, idx = stats(ref, cand)
    ok = maxe <= args.eps

    tag = f"[{args.label}] " if args.label else ""
    print(f"{tag}N={ref.size}  eps={args.eps:.1e}  MAE={mae:.3e}  RMSE={rmse:.3e}  MAX={maxe:.3e} @ idx={idx}")
    print("[PASS]" if ok else "[FAIL]")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
