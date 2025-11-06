# FPGA-Accelerated 1D Kalman Filter (Golden Model for Project Update 1)

Single-axis (1-D) **IMU Kalman filter** baseline for EECE 5698.

**Model:** \(A=1,\; B=\Delta t,\; C=1\)  

This repo produces **golden CSVs & plots** to verify the HLS kernel later on Alveo U280 (OCT).

---

## Quick start

```bash
# 1) Python env & deps
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Generate IMU datasets (time, truth, gyro input, accel measurement)
python golden/gen_imu.py --scenario sine --plots
python golden/gen_imu.py --scenario step --plots

# 3) Run the IMU 1-D Kalman filter (writes x_hat, P, K + plots + RMSE)
python golden/kf_imu_from_csv.py --scenario sine --Q 1e-4 --R 1e-2 --P0 1.0 --plots
python golden/kf_imu_from_csv.py --scenario step --Q 1e-4 --R 1e-2 --P0 1.0 --plots

# 4) Sanity: golden vs golden (PASS)
python golden/compare_golden.py --ref data/sine_x_hat.csv --cand data/sine_x_hat.csv --eps 1e-9 --label x_hat
python golden/compare_golden.py --ref data/sine_P.csv     --cand data/sine_P.csv     --eps 1e-9 --label P

