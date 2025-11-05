# FPGA-Accelerated 1D Kalman Filter (Golden Model for Update 1)

## What this is
A runnable **Python golden model** of a 1-D Kalman Filter (predict/update loop) that generates:
- synthetic inputs (`data/z_const.csv`, `data/z_ramp.csv`)
- **golden outputs** (`data/x_out_*.csv`, `data/P_out_*.csv`)
- plots (`data/plots/*.png`)

This is the baseline implementation for Project Update 1.

## Quick start
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python golden/golden_kf.py --Q 1e-4 --R 1e-2 --x0 0.0 --P0 1.0
