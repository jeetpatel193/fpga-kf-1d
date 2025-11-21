# FPGA-Accelerated 1D Kalman Filter (Golden Model for Project Update 1)

Single-axis (1-D) **IMU Kalman filter** baseline for EECE 5698.

**Model:** (A=1; B=Delta t; C=1)  

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

---

# 5) Csim and Csyn on Vitis Unified IDE 

git clone the directory
run the following command from the main "fpga-kf-1d" directory
vitis -w /home/pateljeet3/projects/fpga-kf-1d &
Once the IDE opens in that workspace, continue with:

File → New Component → HLS
component name: kf1d_float
comonent location: Keep it default which in my case was from the main "fpga-kf-1d" directory
Top: kf1d_float
Sources: hls/kf1d_float/kf1d_float.cpp
Test bench: hls/kf1d_float/tb_kf1d.cpp
Part: xcu280-fsvh2892-2L-e
Clock: 250MHz

"If the flow doesn’t show up restart vitis."

Run C Simulation, then C Synthesis.

You can see the expected PASS output
and the Synthesis report that I got is in the report.

## Steps to run on the hardware:

source /tools/Xilinx/Vitis/2023.1/settings64.sh
export XILINX_XRT=/opt/xilinx/xrt
source $XILINX_XRT/setup.sh

# Choose U280 platform for hardware run.
ls /opt/xilinx/platforms

You’ll see something like:
•	xilinx_u280_gen3x16_xdma_1_202211_1

export PLATFORM=/opt/xilinx/platforms/xilinx_u280_gen3x16_xdma_1_202211_1/xilinx_u280_gen3x16_xdma_1_202211_1.xpfm
# adjust if the directory name is slightly different

# Now make a small Vitis project folder:
cd ~/projects  #this directory is one level above the main git repository directory
mkdir -p kf1d_vitis/{src,build}
cd kf1d_vitis/src

# Copy your latest dataflow HLS kernel here:
cp /home/pateljeet3/projects/fpga-kf-1d/hls/kf1d_float/kf1d_float.cpp .
cp /home/pateljeet3/projects/fpga-kf-1d/vitis_u280_host/host_kf1d.cpp .

# Building the kernel xclbin
cd ~/projects/kf1d_vitis
mkdir -p build

# Compile kernel to object (.xo)
v++ -c -t hw \
    --platform $PLATFORM \
    --kernel kf1d_float \
    -I./src \
    -o build/kf1d_float.hw.xo \
    src/kf1d_float.cpp

# Link into final xclbin
v++ -l -t hw \
    --platform $PLATFORM \
    --kernel kf1d_float \
    -o build/kf1d_float.hw.xclbin \
    build/kf1d_float.hw.xo

# Build the host executable 
cd ~/projects/kf1d_vitis

g++ -std=c++17 -O2 \
    src/host_kf1d.cpp \
    -I$XILINX_XRT/include \
    -L$XILINX_XRT/lib \
    -lOpenCL -lpthread -lrt \
    -o build/host_kf1d

# You should now have:
•	build/kf1d_float.hw.xclbin
•	build/host_kf1d

# Copy these files to the U280 node:
# From cd ~/projects/kf1d_vitis

scp build/kf1d_float.hw.xclbin \
    src/host_kf1d.cpp \
    patelj3@pc172.cloudlab.umass.edu:~/PU2 

# Now on the oct node:
source /opt/xilinx/xrt/setup.sh

unset XCL_EMULATION_MODE

g++ -std=c++17 -O2 \
    host_kf1d.cpp \
    -I$XILINX_XRT/include \
    -L$XILINX_XRT/lib \
    -lOpenCL -lpthread -lrt \
    -o host_kf1d

# HW RUN:
./host_kf1d kf1d_float.hw.xclbin
