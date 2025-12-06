// host_kf1d.cpp
// Host to run kf1d_float on U280.
//
// Usage:
//   Synthetic data:
//     ./host_kf1d kf1d_float.hw.xclbin
//
//   CSV data (single-column CSV with header in row 1):
//     ./host_kf1d kf1d_float.hw.xclbin sine_u_gyro.csv sine_z_accel.csv

#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS

#include <CL/opencl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Simple CPU reference implementation of the same 1D KF
void kf1d_cpu(const std::vector<float> &u_gyro,
              const std::vector<float> &z_accel,
              float dt, float Q, float R,
              float x0, float P0,
              std::vector<float> &x_cpu,
              std::vector<float> &P_cpu) {
    const int N = std::min(u_gyro.size(), z_accel.size());
    x_cpu.resize(N);
    P_cpu.resize(N);

    float x = x0;
    float P = P0;

    for (int k = 0; k < N; ++k) {
        float ug = u_gyro[k];
        float za = z_accel[k];

        // Predict
        float x_pred = x + dt * ug;
        float P_pred = P + Q;

        // Update
        float innov  = za - x_pred;
        float S      = P_pred + R;
        float K      = P_pred / S;

        x = x_pred + K * innov;
        P = (1.0f - K) * P_pred;

        x_cpu[k] = x;
        P_cpu[k] = P;
    }
}


// -------------------------- File helpers ------------------------------

static std::vector<unsigned char> load_binary_file(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        throw std::runtime_error("Failed to open xclbin file: " + path);
    }
    std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);

    std::vector<unsigned char> buffer(size);
    if (!f.read(reinterpret_cast<char *>(buffer.data()), size)) {
        throw std::runtime_error("Failed to read xclbin file: " + path);
    }
    return buffer;
}

// Read a single-column CSV with a header in the first line
static std::vector<float> read_csv_column(const std::string &path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("Failed to open CSV file: " + path);
    }

    std::vector<float> vals;
    std::string line;
    bool first = true;

    while (std::getline(f, line)) {
        // Trim whitespace
        auto start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue; // empty line
        auto end = line.find_last_not_of(" \t\r\n");
        std::string token = line.substr(start, end - start + 1);

        if (first) {
            // Treat first non-empty line as header ("u_gyro", "z_accel", etc.)
            first = false;
            continue;
        }

        vals.push_back(std::stof(token));
    }

    if (vals.empty()) {
        throw std::runtime_error("No numeric data found in CSV: " + path);
    }
    return vals;
}

// ------------------------ Device selection ----------------------------

static cl::Device get_xilinx_device() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        throw std::runtime_error("No OpenCL platforms found");
    }

    for (const auto &p : platforms) {
        std::string pname = p.getInfo<CL_PLATFORM_NAME>();
        if (pname.find("Xilinx") == std::string::npos)
            continue;

        std::vector<cl::Device> devices;
        p.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
        if (!devices.empty()) {
            return devices[0]; // first Xilinx accelerator
        }
    }
    throw std::runtime_error("No Xilinx accelerator device found");
}

// ------------------------------ main ----------------------------------

int main(int argc, char **argv) {
    try {
        // -------------------------------------------------------------
        // 1. Parse arguments
        // -------------------------------------------------------------
        std::string xclbin_path = (argc >= 2)
                                  ? argv[1]
                                  : "kf1d_float.hw.xclbin";

        bool use_csv = (argc >= 4);
        std::string csv_u_path, csv_z_path;

        std::cout << "Using xclbin: " << xclbin_path << "\n";

        if (use_csv) {
            csv_u_path = argv[2];
            csv_z_path = argv[3];
            std::cout << "Using CSV inputs:\n"
                      << "  u_gyro: " << csv_u_path << "\n"
                      << "  z_accel: " << csv_z_path << "\n";
        } else {
            std::cout << "Using synthetic input data.\n";
        }

        // -------------------------------------------------------------
        // 2. Load xclbin and set up OpenCL (device, context, queue, program)
        // -------------------------------------------------------------
        auto binary = load_binary_file(xclbin_path);

        cl::Device device = get_xilinx_device();
        cl::Context context(device);
        cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);

        std::cout << "Device: " << device.getInfo<CL_DEVICE_NAME>() << "\n";

        cl::Program::Binaries bins;
        bins.push_back(binary);  // one binary for this device
        std::vector<cl::Device> devices{device};

        cl::Program program(context, devices, bins);
        cl::Kernel kernel(program, "kf1d_float");

        // -------------------------------------------------------------
        // 3. Build input vectors (CSV or synthetic)
        // -------------------------------------------------------------
        std::vector<float> u_gyro;
        std::vector<float> z_accel;

        if (use_csv) {
            u_gyro  = read_csv_column(csv_u_path);
            z_accel = read_csv_column(csv_z_path);
            if (u_gyro.size() != z_accel.size()) {
                throw std::runtime_error("CSV sizes differ: "
                                        + std::to_string(u_gyro.size())
                                        + " vs "
                                        + std::to_string(z_accel.size()));
            }
        } else {
            // Synthetic case: allow optional N from command line.
            // Usage:
            //   ./host_kf1d kf1d_float.hw.xclbin           -> N = 2048 (default)
            //   ./host_kf1d kf1d_float.hw.xclbin 10000    -> N = 10000 synthetic
            int N_synth = 2048;
            if (argc >= 3) {
                N_synth = std::stoi(argv[2]);   // third arg is N when NOT using CSV
            }
            u_gyro.resize(N_synth);
            z_accel.resize(N_synth);

            float dt = 0.01f;
            float true_rate = 0.5f;  // rad/s
            float angle = 0.0f;
            for (int k = 0; k < N_synth; ++k) {
                u_gyro[k] = true_rate;
                angle += true_rate * dt;
                float noise = 0.05f * std::sin(0.1f * k);
                z_accel[k] = angle + noise;
            }
        }

        const int N = static_cast<int>(u_gyro.size());

        std::vector<float> x_out(N, 0.0f);
        std::vector<float> P_out(N, 0.0f);

        // Kalman parameters – aligned with Python/HLS golden:
        // Q = 1e-4 (process), R = 1e-2 (measurement),
        // x0 = first accel measurement, P0 = 1.0
        float dt = 0.01f;        // matches gen_imu.py default
        float Q  = 1e-4f;
        float R  = 1e-2f;
        float P0 = 1.0f;

        // if we have at least one sample, start at first accel measurement
        float x0 = z_accel.empty() ? 0.0f : z_accel[0];



        // --- CPU golden run for comparison ---
        std::vector<float> x_cpu, P_cpu;

        auto cpu_t0 = std::chrono::high_resolution_clock::now();
        kf1d_cpu(u_gyro, z_accel, dt, Q, R, x0, P0, x_cpu, P_cpu);
        auto cpu_t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms =
            std::chrono::duration<double, std::milli>(cpu_t1 - cpu_t0).count();

        std::cout << "CPU reference time: " << cpu_ms << " ms\n";


        // -------------------------------------------------------------
        // 4. Create device buffers
        // -------------------------------------------------------------
        size_t bytes = sizeof(float) * N;

        cl::Buffer buf_u(context, CL_MEM_READ_ONLY,  bytes);
        cl::Buffer buf_z(context, CL_MEM_READ_ONLY,  bytes);
        cl::Buffer buf_x(context, CL_MEM_WRITE_ONLY, bytes);
        cl::Buffer buf_P(context, CL_MEM_WRITE_ONLY, bytes);

        // -------------------------------------------------------------
        // 5. Copy input data to device
        // -------------------------------------------------------------
        q.enqueueWriteBuffer(buf_u, CL_TRUE, 0, bytes, u_gyro.data());
        q.enqueueWriteBuffer(buf_z, CL_TRUE, 0, bytes, z_accel.data());

        // -------------------------------------------------------------
        // 6. Set kernel args
        // -------------------------------------------------------------
        kernel.setArg(0, buf_u);
        kernel.setArg(1, buf_z);
        kernel.setArg(2, dt);
        kernel.setArg(3, Q);
        kernel.setArg(4, R);
        kernel.setArg(5, x0);
        kernel.setArg(6, P0);
        kernel.setArg(7, N);
        kernel.setArg(8, buf_x);
        kernel.setArg(9, buf_P);

        // -------------------------------------------------------------
        // 7. Run kernel (measure both host wall time and device kernel time)
        // -------------------------------------------------------------
        auto t_start = std::chrono::high_resolution_clock::now();

        cl::Event ev;
        q.enqueueTask(kernel, nullptr, &ev);  // attach event to this launch
        q.finish();                           // wait for kernel to complete

        auto t_end = std::chrono::high_resolution_clock::now();

        // Host wall-clock time
        double elapsed_ms =
            std::chrono::duration<double, std::milli>(t_end - t_start).count();

        // Pure device kernel time from profiling info (ns -> ms)
        cl_ulong t_start_ns = ev.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong t_end_ns   = ev.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        double kernel_ms    = (t_end_ns - t_start_ns) * 1e-6;

        // -------------------------------------------------------------
        // 8. Read results back to host
        // -------------------------------------------------------------
        q.enqueueReadBuffer(buf_x, CL_TRUE, 0, bytes, x_out.data());
        q.enqueueReadBuffer(buf_P, CL_TRUE, 0, bytes, P_out.data());

        std::cout << "Kernel finished. Wall-clock time (host): "
                << elapsed_ms << " ms\n";
        std::cout << "FPGA kernel time (device profiling): "
                << kernel_ms << " ms\n";



        // --- Compare FPGA vs CPU ---
        double max_err_x = 0.0;
        double max_err_P = 0.0;
        int N_eff = std::min<int>(x_out.size(), x_cpu.size());

        for (int k = 0; k < N_eff; ++k) {
            double dx = std::abs(x_out[k] - x_cpu[k]);
            double dP = std::abs(P_out[k] - P_cpu[k]);
            if (dx > max_err_x) max_err_x = dx;
            if (dP > max_err_P) max_err_P = dP;
        }

        std::cout << "FPGA kernel wall time (host): " << elapsed_ms << " ms\n";
        std::cout << "Speedup (CPU / FPGA): " << (cpu_ms / elapsed_ms) << "x\n";
        std::cout << "Max |x_fpga - x_cpu| = " << max_err_x << "\n";
        std::cout << "Max |P_fpga - P_cpu| = " << max_err_P << "\n";


        // -------------------------------------------------------------
        // 9. Print a few samples
        // -------------------------------------------------------------
        std::cout << "First 10 samples (k, u_gyro, z_accel, x_hat, P):\n";
        for (int k = 0; k < std::min(N, 10); ++k) {
            std::cout << k
                      << "  " << u_gyro[k]
                      << "  " << z_accel[k]
                      << "  " << x_out[k]
                      << "  " << P_out[k] << "\n";
        }

        std::cout << "Done.\n";
        return 0;
    }
    catch (const cl::Error &e) {
        std::cerr << "OpenCL error: " << e.what()
                  << " (" << e.err() << ")\n";
        return 1;
    }
    catch (const std::exception &e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
}
