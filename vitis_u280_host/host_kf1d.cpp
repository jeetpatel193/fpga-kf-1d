// host_kf1d.cpp
// Simple host to run kf1d_float on U280.
//
// Usage: ./host_kf1d <xclbin_path>
// If xclbin_path is omitted, defaults to "build/kf1d_float.hw.xclbin".

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

// ---------------------------------------------------------------------
// Helper: read whole file into a byte vector
// ---------------------------------------------------------------------
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

// ---------------------------------------------------------------------
// Helper: find the Xilinx accelerator device
// ---------------------------------------------------------------------
static cl::Device get_xilinx_device() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        throw std::runtime_error("No OpenCL platforms found");
    }

    for (const auto &p : platforms) {
        std::string pname = p.getInfo<CL_PLATFORM_NAME>();
        // Xilinx platforms usually contain "Xilinx" in their name
        if (pname.find("Xilinx") == std::string::npos)
            continue;

        std::vector<cl::Device> devices;
        p.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
        if (!devices.empty()) {
            // Just pick the first accelerator on the Xilinx platform
            return devices[0];
        }
    }
    throw std::runtime_error("No Xilinx accelerator device found");
}

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------
int main(int argc, char **argv) {
    try {
        // -------------------------------------------------------------
        // 1. Parse arguments and load xclbin
        // -------------------------------------------------------------
        std::string xclbin_path =
            (argc > 1) ? argv[1] : "build/kf1d_float.hw.xclbin";

        std::cout << "Using xclbin: " << xclbin_path << "\n";

        auto binary = load_binary_file(xclbin_path);

        // -------------------------------------------------------------
        // 2. Create OpenCL context, command queue, program, and kernel
        // -------------------------------------------------------------
        cl::Device device = get_xilinx_device();
        cl::Context context(device);
        cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);

        std::cout << "Device: "
                  << device.getInfo<CL_DEVICE_NAME>() << "\n";

        // cl::Program::Binaries is std::vector<std::vector<unsigned char>>
        cl::Program::Binaries bins;
        bins.push_back(binary);  // one binary for one device

        std::vector<cl::Device> devices{device};
        cl::Program program(context, devices, bins);
        cl::Kernel kernel(program, "kf1d_float");

        // -------------------------------------------------------------
        // 3. Create input data (SYNTHETIC for now)
        //    Later I will  replace this with CSV loading from fpga-kf-1d/data
        // -------------------------------------------------------------
        const int N = 2048;  // can match your dataset length later

        std::vector<float> u_gyro(N);
        std::vector<float> z_accel(N);
        std::vector<float> x_out(N, 0.0f);
        std::vector<float> P_out(N, 0.0f);

        // Simple synthetic signal: constant rotation + noisy accel angle
        float dt = 0.01f;
        float true_rate = 0.5f;  // rad/s
        float angle = 0.0f;
        for (int k = 0; k < N; ++k) {
            u_gyro[k] = true_rate;      // "measured" gyro
            angle += true_rate * dt;    // true integrated angle
            float noise = 0.05f * std::sin(0.1f * k);
            z_accel[k] = angle + noise; // "measured" accel-based angle
        }

        // Kalman parameters - same as in your HLS test
        float Q  = 0.01f;
        float R  = 0.1f;
        float x0 = 0.0f;
        float P0 = 1.0f;

        // -------------------------------------------------------------
        // 4. Create device buffers and set kernel arguments
        // -------------------------------------------------------------
        cl::Buffer buf_u(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(float) * N,
            u_gyro.data());

        cl::Buffer buf_z(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(float) * N,
            z_accel.data());

        cl::Buffer buf_x(
            context,
            CL_MEM_WRITE_ONLY,
            sizeof(float) * N);

        cl::Buffer buf_P(
            context,
            CL_MEM_WRITE_ONLY,
            sizeof(float) * N);

        // Arg order must match kf1d_float signature!
        // void kf1d_float(const float *u_gyro,
        //                 const float *z_accel,
        //                 float dt, float Q, float R,
        //                 float x0, float P0,
        //                 int N,
        //                 float *x_out,
        //                 float *P_out)
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
        // 5. Migrate input buffers to device, run kernel, read back
        // -------------------------------------------------------------
        q.enqueueMigrateMemObjects({buf_u, buf_z}, 0 /*host->device*/);

        auto t_start = std::chrono::high_resolution_clock::now();
        q.enqueueTask(kernel);
        q.finish();
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms =
            std::chrono::duration<double, std::milli>(t_end - t_start).count();

        q.enqueueMigrateMemObjects(
            {buf_x, buf_P},
            CL_MIGRATE_MEM_OBJECT_HOST);
        q.finish();

        std::cout << "Kernel finished. Wall-clock time (host): "
                  << elapsed_ms << " ms\n";

        // -------------------------------------------------------------
        // 6. Print a few results
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

    } catch (const cl::Error &e) {
        std::cerr << "OpenCL error: " << e.what()
                  << " (" << e.err() << ")\n";
        return 1;
    } catch (const std::exception &e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
}

