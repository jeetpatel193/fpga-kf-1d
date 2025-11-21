// hls/kf1d_float/kf1d_float.cpp

#include <hls_stream.h>

static void read_inputs(
    const float *u_gyro,
    const float *z_accel,
    hls::stream<float> &ug_s,
    hls::stream<float> &za_s,
    int N)
{
#pragma HLS INLINE off
ReadLoop:
    for (int k = 0; k < N; ++k) {
        #pragma HLS PIPELINE II=1
        ug_s.write(u_gyro[k]);
        za_s.write(z_accel[k]);
    }
}

static void kalman_core(
    hls::stream<float> &ug_s,
    hls::stream<float> &za_s,
    float dt, float Q, float R,
    float x0, float P0,
    int N,
    hls::stream<float> &x_s,
    hls::stream<float> &P_s)
{
    #pragma HLS INLINE off
    float x = x0;
    float P = P0;

KF_LOOP:
    for (int k = 0; k < N; ++k) {
        #pragma HLS PIPELINE
        //#pragma HLS unroll factor=8
        float ug = ug_s.read();
        float za = za_s.read();

        // Predict
        float ug_dt  = dt * ug;
        float x_pred = x + ug_dt;
        float P_pred = P + Q;

        // Update
        float innov  = za - x_pred;
        float S      = P_pred + R;
        float K      = P_pred / S;
        float K_innov    = K * innov;
        float one_minus_K = 1.0f - K;

        x = x_pred + K_innov;
        P = one_minus_K * P_pred;

        x_s.write(x);
        P_s.write(P);
    }
}

static void write_outputs(
    hls::stream<float> &x_s,
    hls::stream<float> &P_s,
    float *x_out,
    float *P_out,
    int   N)
{
#pragma HLS INLINE off
WriteLoop:
    for (int k = 0; k < N; ++k) {
#pragma HLS PIPELINE II=1
        x_out[k] = x_s.read();
        P_out[k] = P_s.read();
    }
}

extern "C" {
void kf1d_float(
    const float *u_gyro,   // [N] rad/s
    const float *z_accel,  // [N] rad
    float dt,
    float Q,
    float R,
    float x0,
    float P0,
    int   N,
    float *x_out,          // [N]
    float *P_out           // [N]
) {
    // ---- AXI master ports:
#pragma HLS INTERFACE m_axi     port=u_gyro  offset=slave bundle=gmem_ug depth=2048
#pragma HLS INTERFACE m_axi     port=z_accel offset=slave bundle=gmem_z depth=2048
#pragma HLS INTERFACE m_axi     port=x_out   offset=slave bundle=gmem_x depth=2048
#pragma HLS INTERFACE m_axi     port=P_out   offset=slave bundle=gmem_p depth=2048

// ---- AXI-Lite control ----
#pragma HLS INTERFACE s_axilite port=u_gyro  bundle=control
#pragma HLS INTERFACE s_axilite port=z_accel bundle=control
#pragma HLS INTERFACE s_axilite port=x_out   bundle=control
#pragma HLS INTERFACE s_axilite port=P_out   bundle=control
#pragma HLS INTERFACE s_axilite port=dt      bundle=control
#pragma HLS INTERFACE s_axilite port=Q       bundle=control
#pragma HLS INTERFACE s_axilite port=R       bundle=control
#pragma HLS INTERFACE s_axilite port=x0      bundle=control
#pragma HLS INTERFACE s_axilite port=P0      bundle=control
#pragma HLS INTERFACE s_axilite port=N       bundle=control
#pragma HLS INTERFACE s_axilite port=return  bundle=control

#pragma HLS DATAFLOW

    hls::stream<float> ug_s("ug_s");
    hls::stream<float> za_s("za_s");
    hls::stream<float> x_s("x_s");
    hls::stream<float> P_s("P_s");

#pragma HLS STREAM variable=ug_s depth=16
#pragma HLS STREAM variable=za_s depth=16
#pragma HLS STREAM variable=x_s  depth=16
#pragma HLS STREAM variable=P_s  depth=16

    read_inputs(u_gyro, z_accel, ug_s, za_s, N);
    kalman_core(ug_s, za_s, dt, Q, R, x0, P0, N, x_s, P_s);
    write_outputs(x_s, P_s, x_out, P_out, N);
    
}
}
