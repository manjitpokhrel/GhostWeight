#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cuda_fp16.h>

// -----------------------------------------------------------------------
// GHOSTWEIGHT BLACKWELL KERNEL v1.0
// This kernel uses Warp Shuffles to skip zeroed activations in-silico.
// -----------------------------------------------------------------------

__global__ void ghost_warp_compressor_kernel(
    const half* __restrict__ input,      // Hidden state (activations)
    const half* __restrict__ weights,    // MLP Weights (dense for now)
    half* __restrict__ output,           // Output hidden state
    const float threshold,               // GhostGate Threshold (0.25)
    const int hidden_dim,
    const int intermediate_dim
) {
    // Each warp (32 threads) handles one row of the hidden state
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int tid = threadIdx.x; // Thread ID within the warp (0-31)

    if (row >= 1) return; // For now, handle single-token inference

    // 1. GhostGate: Check activation magnitude
    // Every thread in the warp loads one activation
    half val = input[tid];
    float abs_val = __half2float(__habs(val));

    // 2. Warp Vote: Who is alive?
    // This is the core 'Ghost' logic. Every thread votes.
    unsigned int active_mask = __ballot_sync(0xFFFFFFFF, abs_val > threshold);

    // 3. Early Exit: If the whole warp is dead, stop computing
    if (active_mask == 0) return;

    // 4. Sparse Computation
    // Instead of computing all 31 threads, we could skip.
    // For this benchmark, we only compute if the thread is in the active_mask
    if (active_mask & (1 << tid)) {
        float sum = 0.0f;
        for (int i = 0; i < hidden_dim; ++i) {
            sum += __half2float(input[i]) * __half2float(weights[tid * hidden_dim + i]);
        }
        output[tid] = __float2half(sum);
    } else {
        output[tid] = __float2half(0.0f); // Zero out dead weight
    }
}

// Host function to launch the kernel
extern "C" void launch_ghost_kernel(half* h_input, half* h_weights, half* h_output, float threshold, int h_dim, int i_dim) {
    half *d_input, *d_weights, *d_output;

    cudaMalloc(&d_input, h_dim * sizeof(half));
    cudaMalloc(&d_weights, h_dim * i_dim * sizeof(half));
    cudaMalloc(&d_output, i_dim * sizeof(half));

    cudaMemcpy(d_input, h_input, h_dim * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, h_dim * i_dim * sizeof(half), cudaMemcpyHostToDevice);

    // Launch with 1 block, 32 threads (one warp)
    dim3 block(32, 1);
    dim3 grid(1, 1);

    ghost_warp_compressor_kernel<<<grid, block>>>(d_input, d_weights, d_output, threshold, h_dim, i_dim);

    cudaMemcpy(h_output, d_output, i_dim * sizeof(half), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_output);
}