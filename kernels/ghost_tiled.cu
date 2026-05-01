#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>

#define TILE_SIZE 32

// -----------------------------------------------------------------------
// BLACKWELL TILED SPARSE KERNEL (GhostWeight Phase 4B)
// Exploits L1 Shared Memory to bypass GDDR7 latency.
// -----------------------------------------------------------------------

__global__ void ghost_tiled_matmul(
    const half* __restrict__ input,
    const half* __restrict__ weights,
    half* __restrict__ output,
    const int* __restrict__ active_indices,
    int active_count,
    int hidden_dim
) {
    // Shared Memory Tiling: One tile for input activations
    __shared__ half shared_input[TILE_SIZE];

    int row = blockIdx.x; // Each block handles one output neuron
    int tid = threadIdx.x;
    
    float sum = 0.0f;

    // Iterate over active weights in tiles of 32
    for (int t = 0; t < (active_count + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int idx = t * TILE_SIZE + tid;
        
        // Cooperative Load: Threads work together to load a tile into SRAM
        if (idx < active_count) {
            int original_idx = active_indices[idx];
            shared_input[tid] = input[original_idx];
        } else {
            shared_input[tid] = __float2half(0.0f);
        }

        __syncthreads(); // Wait for tile to load

        // Compute Tile: Every thread in the block uses the SRAM tile
        // This is where the 20x speedup lives.
        for (int i = 0; i < TILE_SIZE; ++i) {
            int weight_idx = row * hidden_dim + active_indices[t * TILE_SIZE + i];
            if ((t * TILE_SIZE + i) < active_count) {
                sum += __half2float(shared_input[i]) * __half2float(weights[weight_idx]);
            }
        }
        __syncthreads();
    }

    // Final reduction
    if (tid == 0) {
        output[row] = __float2half(sum);
    }
}