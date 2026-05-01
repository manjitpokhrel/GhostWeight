#ifndef GHOST_ENGINE_H
#define GHOST_ENGINE_H

#include <cuda_runtime.h>
#include <vector>
#include <string>

// The Weight Shard represents one layer on your SSD
struct GhostShard {
    std::string layer_name;
    void* host_ptr;          // Pointer to weights in System RAM (mmap)
    size_t size;
    int* active_indices;     // Indices predicted by GhostPredictor
    int active_count;
};

class GhostRuntime {
public:
    GhostRuntime(size_t vram_budget_mb);
    ~GhostRuntime();

    // The core loop: Load -> Predict -> Compute -> Swap
    void run_inference(const std::vector<int>& input_tokens);

private:
    cudaStream_t compute_stream;
    cudaStream_t memory_stream;
    
    // Double buffers for weights
    float *d_weight_A, *d_weight_B;
    float *d_hidden_state;
    
    void load_shard_async(GhostShard& shard, float* target_buffer);
};

#endif