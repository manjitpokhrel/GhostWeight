import torch
import cupy as cp
import time
import numpy as np

# ============================================================
# GHOSTWEIGHT HARDWARE BENCHMARK v2
# Technique: Sparse Row Packing
# Instead of branching inside the kernel (causes warp divergence),
# we pack active rows into a dense buffer first.
# Then we run a dense matmul on only the active rows.
# This is the "correct" way to exploit sparsity on a GPU.
# ============================================================

# Kernel 1: The Row Packer
# Takes a sparse input and packs active elements into a dense buffer
row_packer_code = r'''
extern "C" __global__
void pack_active_rows(
    const float* input,
    float* packed_output,
    int* active_indices,
    int* active_count,
    float threshold,
    int dim
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= dim) return;

    float val = input[tid];
    if (abs(val) > threshold) {
        // Atomically claim a slot in the packed buffer
        int slot = atomicAdd(active_count, 1);
        packed_output[slot] = val;
        active_indices[slot] = tid;
    }
}
'''

# Kernel 2: The Sparse Matmul
# Runs dense matmul ONLY on the packed active rows
sparse_matmul_code = r'''
extern "C" __global__
void sparse_matmul_packed(
    const float* packed_input,
    const float* weights,
    float* output,
    const int* active_indices,
    int active_count,
    int hidden_dim,
    int intermediate_dim
) {
    int out_row = blockIdx.x;
    int tid = threadIdx.x;

    if (out_row >= intermediate_dim) return;

    float sum = 0.0f;

    // Only iterate over ACTIVE elements
    for (int i = tid; i < active_count; i += blockDim.x) {
        int original_idx = active_indices[i];
        sum += packed_input[i] * weights[out_row * hidden_dim + original_idx];
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    if (tid == 0) {
        output[out_row] = sum;
    }
}
'''

# Kernel 3: Dense Baseline (cuBLAS equivalent)
dense_matmul_code = r'''
extern "C" __global__
void dense_matmul(
    const float* input,
    const float* weights,
    float* output,
    int hidden_dim,
    int intermediate_dim
) {
    int out_row = blockIdx.x;
    int tid = threadIdx.x;

    if (out_row >= intermediate_dim) return;

    float sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        sum += input[i] * weights[out_row * hidden_dim + i];
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    if (tid == 0) {
        output[out_row] = sum;
    }
}
'''

# Compile all kernels
packer_module = cp.RawModule(code=row_packer_code)
sparse_module = cp.RawModule(code=sparse_matmul_code)
dense_module = cp.RawModule(code=dense_matmul_code)

pack_kernel = packer_module.get_function('pack_active_rows')
sparse_kernel = sparse_module.get_function('sparse_matmul_packed')
dense_kernel = dense_module.get_function('dense_matmul')

def benchmark():
    HIDDEN_DIM = 3584
    INTERMEDIATE_DIM = 18944
    THRESHOLD = 0.25
    SPARSITY = 0.7288
    RUNS = 200

    print("=" * 60)
    print("GHOSTWEIGHT v2: SPARSE ROW PACKING BENCHMARK")
    print(f"Hardware: RTX 5060 Blackwell | CUDA 12.6")
    print(f"Sparsity: {SPARSITY*100:.2f}% | Hidden: {HIDDEN_DIM} | Inter: {INTERMEDIATE_DIM}")
    print("=" * 60)

    # Create sparse input (72.88% zeros)
    input_np = np.random.randn(HIDDEN_DIM).astype(np.float32)
    mask_np = (np.random.rand(HIDDEN_DIM) > SPARSITY).astype(np.float32)
    sparse_input_np = input_np * mask_np

    weights_np = np.random.randn(INTERMEDIATE_DIM, HIDDEN_DIM).astype(np.float32)

    # GPU allocations
    d_input = cp.asarray(sparse_input_np)
    d_weights = cp.asarray(weights_np)
    d_output_dense = cp.zeros(INTERMEDIATE_DIM, dtype=cp.float32)
    d_output_sparse = cp.zeros(INTERMEDIATE_DIM, dtype=cp.float32)

    # Packing buffers
    d_packed = cp.zeros(HIDDEN_DIM, dtype=cp.float32)
    d_indices = cp.zeros(HIDDEN_DIM, dtype=cp.int32)
    d_count = cp.zeros(1, dtype=cp.int32)

    # --------------------------------------------------------
    # BENCHMARK 1: Dense Baseline
    # --------------------------------------------------------
    print("\n[1/3] Dense Baseline (Standard matmul)...")

    # Warmup
    for _ in range(10):
        dense_kernel(
            (INTERMEDIATE_DIM,), (32,),
            (d_input, d_weights, d_output_dense,
             HIDDEN_DIM, INTERMEDIATE_DIM)
        )
    cp.cuda.Stream.null.synchronize()

    t0 = time.perf_counter()
    for _ in range(RUNS):
        dense_kernel(
            (INTERMEDIATE_DIM,), (32,),
            (d_input, d_weights, d_output_dense,
             HIDDEN_DIM, INTERMEDIATE_DIM)
        )
    cp.cuda.Stream.null.synchronize()
    t_dense = (time.perf_counter() - t0) / RUNS

    print(f"Dense time: {t_dense*1000:.4f} ms")

    # --------------------------------------------------------
    # BENCHMARK 2: GhostWeight v1 (Branch Skipping)
    # --------------------------------------------------------
    print("\n[2/3] GhostWeight v1 (Branch skipping - our Phase 3A)...")

    ghost_v1_code = r'''
    extern "C" __global__
    void ghost_v1(const float* input, const float* weights,
                  float* output, float threshold,
                  int hidden_dim, int intermediate_dim) {
        int row = blockIdx.x;
        int tid = threadIdx.x;
        float sum = 0.0f;
        for (int i = tid; i < hidden_dim; i += blockDim.x) {
            float val = input[i];
            if (abs(val) > threshold) {
                sum += val * weights[row * hidden_dim + i];
            }
        }
        for (int offset = 16; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        if (tid == 0) output[row] = sum;
    }
    '''
    ghost_v1_module = cp.RawModule(code=ghost_v1_code)
    ghost_v1_kernel = ghost_v1_module.get_function('ghost_v1')

    for _ in range(10):
        ghost_v1_kernel(
            (INTERMEDIATE_DIM,), (32,),
            (d_input, d_weights, d_output_sparse,
             cp.float32(THRESHOLD), HIDDEN_DIM, INTERMEDIATE_DIM)
        )
    cp.cuda.Stream.null.synchronize()

    t0 = time.perf_counter()
    for _ in range(RUNS):
        ghost_v1_kernel(
            (INTERMEDIATE_DIM,), (32,),
            (d_input, d_weights, d_output_sparse,
             cp.float32(THRESHOLD), HIDDEN_DIM, INTERMEDIATE_DIM)
        )
    cp.cuda.Stream.null.synchronize()
    t_ghost_v1 = (time.perf_counter() - t0) / RUNS

    speedup_v1 = (t_dense / t_ghost_v1 - 1) * 100
    print(f"GhostWeight v1 time: {t_ghost_v1*1000:.4f} ms | Speedup: {speedup_v1:.2f}%")

    # --------------------------------------------------------
    # BENCHMARK 3: GhostWeight v2 (Row Packing)
    # --------------------------------------------------------
    print("\n[3/3] GhostWeight v2 (Sparse Row Packing)...")

    # Pre-pack active rows (this happens during predictor phase)
    d_count.fill(0)
    threads = 256
    blocks = (HIDDEN_DIM + threads - 1) // threads
    pack_kernel(
        (blocks,), (threads,),
        (d_input, d_packed, d_indices, d_count,
         cp.float32(THRESHOLD), HIDDEN_DIM)
    )
    cp.cuda.Stream.null.synchronize()

    active_count = int(d_count[0])
    active_ratio = active_count / HIDDEN_DIM

    print(f"Active elements: {active_count}/{HIDDEN_DIM} ({active_ratio*100:.2f}%)")

    for _ in range(10):
        d_count.fill(0)
        pack_kernel(
            (blocks,), (threads,),
            (d_input, d_packed, d_indices, d_count,
             cp.float32(THRESHOLD), HIDDEN_DIM)
        )
        sparse_kernel(
            (INTERMEDIATE_DIM,), (32,),
            (d_packed, d_weights, d_output_sparse,
             d_indices, active_count,
             HIDDEN_DIM, INTERMEDIATE_DIM)
        )
    cp.cuda.Stream.null.synchronize()

    t0 = time.perf_counter()
    for _ in range(RUNS):
        d_count.fill(0)
        pack_kernel(
            (blocks,), (threads,),
            (d_input, d_packed, d_indices, d_count,
             cp.float32(THRESHOLD), HIDDEN_DIM)
        )
        sparse_kernel(
            (INTERMEDIATE_DIM,), (32,),
            (d_packed, d_weights, d_output_sparse,
             d_indices, active_count,
             HIDDEN_DIM, INTERMEDIATE_DIM)
        )
    cp.cuda.Stream.null.synchronize()
    t_ghost_v2 = (time.perf_counter() - t0) / RUNS

    speedup_v2 = (t_dense / t_ghost_v2 - 1) * 100

    # --------------------------------------------------------
    # Final Results
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("GHOSTWEIGHT HARDWARE BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Hardware:              RTX 5060 Blackwell GDDR7")
    print(f"Sparsity:              {SPARSITY*100:.2f}%")
    print(f"Active elements:       {active_ratio*100:.2f}%")
    print("=" * 60)
    print(f"Dense Baseline:        {t_dense*1000:.4f} ms")
    print(f"GhostWeight v1:        {t_ghost_v1*1000:.4f} ms  ({speedup_v1:+.2f}%)")
    print(f"GhostWeight v2:        {t_ghost_v2*1000:.4f} ms  ({speedup_v2:+.2f}%)")
    print("=" * 60)

    if speedup_v2 > speedup_v1:
        print(f"Row Packing improvement over v1: +{speedup_v2-speedup_v1:.2f}%")

    if speedup_v2 > 50:
        print("\nSTATUS: EXTRAORDINARY ✅")
        print("        >50% hardware speedup on Blackwell")
        print("        This result is NVIDIA-worthy")
    elif speedup_v2 > 28:
        print("\nSTATUS: STRONG ✅")
        print("        Exceeds our Phase 3A baseline (28.30%)")
        print("        Ready for paper submission")
    elif speedup_v2 > 0:
        print("\nSTATUS: POSITIVE ✅")
        print("        Confirmed hardware-level speedup")
    else:
        print("\nSTATUS: Memory-bound — need async prefetch")
        print("        Moving to Phase 3B: Async Streams")
    print("=" * 60)

    import json
    import os
    os.makedirs("F:/GhostWeight/data", exist_ok=True)
    with open("F:/GhostWeight/data/phase3_results.json", "w") as f:
        json.dump({
            "dense_ms": t_dense * 1000,
            "ghost_v1_ms": t_ghost_v1 * 1000,
            "ghost_v2_ms": t_ghost_v2 * 1000,
            "speedup_v1_pct": speedup_v1,
            "speedup_v2_pct": speedup_v2,
            "sparsity_pct": SPARSITY * 100,
            "active_ratio_pct": active_ratio * 100,
            "hardware": "RTX 5060 Blackwell GDDR7",
            "cuda": "12.6"
        }, f, indent=4)

    print("Results saved: F:/GhostWeight/data/phase3_results.json")

if __name__ == "__main__":
    benchmark()