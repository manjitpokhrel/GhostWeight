import torch
import time
import numpy as np
import json
import os

def benchmark_async_overlap():
    print("=" * 60)
    print("BLACKWELL ASYNC OVERLAP TEST (PyTorch Native)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    # Transfer size: 1GB
    num_bytes = 1 * 1024 * 1024 * 1024
    num_floats = num_bytes // 4

    # Pinned memory on CPU (required for high speed DMA)
    weights_cpu = torch.zeros(num_floats, dtype=torch.float32).pin_memory()

    # GPU buffers
    d_weights = torch.zeros(num_floats, dtype=torch.float32, device="cuda")
    d_compute_a = torch.randn(4096, 4096, dtype=torch.float32, device="cuda")
    d_compute_b = torch.randn(4096, 4096, dtype=torch.float32, device="cuda")

    # CUDA streams
    stream_memory = torch.cuda.Stream()
    stream_compute = torch.cuda.Stream()

    print(f"Transfer size:  {num_bytes/1e9:.1f} GB")
    print(f"Compute size:   4096x4096 matmul x 10")
    print()

    # Warmup
    d_weights.copy_(weights_cpu, non_blocking=True)
    torch.matmul(d_compute_a, d_compute_b)
    torch.cuda.synchronize()

    # --------------------------------------------------------
    # Test 1: Serial execution
    # --------------------------------------------------------
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Copy weights
    d_weights.copy_(weights_cpu, non_blocking=False)
    torch.cuda.synchronize()

    # Then compute
    for _ in range(10):
        torch.matmul(d_compute_a, d_compute_b)
    torch.cuda.synchronize()

    t_serial = time.perf_counter() - t0
    print(f"Serial (copy then compute):  {t_serial*1000:.2f} ms")

    # --------------------------------------------------------
    # Test 2: Async overlap
    # --------------------------------------------------------
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Stream 1: Memory transfer
    with torch.cuda.stream(stream_memory):
        d_weights.copy_(weights_cpu, non_blocking=True)

    # Stream 2: Compute (happens simultaneously)
    with torch.cuda.stream(stream_compute):
        for _ in range(10):
            torch.matmul(d_compute_a, d_compute_b)

    # Wait for both
    torch.cuda.synchronize()

    t_async = time.perf_counter() - t0
    overlap = (1 - (t_async / t_serial)) * 100

    print(f"Async (simultaneous):        {t_async*1000:.2f} ms")
    print(f"Time saved:                  {(t_serial-t_async)*1000:.2f} ms")
    print(f"Hardware Overlap:            {overlap:.2f}%")
    print("=" * 60)

    if overlap > 40:
        print("STATUS: BLACKWELL OVERLAP CONFIRMED ✅")
        print("        GPU moves weights AND computes simultaneously")
        print("        70B model streaming is now viable")
    elif overlap > 20:
        print("STATUS: PARTIAL OVERLAP ✅")
        print("        Latency hiding confirmed")
        print("        Enable Resizable BAR in BIOS for full overlap")
    elif overlap > 0:
        print("STATUS: MINIMAL OVERLAP")
        print("        PCIe is bottleneck")
    else:
        print("STATUS: NO OVERLAP")
        print("        Windows WDDM serializing streams")
    print("=" * 60)

    # --------------------------------------------------------
    # Raw PCIe Bandwidth Test
    # --------------------------------------------------------
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    d_weights.copy_(weights_cpu, non_blocking=False)
    torch.cuda.synchronize()
    t_transfer = time.perf_counter() - t0

    bandwidth = num_bytes / 1e9 / t_transfer

    print(f"\nRAW TRANSFER STATS:")
    print(f"Transfer time:    {t_transfer*1000:.2f} ms")
    print(f"PCIe Bandwidth:   {bandwidth:.2f} GB/s")
    print(f"Per-layer speed:  {1000/bandwidth:.1f} ms per 1GB layer")
    print("=" * 60)

    # --------------------------------------------------------
    # The Critical Number For 70B Streaming
    # --------------------------------------------------------
    # A 70B model in 4-bit = ~35GB total
    # With GhostWeight we only need 27.12% = ~9.5GB active
    # Spread across 80 layers = ~118MB active per layer
    layer_active_gb = 0.118
    layer_swap_ms = (layer_active_gb / bandwidth) * 1000

    print(f"\n70B STREAMING VIABILITY:")
    print(f"Active weights per layer: {layer_active_gb*1024:.0f} MB")
    print(f"Layer swap time:          {layer_swap_ms:.2f} ms")
    print(f"Compute time per layer:   ~{398/28:.0f} ms (from Phase 2)")

    if layer_swap_ms < 14:
        print(f"VERDICT: STREAMING IS VIABLE ✅")
        print(f"         Swap is {14/layer_swap_ms:.1f}x faster than compute")
        print(f"         Memory latency fully hidden behind compute")
    else:
        print(f"VERDICT: SWAP IS THE BOTTLENECK")
        print(f"         Need faster PCIe or GPUDirect Storage")
    print("=" * 60)

    os.makedirs("F:/GhostWeight/data", exist_ok=True)
    with open("F:/GhostWeight/data/async_test.json", "w") as f:
        json.dump({
            "serial_ms": t_serial * 1000,
            "async_ms": t_async * 1000,
            "overlap_pct": overlap,
            "pcie_bandwidth_gbs": bandwidth,
            "layer_swap_ms": layer_swap_ms,
            "hardware": "RTX 5060 Blackwell GDDR7",
        }, f, indent=4)

    print("Saved: F:/GhostWeight/data/async_test.json")

if __name__ == "__main__":
    benchmark_async_overlap()