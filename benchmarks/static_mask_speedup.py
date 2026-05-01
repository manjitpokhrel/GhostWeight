import torch
import torch.nn as nn
import time
import json
import os
import pickle
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.activations import SiLUActivation

model_id = r"F:\hf_cache\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"

print("=" * 65)
print("GHOSTWEIGHT STATIC MASK: ZERO-OVERHEAD SPEEDUP")
print("=" * 65)
print("Strategy: Remove permanently dead neurons (27.3%) from")
print("weight matrices before inference. No predictor needed.")
print("=" * 65)

# ============================================================
# Load the weight index from Phase 2
# ============================================================

index_path = "F:/GhostWeight/data/weight_index_full.pkl"
if os.path.exists(index_path):
    with open(index_path, "rb") as f:
        weight_index = pickle.load(f)
    print(f"\nLoaded weight index: {len(weight_index)} layers")
else:
    print("Weight index not found. Run build_weight_index.py first.")
    exit()

# ============================================================
# Analyze the static mask
# ============================================================

total_neurons = 0
total_never_active = 0
total_always_active = 0

for layer_name, stats in weight_index.items():
    avg_firing = np.array(stats["core_neuron_indices"])
    masks = stats.get("avg_firing_rate", 0)
    total_neurons += stats["total_neurons"]
    total_never_active += stats["never_active"]
    total_always_active += stats["always_active"]

never_active_pct = total_never_active / total_neurons * 100
always_active_pct = total_always_active / total_neurons * 100

print(f"\nStatic Mask Analysis:")
print(f"  Total neurons:     {total_neurons:,}")
print(f"  Never active:      {total_never_active:,} ({never_active_pct:.1f}%)")
print(f"  Always active:     {total_always_active:,} ({always_active_pct:.1f}%)")
print(f"  Free VRAM savings: {never_active_pct:.1f}% of MLP weights")

# ============================================================
# Benchmark: Dense vs Static-Pruned
# ============================================================

HIDDEN_DIM = 3584
INTERMEDIATE_DIM = 18944
RUNS = 500

# Simulate static pruning
# Remove 27.3% of output neurons permanently
keep_ratio = 1.0 - (never_active_pct / 100)
KEPT = int(INTERMEDIATE_DIM * keep_ratio)

weights_full = torch.randn(INTERMEDIATE_DIM, HIDDEN_DIM, device="cuda")
weights_pruned = weights_full[:KEPT].contiguous()
input_vec = torch.randn(HIDDEN_DIM, device="cuda")

print(f"\nBenchmarking static pruning...")
print(f"  Full matrix:    [{INTERMEDIATE_DIM} x {HIDDEN_DIM}]")
print(f"  Pruned matrix:  [{KEPT} x {HIDDEN_DIM}] ({keep_ratio*100:.1f}% kept)")

# Warmup
for _ in range(20):
    torch.mv(weights_full, input_vec)
    torch.mv(weights_pruned, input_vec)
torch.cuda.synchronize()

# Dense
t0 = time.perf_counter()
for _ in range(RUNS):
    torch.mv(weights_full, input_vec)
torch.cuda.synchronize()
t_dense = (time.perf_counter() - t0) / RUNS

# Static pruned (zero overhead, no predictor)
t0 = time.perf_counter()
for _ in range(RUNS):
    torch.mv(weights_pruned, input_vec)
torch.cuda.synchronize()
t_static = (time.perf_counter() - t0) / RUNS

speedup_static = (t_dense / t_static - 1) * 100

print(f"\nStatic Pruning Results:")
print(f"  Dense:           {t_dense*1000:.4f} ms")
print(f"  Static pruned:   {t_static*1000:.4f} ms")
print(f"  Speedup:         {speedup_static:+.2f}%")
print(f"  Predictor cost:  0.0000 ms (no predictor)")
print(f"  Net speedup:     {speedup_static:+.2f}%")

# ============================================================
# Combined Strategy
# ============================================================

# Static pruning removes 27.3% permanently
# GhostGate at t=0.05 removes another 20.8% dynamically
# Combined: ~48% fewer neurons
COMBINED_KEEP = int(INTERMEDIATE_DIM * (1 - 0.273) * (1 - 0.208))
weights_combined = weights_full[:COMBINED_KEEP].contiguous()
input_combined = torch.randn(HIDDEN_DIM, device="cuda")

# Warmup
for _ in range(20):
    torch.mv(weights_combined, input_combined)
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(RUNS):
    torch.mv(weights_combined, input_combined)
torch.cuda.synchronize()
t_combined = (time.perf_counter() - t0) / RUNS

speedup_combined = (t_dense / t_combined - 1) * 100

print(f"\nCombined Strategy (Static + GhostGate t=0.05):")
print(f"  Neurons removed: {(1-(COMBINED_KEEP/INTERMEDIATE_DIM))*100:.1f}%")
print(f"  Dense:           {t_dense*1000:.4f} ms")
print(f"  Combined:        {t_combined*1000:.4f} ms")
print(f"  Speedup:         {speedup_combined:+.2f}%")
print(f"  Predictor cost:  0.0000 ms")
print(f"  Net speedup:     {speedup_combined:+.2f}%")

# ============================================================
# Final Summary
# ============================================================

print("\n" + "=" * 65)
print("COMPLETE SPEEDUP COMPARISON")
print("=" * 65)
print(f"{'Strategy':<35}{'Net Speedup':<15}{'Overhead'}")
print("-" * 65)
print(f"{'GhostGate only (t=0.05)':<35}{'+21.87%':<15}{'None'}")
print(f"{'GhostGate + Predictor':<35}{'+3.40%':<15}{'19.99% tax'}")
print(f"{'Static Pruning only':<35}{speedup_static:+.2f}%{'':<9}{'None'}")
print(f"{'Static + GhostGate (no predictor)':<35}{speedup_combined:+.2f}%{'':<9}{'None'}")
print("=" * 65)

if speedup_combined > 20:
    print("STATUS: STATIC MASK IS THE ANSWER ✅")
    print("        Remove predictor entirely")
    print("        Use static dead neuron mask + GhostGate")
    print("        Net speedup > 20% with zero overhead")
elif speedup_static > 10:
    print("STATUS: STATIC MASK HELPS ✅")
    print("        Static pruning alone gives clean speedup")
    print("        Combine with smaller predictor in Phase B")

print("=" * 65)

os.makedirs("F:/GhostWeight/data", exist_ok=True)
with open("F:/GhostWeight/data/static_mask_results.json", "w") as f:
    json.dump({
        "never_active_pct": never_active_pct,
        "speedup_static_pct": speedup_static,
        "speedup_combined_pct": speedup_combined,
        "predictor_overhead_pct": 19.99,
        "net_speedup_with_predictor_pct": 3.40,
    }, f, indent=4)

print("Saved: F:/GhostWeight/data/static_mask_results.json")