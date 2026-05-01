import torch
import torch.nn as nn
import time
import json
import os

# ============================================================
# Measure the real cost of running the Ghost Predictor
# on every forward pass
# ============================================================

class GhostPredictorLogits(nn.Module):
    def __init__(self, hidden_dim=3584, intermediate_dim=18944, bottleneck=256):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck),
            nn.LayerNorm(bottleneck),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(bottleneck, bottleneck),
            nn.LayerNorm(bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, intermediate_dim)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, -1, :]
        return self.predictor(x)

    def predict_mask(self, x, threshold=0.5):
        with torch.no_grad():
            logits = self.forward(x)
            return (torch.sigmoid(logits) > threshold).float()

print("=" * 65)
print("GHOSTWEIGHT PREDICTOR OVERHEAD MEASUREMENT")
print("=" * 65)
print(f"GPU: {torch.cuda.get_device_name(0)}")

HIDDEN_DIM = 3584
INTERMEDIATE_DIM = 18944
NUM_LAYERS = 28  # Qwen-7B has 28 layers
RUNS = 500

# Load trained predictor if available
predictor = GhostPredictorLogits(
    hidden_dim=HIDDEN_DIM,
    intermediate_dim=INTERMEDIATE_DIM,
    bottleneck=256
).cuda()

predictor_path = "F:/GhostWeight/models/ghost_predictor_v2.pt"
if os.path.exists(predictor_path):
    predictor.load_state_dict(
        torch.load(predictor_path, map_location="cuda")
    )
    print("Loaded trained predictor from disk.")
else:
    print("Using untrained predictor (timing only).")

predictor.eval()

param_count = sum(p.numel() for p in predictor.parameters())
print(f"Predictor size: {param_count:,} params ({param_count*4/1e6:.1f} MB)")

# ============================================================
# Test 1: Single predictor call overhead
# ============================================================

hidden_state = torch.randn(1, 1, HIDDEN_DIM, device="cuda")

# Warmup
for _ in range(50):
    predictor.predict_mask(hidden_state)
torch.cuda.synchronize()

# Time single call
t0 = time.perf_counter()
for _ in range(RUNS):
    mask = predictor.predict_mask(hidden_state)
torch.cuda.synchronize()
t_single = (time.perf_counter() - t0) / RUNS

print(f"\nSingle predictor call: {t_single*1000:.4f} ms")

# ============================================================
# Test 2: Full layer overhead (predictor + pack + compute)
# ============================================================

SPARSITY = 0.2080  # threshold=0.05 production point
ACTIVE = int(HIDDEN_DIM * (1 - SPARSITY))

weights = torch.randn(INTERMEDIATE_DIM, HIDDEN_DIM, device="cuda")
dense_input = torch.randn(HIDDEN_DIM, device="cuda")
packed_input = torch.randn(ACTIVE, device="cuda")
indices = torch.randperm(HIDDEN_DIM, device="cuda")[:ACTIVE].long()
packed_weights = weights[:, indices].contiguous()

# Warmup
for _ in range(20):
    torch.mv(weights, dense_input)
    mask = predictor.predict_mask(hidden_state)
    torch.mv(packed_weights, packed_input)
torch.cuda.synchronize()

# Dense baseline (no predictor)
t0 = time.perf_counter()
for _ in range(RUNS):
    torch.mv(weights, dense_input)
torch.cuda.synchronize()
t_dense = (time.perf_counter() - t0) / RUNS

# Ghost with predictor overhead included
t0 = time.perf_counter()
for _ in range(RUNS):
    # Step 1: Predict (overhead)
    mask = predictor.predict_mask(hidden_state)
    # Step 2: Compute sparse matmul
    torch.mv(packed_weights, packed_input)
torch.cuda.synchronize()
t_ghost_with_predictor = (time.perf_counter() - t0) / RUNS

# Ghost without predictor (pure compute)
t0 = time.perf_counter()
for _ in range(RUNS):
    torch.mv(packed_weights, packed_input)
torch.cuda.synchronize()
t_ghost_no_predictor = (time.perf_counter() - t0) / RUNS

# ============================================================
# Test 3: Full model overhead (28 layers)
# ============================================================

t_full_dense = t_dense * NUM_LAYERS
t_full_ghost_no_pred = t_ghost_no_predictor * NUM_LAYERS
t_full_ghost_with_pred = t_ghost_with_predictor * NUM_LAYERS

speedup_no_pred = (t_full_dense / t_full_ghost_no_pred - 1) * 100
speedup_with_pred = (t_full_dense / t_full_ghost_with_pred - 1) * 100
predictor_tax = speedup_no_pred - speedup_with_pred

# ============================================================
# Results
# ============================================================

print("\n" + "=" * 65)
print("PREDICTOR OVERHEAD RESULTS")
print("=" * 65)
print(f"\nSingle Layer Analysis (Qwen-7B, threshold=0.05):")
print(f"  Dense baseline:              {t_dense*1000:.4f} ms")
print(f"  Ghost (no predictor):        {t_ghost_no_predictor*1000:.4f} ms")
print(f"  Ghost (with predictor):      {t_ghost_with_predictor*1000:.4f} ms")
print(f"  Predictor cost alone:        {t_single*1000:.4f} ms")
print(f"  Predictor overhead ratio:    {t_single/t_dense*100:.2f}% of dense")
print()
print(f"Full Model Analysis ({NUM_LAYERS} layers):")
print(f"  Dense total:                 {t_full_dense*1000:.2f} ms/token")
print(f"  Ghost without predictor:     {t_full_ghost_no_pred*1000:.2f} ms/token")
print(f"  Ghost with predictor:        {t_full_ghost_with_pred*1000:.2f} ms/token")
print()
print(f"Speedup Analysis:")
print(f"  Speedup (no predictor):      {speedup_no_pred:+.2f}%")
print(f"  Speedup (with predictor):    {speedup_with_pred:+.2f}%")
print(f"  Predictor tax:               -{predictor_tax:.2f}%")
print()
print(f"Net Speedup at threshold=0.05: {speedup_with_pred:+.2f}%")
print("=" * 65)

if speedup_with_pred > 10:
    print("STATUS: PREDICTOR OVERHEAD ACCEPTABLE ✅")
    print("        Net speedup remains positive after predictor cost")
elif speedup_with_pred > 0:
    print("STATUS: MARGINAL ⚠️")
    print("        Predictor cost reduces but does not eliminate speedup")
else:
    print("STATUS: PREDICTOR TOO EXPENSIVE ❌")
    print("        Need smaller predictor or async execution")

print("=" * 65)

os.makedirs("F:/GhostWeight/data", exist_ok=True)
with open("F:/GhostWeight/data/predictor_overhead.json", "w") as f:
    json.dump({
        "predictor_single_call_ms": t_single * 1000,
        "dense_layer_ms": t_dense * 1000,
        "ghost_no_predictor_ms": t_ghost_no_predictor * 1000,
        "ghost_with_predictor_ms": t_ghost_with_predictor * 1000,
        "speedup_no_predictor_pct": speedup_no_pred,
        "speedup_with_predictor_pct": speedup_with_pred,
        "predictor_tax_pct": predictor_tax,
        "num_layers": NUM_LAYERS,
        "sparsity_pct": SPARSITY * 100,
        "threshold": 0.05,
    }, f, indent=4)

print("Saved: F:/GhostWeight/data/predictor_overhead.json")