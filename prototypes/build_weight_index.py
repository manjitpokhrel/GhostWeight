import os
import torch
import numpy as np
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.activations import SiLUActivation
import json

model_id = r"F:\hf_cache\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"

# ============================================================
# GhostGate (same as Phase 1)
# ============================================================

class GhostGate(nn.Module):
    def __init__(self, threshold=0.25):
        super().__init__()
        self.silu = nn.SiLU()
        self.threshold = threshold

    def forward(self, x):
        dtype = x.dtype
        activated = self.silu(x.float()).to(dtype)
        mask = (torch.abs(activated) > self.threshold).to(dtype)
        return activated * mask

# ============================================================
# Weight Index Builder
#
# For each layer, we record:
# - Which output neurons fired (non-zero after GhostGate)
# - Across many different prompts
# - We then cluster these into "activation patterns"
# - Each pattern becomes a "Weight Block" we can pre-fetch
# ============================================================

print("=" * 60)
print("GhostWeight Phase 2 — Weight Index Builder")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="cuda",
    local_files_only=True,
    low_cpu_mem_usage=True,
)
model.eval()

# Apply GhostGate
replaced = 0
for name, module in model.named_modules():
    if isinstance(module, SiLUActivation):
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], GhostGate(threshold=0.25))
        replaced += 1

print(f"GhostGate applied: {replaced} layers")

# ============================================================
# Hook: Record activation MASKS per layer
# We record a binary mask (1=active, 0=dead) for each layer
# across many prompts. This builds our "index."
# ============================================================

# layer_index[layer_name] = list of binary masks
layer_activation_masks = {}

def make_hook(layer_name):
    def hook_fn(module, input, output):
        with torch.no_grad():
            # Binary mask: which neurons fired
            mask = (output != 0.0).float()
            # Average across batch and sequence dims
            # Shape: [hidden_dim]
            avg_mask = mask.mean(dim=[0, 1]).cpu().numpy()
            if layer_name not in layer_activation_masks:
                layer_activation_masks[layer_name] = []
            layer_activation_masks[layer_name].append(avg_mask)
    return hook_fn

# Attach hooks to GhostGate layers
hook_count = 0
for name, module in model.named_modules():
    if isinstance(module, GhostGate):
        module.register_forward_hook(make_hook(name))
        hook_count += 1

print(f"Index hooks attached: {hook_count} layers")

# ============================================================
# Diverse prompts to build a representative index
# We want to cover: factual, math, code, science, reasoning
# ============================================================

index_prompts = [
    # Factual
    "The capital of France is",
    "The largest ocean on Earth is",
    "The chemical formula for water is",
    "Mount Everest is located in",
    "The speed of light is approximately",
    # Math
    "2 + 2 equals",
    "The square root of 144 is",
    "Solve for x: 2x + 5 = 15",
    "The derivative of x squared is",
    "Pi is approximately equal to",
    # Code
    "To reverse a string in Python",
    "A binary search algorithm works by",
    "To sort a list in Python you can use",
    "A recursive function in Python must have",
    "The time complexity of quicksort is",
    # Science
    "Einstein developed the theory of",
    "Neural networks learn by",
    "Gravity pulls objects",
    "DNA is made of",
    "The human brain contains approximately",
    # Reasoning
    "If all cats are mammals and all mammals are animals then",
    "The most efficient way to travel between two points is",
    "To debug a program you should first",
    "When a model overfits it means",
    "The difference between supervised and unsupervised learning is",
]

print(f"\nBuilding index from {len(index_prompts)} diverse prompts...")
print("-" * 60)

for i, prompt in enumerate(index_prompts):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        _ = model(**inputs)
    if (i + 1) % 5 == 0:
        print(f"  Processed {i+1}/{len(index_prompts)} prompts...")

print("\nIndex collection complete.")

# ============================================================
# Build the Weight Index
# For each layer, compute:
# 1. Average activation frequency per neuron
# 2. "Always active" neurons (fire >80% of time)
# 3. "Never active" neurons (fire <20% of time)
# 4. "Context dependent" neurons (fire 20-80% of time)
# ============================================================

print("\nAnalyzing activation patterns...")

weight_index = {}

for layer_name, masks in layer_activation_masks.items():
    masks_array = np.array(masks)  # [num_prompts, hidden_dim]

    # Average firing rate per neuron
    avg_firing = masks_array.mean(axis=0)

    always_active = (avg_firing > 0.80).sum()
    never_active = (avg_firing < 0.20).sum()
    context_dependent = ((avg_firing >= 0.20) & (avg_firing <= 0.80)).sum()
    total = len(avg_firing)

    weight_index[layer_name] = {
        "total_neurons": int(total),
        "always_active": int(always_active),
        "never_active": int(never_active),
        "context_dependent": int(context_dependent),
        "always_active_pct": float(always_active / total * 100),
        "never_active_pct": float(never_active / total * 100),
        "context_dependent_pct": float(context_dependent / total * 100),
        "avg_firing_rate": float(avg_firing.mean()),
        # Store the top always-active neuron indices
        # These are the "Core Weights" we always pre-fetch
        "core_neuron_indices": avg_firing.argsort()[-int(always_active):][::-1].tolist()
    }

# ============================================================
# Print Summary
# ============================================================

print("\n" + "=" * 60)
print("WEIGHT INDEX SUMMARY")
print("=" * 60)

total_always = 0
total_never = 0
total_context = 0
total_neurons = 0

for layer_name, stats in weight_index.items():
    total_always += stats["always_active"]
    total_never += stats["never_active"]
    total_context += stats["context_dependent"]
    total_neurons += stats["total_neurons"]

print(f"Layers indexed:          {len(weight_index)}")
print(f"Total neurons mapped:    {total_neurons:,}")
print(f"Always active:           {total_always:,} ({total_always/total_neurons*100:.1f}%)")
print(f"Never active:            {total_never:,} ({total_never/total_neurons*100:.1f}%)")
print(f"Context dependent:       {total_context:,} ({total_context/total_neurons*100:.1f}%)")
print("=" * 60)
print()
print("STREAMING STRATEGY:")
print(f"  Core load (always):    {total_always/total_neurons*100:.1f}% of weights")
print(f"  Skip entirely (never): {total_never/total_neurons*100:.1f}% of weights")
print(f"  Predict + fetch:       {total_context/total_neurons*100:.1f}% of weights")
print()
print("This index tells Phase 2 exactly what to stream.")
print("=" * 60)

# ============================================================
# Save Index
# ============================================================

os.makedirs("F:/GhostWeight/data", exist_ok=True)
index_path = "F:/GhostWeight/data/weight_index.json"

# Save without core_neuron_indices for readability
# (those are large arrays)
summary_index = {
    k: {
        key: val for key, val in v.items()
        if key != "core_neuron_indices"
    }
    for k, v in weight_index.items()
}

with open(index_path, "w") as f:
    json.dump(summary_index, f, indent=4)

# Save full index with numpy for Phase 2 use
import pickle
full_index_path = "F:/GhostWeight/data/weight_index_full.pkl"
with open(full_index_path, "wb") as f:
    pickle.dump(weight_index, f)

print(f"\nIndex saved:")
print(f"  Summary: {index_path}")
print(f"  Full:    {full_index_path}")
print()
print("Phase 2 Component 1: COMPLETE")
print("Next: Ghost Predictor (Component 2)")
print("=" * 60)