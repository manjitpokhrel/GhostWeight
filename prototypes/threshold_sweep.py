import torch
import numpy as np
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.activations import SiLUActivation

model_id = r"F:\hf_cache\hub\models--meta-llama--Llama-3.2-1B-Instruct\snapshots\9213176726f574b556790deb65791e0c5aa438b6"

class GhostGate(nn.Module):
    def __init__(self, threshold=0.1):
        super().__init__()
        self.silu = nn.SiLU()
        self.threshold = threshold

    def forward(self, x):
        dtype = x.dtype
        activated = self.silu(x.float()).to(dtype)
        mask = (torch.abs(activated) > self.threshold).to(dtype)
        return activated * mask

def apply_ghostgate(model, threshold):
    replaced = 0
    for name, module in model.named_modules():
        if isinstance(module, SiLUActivation):
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], GhostGate(threshold=threshold))
            replaced += 1
    return replaced

def measure_sparsity_and_quality(model, tokenizer, prompts, device):
    sparsity_data = []

    def hook_fn(module, input, output):
        with torch.no_grad():
            dead = (output == 0.0).float().mean().item()
            sparsity_data.append(dead)

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, GhostGate):
            handles.append(module.register_forward_hook(hook_fn))

    results = []
    all_sparsity = []

    for prompt in prompts:
        sparsity_data.clear()
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False
            )

        text = tokenizer.decode(out[0], skip_special_tokens=True)
        sparsity = np.mean(sparsity_data) if sparsity_data else 0.0
        all_sparsity.append(sparsity)

        # Check coherence
        words = [w for w in text.split() if len(w) > 2]
        looping = any(words.count(w) > 4 for w in words) if words else True
        results.append(not looping)

    for h in handles:
        h.remove()

    return np.mean(all_sparsity), sum(results), results

# ============================================================
# Main Sweep
# ============================================================

print("=" * 60)
print("GhostWeight Threshold Sweep")
print("Finding Maximum Sparsity Before Collapse")
print("=" * 60)
print(f"GPU: {torch.cuda.get_device_name(0)}")

tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

prompts = [
    "The capital of France is",
    "The theory of relativity states that",
    "To sort a list in Python you can use",
    "Neural networks learn by",
    "The largest planet in our solar system is"
]

thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

sweep_results = []

for threshold in thresholds:
    print(f"\nTesting threshold={threshold}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        dtype=torch.float16,
        local_files_only=True
    )
    model.eval()

    apply_ghostgate(model, threshold)

    sparsity, coherent, _ = measure_sparsity_and_quality(
        model, tokenizer, prompts, "cuda"
    )

    sweep_results.append({
        "threshold": threshold,
        "sparsity": sparsity,
        "coherent": coherent
    })

    print(f"  Sparsity: {sparsity*100:.2f}% | Coherent: {coherent}/5")

    # Free VRAM between runs
    del model
    torch.cuda.empty_cache()

# ============================================================
# Results Table
# ============================================================

print("\n" + "=" * 60)
print("THRESHOLD SWEEP RESULTS")
print("=" * 60)
print(f"{'Threshold':<12} {'Sparsity':<12} {'Coherent':<12} {'Status'}")
print("-" * 60)

best = None
for r in sweep_results:
    status = "✅ VIABLE" if r["coherent"] >= 4 else "⚠️ DEGRADED" if r["coherent"] >= 2 else "❌ COLLAPSED"
    print(f"{r['threshold']:<12} {r['sparsity']*100:<12.2f} {r['coherent']}/5         {status}")
    if r["coherent"] >= 4 and (best is None or r["sparsity"] > best["sparsity"]):
        best = r

print("=" * 60)

if best:
    print(f"\nOPTIMAL GHOSTGATE THRESHOLD: {best['threshold']}")
    print(f"Maximum Viable Sparsity:     {best['sparsity']*100:.2f}%")
    print(f"VRAM Multiplier:             {1/max(1-best['sparsity'], 0.01):.2f}x")
    print(f"\nThis threshold becomes the GhostGate parameter.")
    print(f"Phase 2: Build the Sparse Weight Streamer around this value.")

print("=" * 60)