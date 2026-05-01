import torch
import numpy as np
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.activations import SiLUActivation

model_id = r"F:\hf_cache\hub\models--meta-llama--Llama-3.2-1B-Instruct\snapshots\9213176726f574b556790deb65791e0c5aa438b6"

class GhostGate(nn.Module):
    """
    GhostWeight Core Activation Function.
    Combines SiLU smoothness with ReLU sparsity.
    Preserves dtype to prevent float32/float16 mismatch.
    """
    def __init__(self, threshold=0.1):
        super().__init__()
        self.silu = nn.SiLU()
        self.threshold = threshold

    def forward(self, x):
        # Preserve input dtype throughout
        dtype = x.dtype
        activated = self.silu(x.float()).to(dtype)
        mask = (torch.abs(activated) > self.threshold).to(dtype)
        return activated * mask

print("=" * 60)
print("GhostWeight Phase 1d: GhostGate Soft Surgery")
print("=" * 60)
print(f"GPU: {torch.cuda.get_device_name(0)}")

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    local_files_only=True
)

# ============================================================
# Load two copies: Baseline and GhostGate
# ============================================================
print("\n[1/4] Loading Baseline (SiLU)...")
baseline = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    dtype=torch.float16,
    local_files_only=True
)
baseline.eval()

print("[2/4] Loading GhostGate Model...")
ghost = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu",
    dtype=torch.float16,
    local_files_only=True
)

# Apply GhostGate surgery
replaced = 0
for name, module in ghost.named_modules():
    if isinstance(module, SiLUActivation):
        parts = name.split(".")
        parent = ghost
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], GhostGate(threshold=0.1))
        replaced += 1

print(f"GhostGate applied to {replaced} layers")
ghost = ghost.to("cuda")
ghost.eval()

# ============================================================
# Measure Sparsity of GhostGate
# ============================================================
sparsity_data = []

def ghost_sparsity_hook(module, input, output):
    with torch.no_grad():
        dead = (output == 0.0).float().mean().item()
        sparsity_data.append(dead)

for name, module in ghost.named_modules():
    if isinstance(module, GhostGate):
        module.register_forward_hook(ghost_sparsity_hook)

# ============================================================
# Quality + Sparsity Test Together
# ============================================================
prompts = [
    "The capital of France is",
    "The theory of relativity states that",
    "To sort a list in Python you can use",
    "Neural networks learn by",
    "The largest planet in our solar system is"
]

print("\n[3/4] Testing Quality + Sparsity...")
print("=" * 60)

results = []
all_sparsity = []

for prompt in prompts:
    sparsity_data.clear()
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        base_out = baseline.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False
        )
        ghost_out = ghost.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False
        )

    base_text = tokenizer.decode(base_out[0], skip_special_tokens=True)
    ghost_text = tokenizer.decode(ghost_out[0], skip_special_tokens=True)

    sparsity = np.mean(sparsity_data) if sparsity_data else 0.0
    all_sparsity.append(sparsity)

    coherent = not any(
        word * 3 in ghost_text
        for word in ghost_text.split()
        if len(word) > 2
    )

    results.append({
        "match": base_text == ghost_text,
        "coherent": coherent
    })

    print(f"\nPrompt:   {prompt}")
    print(f"Baseline: {base_text}")
    print(f"Ghost:    {ghost_text}")
    print(f"Sparsity: {sparsity*100:.2f}% | Coherent: {'✅' if coherent else '⚠️'}")
    print("-" * 60)

# ============================================================
# Final Results
# ============================================================
overall_sparsity = np.mean(all_sparsity)
coherent_count = sum(1 for r in results if r["coherent"])

print("\n" + "=" * 60)
print("GHOSTGATE RESULTS")
print("=" * 60)
print(f"GhostGate Sparsity:    {overall_sparsity*100:.2f}%")
print(f"Coherent Outputs:      {coherent_count}/5")
print(f"VRAM Peak:             {torch.cuda.max_memory_allocated(0)/1e9:.2f} GB")
print("=" * 60)

if overall_sparsity >= 0.40 and coherent_count >= 4:
    print("STATUS: BREAKTHROUGH")
    print("        High sparsity + coherent output achieved")
    print("        GhostGate is the core innovation")
    print("        Ready for Phase 2: Sparse Weight Streamer")
elif overall_sparsity >= 0.30 and coherent_count >= 3:
    print("STATUS: PROMISING")
    print("        Adjust threshold and retest")
elif coherent_count < 2:
    print("STATUS: THRESHOLD TOO HIGH")
    print("        Lower threshold to 0.05 and retest")
print("=" * 60)