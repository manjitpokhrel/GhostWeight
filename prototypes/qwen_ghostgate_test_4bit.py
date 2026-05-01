import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.activations import SiLUActivation

model_id = r"F:\hf_cache\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"

class GhostGate(nn.Module):
    def __init__(self, threshold=0.15):
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
            setattr(parent, parts[-1], GhostGate(threshold))
            replaced += 1
    return replaced

def evaluate(model, tokenizer):
    prompts = [
        "The capital of France is",
        "2 + 2 equals",
        "Water is made of",
        "Einstein developed the theory of",
        "To reverse a string in Python"
    ]

    sparsity_data = []

    def hook_fn(module, input, output):
        with torch.no_grad():
            dead = (output == 0.0).float().mean().item()
            sparsity_data.append(dead)

    handles = []
    for _, module in model.named_modules():
        if isinstance(module, GhostGate):
            handles.append(module.register_forward_hook(hook_fn))

    results = []
    all_sparsity = []

    for prompt in prompts:
        sparsity_data.clear()
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                repetition_penalty=1.2,
                use_cache=True,
            )

        text = tokenizer.decode(out[0], skip_special_tokens=True)
        avg_sparsity = np.mean(sparsity_data) if sparsity_data else 0.0
        all_sparsity.append(avg_sparsity)

        words = text.split()
        unique_ratio = len(set(words)) / max(len(words), 1)
        looping = unique_ratio < 0.5 and len(words) > 5

        results.append({
            "prompt": prompt,
            "text": text,
            "sparsity": avg_sparsity,
            "looping": looping,
        })

    for h in handles:
        h.remove()

    coherent = sum(0 if r["looping"] else 1 for r in results)
    return np.mean(all_sparsity), coherent, results


print("=" * 60)
print("Qwen2.5-7B GhostGate Test (4-bit)")
print("=" * 60)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

thresholds = [0.10, 0.15, 0.20, 0.25]

for t in thresholds:
    print(f"\nTesting threshold = {t}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="cuda",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    replaced = apply_ghostgate(model, t)
    sparsity, coherent, results = evaluate(model, tokenizer)

    print(f"  Layers modified: {replaced}")
    print(f"  Avg sparsity:    {sparsity*100:.2f}%")
    print(f"  Coherent:        {coherent}/5")
    print(f"  Peak VRAM:       {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    for r in results:
        status = "OK" if not r["looping"] else "LOOP"
        print(f"    [{status}] {r['prompt']} -> {r['text'][:90]}")

    del model
    torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("Done")
print("=" * 60)