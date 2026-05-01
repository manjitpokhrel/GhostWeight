import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import numpy as np
import json
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.activations import SiLUActivation

model_id = r"F:\hf_cache\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"

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

def apply_ghostgate(model, threshold):
    for name, module in model.named_modules():
        if isinstance(module, SiLUActivation):
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], GhostGate(threshold))

def free_vram():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def load_model():
    free_vram()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="cuda",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model

def compute_perplexity(model, tokenizer, texts, max_length=256):
    total_nll = 0.0
    total_tokens = 0

    for text in texts:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        )
        input_ids = enc.input_ids.to("cuda")
        if input_ids.shape[1] < 2:
            continue
        with torch.no_grad():
            out = model(input_ids, labels=input_ids)
            nll = out.loss.item()
        total_nll += nll * (input_ids.shape[1] - 1)
        total_tokens += input_ids.shape[1] - 1

    return torch.exp(torch.tensor(total_nll / total_tokens)).item()

eval_texts = [
    "The history of artificial intelligence began in antiquity with myths and stories of artificial beings endowed with intelligence by master craftsmen.",
    "Machine learning is a method of data analysis that automates analytical model building based on the idea that systems can learn from data.",
    "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
    "Natural language processing is a subfield of linguistics computer science and artificial intelligence concerned with interactions between computers and human language.",
    "The transformer architecture relies on self-attention mechanisms to process sequential data in parallel without recurrence.",
    "Generative adversarial networks consist of two neural networks that contest with each other in a zero-sum game framework.",
    "Transfer learning is a machine learning method where a model developed for one task is reused as the starting point for a model on a different task.",
    "The vanishing gradient problem occurs when the gradient becomes too small effectively preventing the weight from changing its value.",
    "Batch normalization is a technique to improve training of deep neural networks by normalizing the inputs of each layer.",
    "Dropout is a regularization technique for reducing overfitting in neural networks by preventing complex co-adaptations on training data.",
]

print("=" * 60)
print("GhostWeight Perplexity Evaluation v2")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

# Baseline
print("\nComputing baseline perplexity...")
model = load_model()
baseline_ppl = compute_perplexity(model, tokenizer, eval_texts)
print(f"Baseline: {baseline_ppl:.4f}")
del model
free_vram()

# Thresholds
thresholds = [0.05, 0.10, 0.15, 0.20, 0.25]
results = []

sparsity_map = {
    0.05: 20.80,
    0.10: 34.43,
    0.15: 44.54,
    0.20: 55.85,
    0.25: 70.28
}

for t in thresholds:
    print(f"\nTesting threshold={t}...")
    model = load_model()
    apply_ghostgate(model, t)
    ppl = compute_perplexity(model, tokenizer, eval_texts)
    delta_pct = (ppl - baseline_ppl) / baseline_ppl * 100

    if delta_pct < 5.0:
        status = "✅ NEGLIGIBLE"
    elif delta_pct < 15.0:
        status = "✅ ACCEPTABLE"
    elif delta_pct < 50.0:
        status = "⚠️ MODERATE"
    else:
        status = "❌ SIGNIFICANT"

    results.append({
        "threshold": t,
        "sparsity": sparsity_map[t],
        "perplexity": ppl,
        "delta_pct": delta_pct,
        "status": status
    })

    print(f"  Perplexity: {ppl:.4f} | Delta: {delta_pct:+.2f}% | {status}")
    del model
    free_vram()

# Results
print("\n" + "=" * 70)
print("PERPLEXITY TRADEOFF CURVE")
print("=" * 70)
print(f"{'Threshold':<12}{'Sparsity':<12}{'Perplexity':<14}{'Delta':<12}{'Status'}")
print("-" * 70)
print(f"{'Baseline':<12}{'0.00%':<12}{baseline_ppl:<14.4f}{'0.00%':<12}✅ REFERENCE")
for r in results:
    print(f"{r['threshold']:<12}{r['sparsity']:<12.2f}{r['perplexity']:<14.4f}{r['delta_pct']:+.2f}%{'':<6}{r['status']}")
print("=" * 70)

os.makedirs("F:/GhostWeight/data", exist_ok=True)
with open("F:/GhostWeight/data/perplexity_curve.json", "w") as f:
    json.dump({
        "baseline": baseline_ppl,
        "results": results,
        "model": "Qwen2.5-7B-Instruct 4bit NF4",
        "hardware": "RTX 5060 8GB Blackwell"
    }, f, indent=4)

print("\nSaved: F:/GhostWeight/data/perplexity_curve.json")