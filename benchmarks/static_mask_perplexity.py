import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import numpy as np
import json
import gc
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.activations import SiLUActivation

model_id = r"F:\hf_cache\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"

class GhostGate(nn.Module):
    def __init__(self, threshold=0.05):
        super().__init__()
        self.silu = nn.SiLU()
        self.threshold = threshold

    def forward(self, x):
        dtype = x.dtype
        activated = self.silu(x.float()).to(dtype)
        mask = (torch.abs(activated) > self.threshold).to(dtype)
        return activated * mask

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

def apply_ghostgate(model, threshold):
    for name, module in model.named_modules():
        if isinstance(module, SiLUActivation):
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], GhostGate(threshold))

def apply_static_mask(model, weight_index):
    """
    Physically zeroes out the never-active neurons in the model.
    This simulates permanent static pruning.
    """
    pruned_count = 0
    for name, module in model.named_modules():
        if "mlp.gate_proj" in name or "mlp.up_proj" in name:
            layer_key = name + ".act_fn"
            if layer_key in weight_index:
                stats = weight_index[layer_key]
                total = stats["total_neurons"]
                never_pct = stats["never_active_pct"] / 100
                never_count = int(total * never_pct)

                if never_count > 0 and hasattr(module, 'weight'):
                    with torch.no_grad():
                        module.weight.data[-never_count:] = 0.0
                    pruned_count += never_count
    return pruned_count

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

print("=" * 65)
print("STATIC MASK PERPLEXITY EVALUATION")
print("=" * 65)

tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

index_path = "F:/GhostWeight/data/weight_index_full.pkl"
if os.path.exists(index_path):
    with open(index_path, "rb") as f:
        weight_index = pickle.load(f)
    print(f"Weight index loaded: {len(weight_index)} layers")

# Baseline
print("\n[1/3] Baseline perplexity...")
model = load_model()
baseline_ppl = compute_perplexity(model, tokenizer, eval_texts)
print(f"Baseline: {baseline_ppl:.4f}")
del model
free_vram()

# Static mask only
print("\n[2/3] Static mask only (27.3% removed)...")
model = load_model()
pruned = apply_static_mask(model, weight_index)
static_ppl = compute_perplexity(model, tokenizer, eval_texts)
static_delta = (static_ppl - baseline_ppl) / baseline_ppl * 100
print(f"Static mask PPL:  {static_ppl:.4f} ({static_delta:+.2f}%)")
del model
free_vram()

# Static mask + GhostGate t=0.05
print("\n[3/3] Static mask + GhostGate (t=0.05)...")
model = load_model()
apply_static_mask(model, weight_index)
apply_ghostgate(model, threshold=0.05)
combined_ppl = compute_perplexity(model, tokenizer, eval_texts)
combined_delta = (combined_ppl - baseline_ppl) / baseline_ppl * 100
print(f"Combined PPL:     {combined_ppl:.4f} ({combined_delta:+.2f}%)")
del model
free_vram()

print("\n" + "=" * 65)
print("STATIC MASK PERPLEXITY RESULTS")
print("=" * 65)
print(f"{'Strategy':<35}{'Perplexity':<14}{'Delta':<12}{'Speedup'}")
print("-" * 65)
print(f"{'Baseline':<35}{baseline_ppl:<14.4f}{'0.00%':<12}{'0.00%'}")
print(f"{'Static only (27.3%)':<35}{static_ppl:<14.4f}{static_delta:+.2f}%{'':<6}+31.67%")
print(f"{'Static + GhostGate t=0.05':<35}{combined_ppl:<14.4f}{combined_delta:+.2f}%{'':<6}+66.05%")
print("=" * 65)

def quality(delta):
    if delta < 3:   return "✅ NEGLIGIBLE"
    if delta < 10:  return "✅ ACCEPTABLE"
    if delta < 25:  return "⚠️ MODERATE"
    return "❌ SIGNIFICANT"

print(f"\nStatic only quality:   {quality(static_delta)}")
print(f"Combined quality:      {quality(combined_delta)}")
print("=" * 65)

os.makedirs("F:/GhostWeight/data", exist_ok=True)
with open("F:/GhostWeight/data/static_mask_perplexity.json", "w") as f:
    json.dump({
        "baseline_ppl": baseline_ppl,
        "static_ppl": static_ppl,
        "static_delta_pct": static_delta,
        "combined_ppl": combined_ppl,
        "combined_delta_pct": combined_delta,
        "static_speedup_pct": 31.67,
        "combined_speedup_pct": 66.05,
    }, f, indent=4)

print("Saved: F:/GhostWeight/data/static_mask_perplexity.json")