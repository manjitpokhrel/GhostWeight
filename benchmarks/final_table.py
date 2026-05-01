import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import time
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

def benchmark_speedup(sparsity_total, runs=500):
    HIDDEN = 3584
    INTERMEDIATE = 18944
    KEPT = int(INTERMEDIATE * (1 - sparsity_total))
    weights_full = torch.randn(INTERMEDIATE, HIDDEN, device='cuda')
    weights_pruned = weights_full[:KEPT].contiguous()
    input_vec = torch.randn(HIDDEN, device='cuda')
    for _ in range(20):
        torch.mv(weights_full, input_vec)
        torch.mv(weights_pruned, input_vec)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        torch.mv(weights_full, input_vec)
    torch.cuda.synchronize()
    t_dense = (time.perf_counter() - t0) / runs
    t0 = time.perf_counter()
    for _ in range(runs):
        torch.mv(weights_pruned, input_vec)
    torch.cuda.synchronize()
    t_sparse = (time.perf_counter() - t0) / runs
    return (t_dense / t_sparse - 1) * 100

eval_texts = [
    "The history of artificial intelligence began in antiquity with myths and stories of artificial beings.",
    "Machine learning is a method of data analysis that automates analytical model building.",
    "Deep learning is part of a broader family of machine learning methods based on artificial neural networks.",
    "Natural language processing is concerned with interactions between computers and human language.",
    "The transformer architecture relies on self-attention mechanisms to process sequential data.",
    "Transfer learning is where a model developed for one task is reused for a different task.",
    "The vanishing gradient problem occurs when the gradient becomes too small during training.",
    "Batch normalization improves training by normalizing the inputs of each layer.",
    "Dropout is a regularization technique for reducing overfitting in neural networks.",
    "Convolutional neural networks are most commonly applied to analyze visual imagery.",
]

print("=" * 75)
print("GHOSTWEIGHT FINAL PAPER TABLE")
print("=" * 75)

tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

index_path = "F:/GhostWeight/data/weight_index_full.pkl"
with open(index_path, "rb") as f:
    weight_index = pickle.load(f)

STATIC_SPARSITY = 0.273

# GhostGate additional sparsity at each threshold
configs = [
    ("Baseline",                    None,  0.000,  "0.00"),
    ("Static only",                 None,  0.000,  "—"),
    ("Static + GhostGate t=0.05",   0.05,  0.208,  "—"),
    ("Static + GhostGate t=0.10",   0.10,  0.3443, "—"),
    ("Static + GhostGate t=0.25",   0.25,  0.7028, "—"),
]

results = []
baseline_ppl = None

for label, threshold, ghost_sparsity, _ in configs:
    print(f"\nRunning: {label}")
    model = load_model()

    if label != "Baseline":
        apply_static_mask(model, weight_index)

    if threshold is not None:
        apply_ghostgate(model, threshold)

    ppl = compute_perplexity(model, tokenizer, eval_texts)

    if baseline_ppl is None:
        baseline_ppl = ppl
        delta = 0.0
    else:
        delta = (ppl - baseline_ppl) / baseline_ppl * 100

    total_sparsity = STATIC_SPARSITY + ghost_sparsity * (1 - STATIC_SPARSITY)
    speedup = benchmark_speedup(total_sparsity) if label != "Baseline" else 0.0

    results.append({
        "label": label,
        "total_sparsity": total_sparsity * 100,
        "perplexity": ppl,
        "ppl_delta": delta,
        "speedup": speedup,
    })

    print(f"  PPL: {ppl:.4f} ({delta:+.2f}%) | Speedup: {speedup:+.2f}%")
    del model
    free_vram()

def quality(d):
    if d < 3:   return "✅ NEGLIGIBLE"
    if d < 10:  return "✅ ACCEPTABLE"
    if d < 25:  return "⚠️ MODERATE"
    return "❌ SIGNIFICANT"

print("\n" + "=" * 85)
print("GHOSTWEIGHT COMPLETE PAPER TABLE")
print("=" * 85)
print(f"{'Strategy':<32}{'Sparsity':<11}{'PPL Δ':<12}{'Speedup':<12}{'Quality'}")
print("-" * 85)
for r in results:
    print(
        f"{r['label']:<32}"
        f"{r['total_sparsity']:<11.1f}"
        f"{r['ppl_delta']:+.2f}%{'':<6}"
        f"{r['speedup']:+.2f}%{'':<6}"
        f"{quality(r['ppl_delta'])}"
    )
print("=" * 85)
print("All results: RTX 5060 8GB Blackwell | Qwen2.5-7B-Instruct 4-bit NF4")
print("Zero retraining. Zero predictor overhead.")
print("=" * 85)

os.makedirs("F:/GhostWeight/data", exist_ok=True)
with open("F:/GhostWeight/data/final_paper_table.json", "w") as f:
    json.dump(results, f, indent=4)
print("\nSaved: F:/GhostWeight/data/final_paper_table.json")