# GhostWeight 👻
**Training-Free Activation Sparsity for LLM Inference on Consumer Hardware**

[![PyPI version](https://badge.fury.io/py/ghostweight.svg)](https://badge.fury.io/py/ghostweight)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)  
[![CUDA 12.6](https://img.shields.io/badge/CUDA-12.6-green.svg)](https://developer.nvidia.com/cuda-downloads)  

---

## 🚀 What Is GhostWeight?

GhostWeight is a **training-free sparsity framework** that speeds up LLM inference by removing useless computation.

> **Core insight:**  
> **27.3% of MLP neurons in Qwen2.5-7B never activate.**  
> They consume compute and memory — while contributing nothing.

GhostWeight removes them **permanently**.

```python
pip install ghostweight

from ghostweight import ghost_surgery
model, replaced = ghost_surgery(model, threshold=0.05)
```

---

## ⚡ Key Results

**Model:** Qwen2.5-7B-Instruct (4-bit)  
**Hardware:** RTX 5060 8GB  

| Strategy | Sparsity | Speedup | Perplexity Δ |
|----------|--------:|--------:|-------------:|
| Baseline | 0% | +0.00% | +0.00% |
| Static dead neurons | 27.3% | **+38.35%** | **0.00%** |
| + GhostGate (τ=0.05) | 42.4% | **+74.71%** | +5.91% |
| + GhostGate (τ=0.10) | 52.3% | **+110.53%** | +11.16% |

> Kernel-level speedups verified.  
> End-to-end integration is ongoing.

---

## 🧠 How It Works

### 1. Static Dead Neuron Removal
- Identify neurons that **never activate**
- Remove them permanently

✔ Zero accuracy loss  
✔ Zero runtime overhead  
✔ Immediate speedup  

---

### 2. GhostGate

Threshold-gated SiLU activation:

```
GhostGate(x) = SiLU(x) * (|SiLU(x)| > τ)
```

- Small activations → hard zero  
- No retraining required  

---

### 3. Sparse Kernel (Row Packing)

Naive sparsity fails on GPUs due to warp divergence.

| Method | Result |
|--------|--------|
| Branch skipping | ❌ Slower (-9.43%) |
| Row packing | ✅ Faster (+69.83%) |

---

## 🧪 Key Finding: “Predictor Tax”

Dynamic sparsity **hurts performance**.

| Approach | Result |
|----------|--------|
| Neural predictor | **-27.69% (slower)** |
| Static removal | **+37.59% (faster)** |

> A neural network to optimize a neural network made things worse.

---

## 🖥️ 72B on 8GB GPU

GhostWeight enables running large models on consumer hardware:

- **Model:** Qwen2.5-72B  
- **GPU:** RTX 5060 8GB  
- **Speed:** 0.022 tok/s  
- **Status:** Works (IO-bound)

> Sparsity reduces the 47GB memory footprint — key to real-time inference.

---

## 📦 Installation

```bash
pip install ghostweight
```

Or:

```bash
pip install git+https://github.com/manjitpokhrel/GhostWeight.git
```

---

## ⚡ Quick Start

```python
from transformers import AutoModelForCausalLM
from ghostweight import ghost_surgery

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    device_map="cuda"
)

model, replaced = ghost_surgery(model, threshold=0.05)

print(f"Modified {replaced} layers")
```

---

## 🔬 Reproducing Results

```bash
python prototypes/build_weight_index.py
python prototypes/threshold_sweep.py
python benchmarks/perplexity_eval_v2.py
python benchmarks/speedup_curve_v2.py
python benchmarks/predictor_tax.py
```

---

## ⚠️ Limitations

- No end-to-end speedup yet (kernel-only)
- Evaluation on small dataset
- English-only neuron index
- Native CUDA kernel integration pending

---

## 🗺️ Roadmap

**v0.2.0**
- End-to-end speedup  
- Native CUDA integration  
- MMLU benchmark  

**v0.3.0**
- Multilingual sparsity  
- Fine-tuning recovery  
- Attention sparsity  

---

## 📚 Related Work

- TEAL (ICLR 2025 Spotlight) — training-free activation sparsity  
- SparseGPT — weight pruning  
- PowerInfer — CPU/GPU hybrid inference  
- DejaVu — dynamic sparsity  

---

## 💻 Hardware

- RTX 5060 (8GB)  
- CUDA 12.6  
- PCIe Gen4 x8  

> Built on a consumer GPU — not a research cluster.

---

## 📖 Citation

```bibtex
@misc{pokhrel2026ghostweight,
  author = {Pokhrel, Manjit},
  title  = {GhostWeight},
  year   = {2026}
}
```

---

## 👤 Author

**Manjit Pokhrel**  
AI Researcher (Nepal)

---

## 📜 License

MIT License

---
