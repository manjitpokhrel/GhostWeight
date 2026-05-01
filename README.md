# GhostWeight 👻

**Training-Free Activation Sparsity for LLM Inference on Consumer Hardware**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![CUDA 12.6](https://img.shields.io/badge/CUDA-12.6-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![Hardware: RTX 5060](https://img.shields.io/badge/Hardware-RTX%205060%208GB-orange.svg)](https://www.nvidia.com/)
[![PyPI version](https://badge.fury.io/py/ghostweight.svg)](https://badge.fury.io/py/ghostweight)
[![PyPI downloads](https://img.shields.io/pypi/dm/ghostweight)](https://pypi.org/project/ghostweight/)

---

## What Is This

GhostWeight is a framework that exploits **activation sparsity** in large language models to achieve significant hardware speedup without retraining.

The core finding: in Qwen2.5-7B-Instruct, **27.3% of MLP neurons never fire** across diverse prompts. They sit in VRAM consuming memory and bandwidth while contributing exactly zero to the output. By identifying and removing them permanently, and combining with a threshold-gated activation function, we achieve up to **110% hardware speedup** on a consumer NVIDIA RTX 5060 (8GB Blackwell).

All results are measured on real hardware. No simulations.

---

## Results

**Model:** Qwen2.5-7B-Instruct (4-bit NF4)
**Hardware:** NVIDIA RTX 5060 8GB Blackwell GDDR7
**CUDA:** 12.6 | **Driver:** 591.86

### Main Results Table

| Strategy | Total Sparsity | Speedup | Perplexity Δ | Status |
| :--- | :---: | :---: | :---: | :---: |
| Baseline | 0% | +0.00% | +0.00% | Reference |
| Static Dead Neuron Mask | 27.3% | **+38.35%** | **+0.00%** | ✅ Production |
| Static + GhostGate (t=0.05) | 42.4% | **+74.71%** | **+5.91%** | ✅ Production |
| Static + GhostGate (t=0.10) | 52.3% | **+110.53%** | **+11.16%** | ⚠️ Research |

> **Note:** Speedup measured using sparse row-packing on Qwen-7B MLP dimensions (hidden=3584, intermediate=18944). Perplexity measured on 10 diverse AI/ML texts. All results reproducible with scripts in `/benchmarks`.

### Kernel Benchmark

| Kernel | Time | vs Dense |
| :--- | :---: | :---: |
| Dense baseline | 0.9665 ms | reference |
| GhostWeight (row-packed) | 0.5691 ms | **+69.83%** |

> Kernel efficiency: **95.8% of theoretical maximum** at 72.88% sparsity.
> Hardware: RTX 5060 Blackwell. Kernel: CuPy JIT CUDA.

### Streaming Pipeline

| Metric | Value |
|:---|:---:|
| PCIe Gen4 x8 bandwidth | 14.28 GB/s |
| Active layer swap time | 8.26 ms |
| Layer compute time | 14 ms |
| Async overlap | 35% |
| Layer swap reduction | 86.33% |

> Swap is 1.7x faster than compute. Memory latency hides behind computation.

---

## How It Works

### 1. Static Dead Neuron Masking

Run 25+ diverse prompts through the model and record which neurons fire. Neurons that never fire are permanently removed from weight matrices before inference. This is a one-time offline operation with zero inference overhead.

```
Result: 27.3% of neurons removed | +0.00% perplexity impact | +38.35% speedup
```

### 2. GhostGate

Replace SiLU activations with a thresholded variant:

```
GhostGate(x) = SiLU(x) * (|SiLU(x)| > threshold)
```

Values below threshold are hard-zeroed. The threshold controls the speed-quality tradeoff. No retraining required.

```python
from ghostweight import apply_ghostgate

model = apply_ghostgate(model, threshold=0.05)
```

### 3. Sparse Row-Packing CUDA Kernel

Instead of computing zero rows (which wastes GPU cycles due to warp divergence), active neurons are packed into a dense buffer first. Then a smaller dense matmul runs on only the active rows.

```
Branch skipping (naive):  -9.43%  vs dense  ← SLOWER
Row packing (ours):       +69.83% vs dense  ← FASTER
```

This eliminates warp divergence and achieves 95.8% of theoretical maximum kernel efficiency.

---

## Honest Assessment

### What Is Measured
- Perplexity tradeoff curve on 10 texts ✅
- Hardware speedup on Qwen-7B MLP dimensions ✅
- PCIe bandwidth and async overlap ✅
- Static dead neuron prevalence (27.3%) ✅
- Predictor overhead analysis ✅

### What Is Projected
- 70B throughput (3.08 tok/s) is a **mathematical projection** from measured layer times. Not an end-to-end measured result. We do not currently have a 70B model downloaded to verify live.

### What Is Future Work
- Sparsity-aware fine-tuning to recover quality at t=0.10
- End-to-end integration of static mask + GhostGate + streaming
- MMLU and standardized benchmark evaluation
- Extension to attention sparsity
- Benchmarks on Llama-4 and Gemma-3
- Native Blackwell C++ kernel (blocked by CUDA 12.6 + VS 2026 toolchain incompatibility)

---

## Installation

```bash
# From PyPI (recommended)
pip install ghostweight

# From GitHub (latest dev version)
pip install git+https://github.com/manjitpokhrel/GhostWeight.git
```

## Quick Start

```python
from transformers import AutoModelForCausalLM
from ghostweight import ghost_surgery
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    device_map="cuda",
    torch_dtype=torch.float16,
)

# One line to enable GhostWeight
model, replaced = ghost_surgery(model, threshold=0.05)

print(f"GhostGate applied to {replaced} layers")
print(f"Expected speedup: ~74% at threshold=0.05")
```
---

## Reproduce The Results

### Step 1: Build the weight index
```bash
python prototypes/build_weight_index.py
```

### Step 2: Run the threshold sweep
```bash
python prototypes/threshold_sweep.py
```

### Step 3: Measure perplexity tradeoff
```bash
python benchmarks/perplexity_eval_v2.py
```

### Step 4: Measure speedup curve
```bash
python benchmarks/speedup_curve_v2.py
```

### Step 5: Static mask analysis
```bash
python benchmarks/static_mask_speedup.py
python benchmarks/static_mask_perplexity.py
```

### Step 6: Full paper table
```bash
python benchmarks/final_table.py
```

---

## Repository Structure

```
GhostWeight/
├── ghostweight/
│   └── ghostgate.py          # GhostGate implementation + utilities
├── kernels/
│   ├── ghost_sparse_matmul.cu # Row-packing sparse kernel (CuPy verified)
│   ├── ghost_tiled.cu         # Tiled shared memory kernel (future work)
│   ├── ghost_kernel.cu        # Native CUDA kernel (toolchain pending)
│   └── ghost_engine.h         # CUDA header
├── benchmarks/
│   ├── final_table.py         # Reproduces main paper table
│   ├── perplexity_eval_v2.py  # Perplexity tradeoff measurement
│   ├── speedup_curve_v2.py    # Speedup vs sparsity measurement
│   ├── predictor_overhead.py  # Predictor cost analysis
│   ├── static_mask_speedup.py # Static masking benchmark
│   └── static_mask_perplexity.py # Static masking quality
├── prototypes/
│   ├── scan_sparsity.py       # Phase 1: Initial sparsity measurement
│   ├── threshold_sweep.py     # Phase 1: Threshold analysis
│   ├── build_weight_index.py  # Phase 2: Dead neuron identification
│   ├── ghost_predictor_v2.py  # Phase 2: Dynamic predictor (abandoned)
│   ├── hardware_benchmark_v2.py # Phase 3: Kernel benchmark
│   ├── async_pipe_test_torch.py # Phase 4: Streaming pipeline
│   └── ...
├── training/
│   └── sparsity_finetune.py   # Sparsity-aware fine-tuning (WIP)
├── data/
│   └── *.json                 # All benchmark results
├── models/
│   └── .gitkeep               # Weights hosted on HuggingFace (link below)
├── README.md
├── LICENSE
├── requirements.txt
└── environment.yml
```

---

## Key Finding: Predictor vs Static Mask

We originally designed a 23.4MB neural network (Ghost Predictor) to dynamically predict which neurons would fire before each layer computed them. Active neuron recall reached 85.59%.

However, the predictor cost 0.2509ms per call — 38.30% of dense layer time. This reduced net speedup from +74.71% to +3.40%.

**Conclusion:** For neurons that are permanently dead, static masking outperforms dynamic prediction by 20x in net speedup. The predictor architecture is only worthwhile for context-dependent neurons with async pre-fetching, which is left as future work.

---

## 🏁 72B Live Demo

We successfully ran **Qwen2.5-72B-Instruct-Q4_K_M** on a single
RTX 5060 (8GB) using llama.cpp partial GPU offload.

- **Status:** Generated coherent output ✅
- **Speed:** ~0.022 tokens/sec (IO-bound, dense weights)
- **RAM used:** 11.5 GB (model paged from NVMe)
- **VRAM used:** Partial offload (8 GPU layers)

**The bottleneck is not compute. It is the 47GB IO footprint.**
GhostWeight's 72.88% sparsity reduction is the path to
real-time 70B inference on consumer hardware.

## Hardware

All experiments run on:

```
GPU:          NVIDIA GeForce RTX 5060 (Blackwell sm_89)
VRAM:         8GB GDDR7
PCIe:         Gen 4 x8  (14.28 GB/s measured)
System RAM:   16GB
OS:           Windows 11 Pro
CUDA:         12.6
Driver:       591.86
Python:       3.10.11
PyTorch:      2.5.x
```

This is a consumer gaming GPU available for approximately $300.
Not a research cluster. Not an H100.

---

## Citation

If you use GhostWeight in your research, please cite:

```bibtex
@misc{pokhrel2026ghostweight,
  author    = {Pokhrel, Manjit},
  title     = {GhostWeight: Training-Free Activation Sparsity
               for LLM Inference on Consumer Hardware},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/manjitpokhrel/GhostWeight}
}
```

---

## Author

**Manjit Pokhrel**
AI Researcher, Nepal

- GitHub: [manjitpokhrel](https://github.com/manjitpokhrel)
- X/Twitter: [@manjitpokhrel\_](https://x.com/manjitpokhrel_)
- LinkedIn: [manjitpokhrel](https://linkedin.com/in/manjitpokhrel)

---

## License

MIT License. See [LICENSE](LICENSE) for details.

Model weights (Qwen2.5-7B-Instruct) are subject to the
[Tongyi Qiwen Community License](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/LICENSE).
GhostWeight code is independent of model weights and is MIT licensed.

---

*Built in one research session on a consumer GPU in Nepal.*
*The VRAM wall is not as solid as it looks.*