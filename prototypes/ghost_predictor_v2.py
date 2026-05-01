import os
import torch
import numpy as np
import torch.nn as nn
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

class GhostPredictor(nn.Module):
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
            nn.Linear(bottleneck, intermediate_dim),
            nn.Sigmoid()
        )

    def forward(self, hidden_state):
        if hidden_state.dim() == 3:
            hidden_state = hidden_state[:, -1, :]
        return self.predictor(hidden_state)

print("=" * 60)
print("GhostWeight Phase 2 — Ghost Predictor v2")
print("Weighted Loss + Better Architecture")
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

for name, module in model.named_modules():
    if isinstance(module, SiLUActivation):
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], GhostGate(threshold=0.25))

print("GhostGate applied.")

# ============================================================
# Data Collection
# ============================================================

training_data = []
hidden_states_buffer = {}
activation_masks_buffer = {}

def make_hidden_hook(layer_idx):
    def hook_fn(module, input, output):
        with torch.no_grad():
            hidden = input[0].detach().cpu().float()
            hidden_states_buffer[layer_idx] = hidden
    return hook_fn

def make_mask_hook(layer_idx):
    def hook_fn(module, input, output):
        with torch.no_grad():
            mask = (output != 0.0).float().detach().cpu()
            activation_masks_buffer[layer_idx] = mask
    return hook_fn

for name, module in model.named_modules():
    if "model.layers." in name and name.count(".") == 2:
        try:
            layer_idx = int(name.split(".")[2])
            module.register_forward_hook(make_hidden_hook(layer_idx))
        except (ValueError, IndexError):
            pass
    if isinstance(module, GhostGate):
        try:
            layer_idx = int(name.split(".")[2])
            module.register_forward_hook(make_mask_hook(layer_idx))
        except (ValueError, IndexError):
            pass

training_prompts = [
    "The capital of France is",
    "2 + 2 equals",
    "Water is made of",
    "Einstein developed the theory of",
    "To reverse a string in Python",
    "The speed of light is approximately",
    "Neural networks learn by",
    "The largest planet in our solar system is",
    "Gravity pulls objects towards",
    "The human body has approximately",
    "Quantum mechanics describes the behavior of",
    "The mitochondria is the powerhouse",
    "To train a neural network you need",
    "The French Revolution began in",
    "Black holes are formed when",
    "The Pythagorean theorem states that",
    "DNA replication occurs when",
    "Machine learning differs from deep learning because",
    "The derivative of sin(x) is",
    "To implement a linked list in Python",
    "The Big Bang theory suggests",
    "Photosynthesis is the process by which",
    "The law of conservation of energy states",
    "In supervised learning the model learns from",
    "The attention mechanism in transformers works by",
    "Gradient descent minimizes the loss by",
    "Backpropagation computes gradients by",
    "The softmax function converts",
    "Regularization in machine learning prevents",
    "Transfer learning allows models to",
]

print(f"\nCollecting training data from {len(training_prompts)} prompts...")

for i, prompt in enumerate(training_prompts):
    hidden_states_buffer.clear()
    activation_masks_buffer.clear()

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        _ = model(**inputs)

    for layer_idx in hidden_states_buffer:
        if layer_idx in activation_masks_buffer:
            h = hidden_states_buffer[layer_idx]
            m = activation_masks_buffer[layer_idx]
            h_last = h[0, -1, :]
            m_last = m[0, -1, :]
            training_data.append((h_last, m_last))

    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(training_prompts)} prompts | {len(training_data)} pairs")

print(f"\nTotal pairs: {len(training_data)}")

X = torch.stack([p[0] for p in training_data]).float()
Y = torch.stack([p[1] for p in training_data]).float()

print(f"X: {X.shape} | Y: {Y.shape}")

# ============================================================
# Class Imbalance Analysis
# ============================================================

active_ratio = Y.mean().item()
dead_ratio = 1.0 - active_ratio
pos_weight_value = dead_ratio / active_ratio

print(f"\nClass Balance:")
print(f"  Active neurons: {active_ratio*100:.1f}%")
print(f"  Dead neurons:   {dead_ratio*100:.1f}%")
print(f"  Pos weight:     {pos_weight_value:.1f}x")

hidden_dim = X.shape[1]
intermediate_dim = Y.shape[1]

# ============================================================
# Train With Weighted BCE Loss
# ============================================================

predictor = GhostPredictor(
    hidden_dim=hidden_dim,
    intermediate_dim=intermediate_dim,
    bottleneck=256
).cuda()

param_count = sum(p.numel() for p in predictor.parameters())
print(f"Predictor params: {param_count:,} ({param_count*4/1e6:.1f} MB)")

optimizer = torch.optim.AdamW(predictor.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# Weighted loss: penalize missing active neurons heavily
pos_weight = torch.tensor([pos_weight_value]).cuda()
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Remove sigmoid from predictor for BCEWithLogitsLoss
class GhostPredictorLogits(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim, bottleneck=256):
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
            # No sigmoid here — BCEWithLogitsLoss handles it
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, -1, :]
        return self.predictor(x)

    def predict_mask(self, x, threshold=0.5):
        with torch.no_grad():
            logits = self.forward(x)
            return (torch.sigmoid(logits) > threshold).float()

predictor = GhostPredictorLogits(
    hidden_dim=hidden_dim,
    intermediate_dim=intermediate_dim,
    bottleneck=256
).cuda()

optimizer = torch.optim.AdamW(predictor.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

dataset = torch.utils.data.TensorDataset(X, Y)
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

print(f"\nTraining with weighted loss (pos_weight={pos_weight_value:.1f}x)...")
print("-" * 60)

epochs = 30
for epoch in range(epochs):
    predictor.train()
    total_loss = 0
    total_recall = 0
    batches = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()

        optimizer.zero_grad()
        logits = predictor(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            pred_mask = (torch.sigmoid(logits) > 0.5).float()
            true_active = y_batch == 1.0
            if true_active.sum() > 0:
                recall = (pred_mask[true_active] == 1.0).float().mean()
                total_recall += recall.item()

        total_loss += loss.item()
        batches += 1

    scheduler.step()

    if (epoch + 1) % 5 == 0:
        avg_loss = total_loss / batches
        avg_recall = total_recall / batches
        print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | Recall: {avg_recall*100:.2f}%")

# ============================================================
# Final Evaluation
# ============================================================

predictor.eval()
with torch.no_grad():
    X_gpu = X.cuda()
    Y_gpu = Y.cuda()

    logits = predictor(X_gpu)
    pred_masks = (torch.sigmoid(logits) > 0.5).float()

    overall_acc = (pred_masks == Y_gpu).float().mean()

    true_active = Y_gpu == 1.0
    true_dead = Y_gpu == 0.0

    recall = (pred_masks[true_active] == 1.0).float().mean()
    miss_rate = (pred_masks[true_active] == 0.0).float().mean()
    false_fetch = (pred_masks[true_dead] == 1.0).float().mean()

    # VRAM savings calculation
    predicted_active = pred_masks.mean()
    vram_reduction = 1.0 - predicted_active.item()

print("\n" + "=" * 60)
print("GHOST PREDICTOR v2 RESULTS")
print("=" * 60)
print(f"Overall Accuracy:      {overall_acc*100:.2f}%")
print(f"Active Neuron Recall:  {recall*100:.2f}%")
print(f"Miss Rate:             {miss_rate*100:.2f}%")
print(f"False Fetch Rate:      {false_fetch*100:.2f}%")
print(f"Predicted Active:      {predicted_active*100:.2f}%")
print(f"VRAM Reduction:        {vram_reduction*100:.2f}%")
print(f"Predictor Size:        {sum(p.numel() for p in predictor.parameters())*4/1e6:.1f} MB")
print("=" * 60)

if recall >= 0.90:
    print("STATUS: EXCELLENT ✅ — Ready for Blackwell Loader")
elif recall >= 0.75:
    print("STATUS: GOOD ✅ — Viable for streaming")
elif recall >= 0.60:
    print("STATUS: ACCEPTABLE — Minor quality tradeoff")
else:
    print("STATUS: NEEDS WORK — Increase training data")
print("=" * 60)

os.makedirs("F:/GhostWeight/models", exist_ok=True)
torch.save(predictor.state_dict(), "F:/GhostWeight/models/ghost_predictor_v2.pt")
print("Saved: F:/GhostWeight/models/ghost_predictor_v2.pt")
print("=" * 60)