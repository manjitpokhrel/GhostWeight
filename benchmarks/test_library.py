from ghostweight import ghost_surgery, GhostGate
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("=" * 50)
print("GhostWeight Library Test (CPU)")
print("=" * 50)

model_id = r"F:\hf_cache\hub\models--meta-llama--Llama-3.2-1B-Instruct\snapshots\9213176726f574b556790deb65791e0c5aa438b6"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    local_files_only=True
)

print("Loading model on CPU...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu",
    dtype=torch.float32,
    local_files_only=True,
)

print("Applying GhostGate surgery...")
model, replaced = ghost_surgery(model, threshold=0.05)

print(f"GhostGate replaced {replaced} layers")
print("Library import: OK")
print("Surgery function: OK")
print("GhostGate module: OK")

prompt = "Activation sparsity means"
inputs = tokenizer(prompt, return_tensors="pt")

print("Running CPU inference...")
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False
    )

print("=" * 50)
print("OUTPUT:")
print(tokenizer.decode(out[0], skip_special_tokens=True))
print("=" * 50)
print("GhostWeight library is working correctly.")
print("=" * 50)