import torch
from huggingface_hub import hf_hub_download
import os

# Download the checkpoint
model_path = hf_hub_download(
    repo_id="Arko007/deepfake-image-detector",
    filename="pytorch_model.bin",
    token=os.getenv("HUGGINGFACE_TOKEN", "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")
)

# Load and inspect
checkpoint = torch.load(model_path, map_location='cpu')

print("="*60)
print("CHECKPOINT STRUCTURE")
print("="*60)

for key in sorted(checkpoint.keys()):
    shape = checkpoint[key].shape if hasattr(checkpoint[key], 'shape') else 'N/A'
    print(f"{key}: {shape}")

print("\n" + "="*60)
print("CLASSIFIER LAYERS")
print("="*60)

classifier_layers = {k: v.shape for k, v in checkpoint.items() if 'classifier' in k}
for k, v in sorted(classifier_layers.items()):
    print(f"{k}: {v}")
