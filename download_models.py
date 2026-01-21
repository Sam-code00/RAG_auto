# Code to download model if not found offline
from transformers import CLIPProcessor, CLIPModel

model_name = "openai/clip-vit-base-patch32"

print(f"Downloading/Loading {model_name} to cache...")
try:
    CLIPModel.from_pretrained(model_name)
    CLIPProcessor.from_pretrained(model_name)
    print("Successfully cached CLIP model.")
except Exception as e:
    print(f"Failed to download CLIP model: {e}")
    print("Please ensure you have an active internet connection for this one-time setup step.")
