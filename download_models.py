import os
import urllib.request
import zipfile
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel

# CLIP Model
model_name = "openai/clip-vit-base-patch32"

print(f"Downloading/Loading {model_name} to cache...")
try:
    CLIPModel.from_pretrained(model_name)
    CLIPProcessor.from_pretrained(model_name)
    print("Successfully cached CLIP model.")
except Exception as e:
    print(f"Failed to download CLIP model: {e}")
    print("Please ensure you have an active internet connection for this one-time setup step.")


# Whisper Model
print("\nDownloading/Loading Whisper base model...")
try:
    import whisper
    model = whisper.load_model("base")
    print("Successfully cached Whisper base model.")
except Exception as e:
    print(f"Failed to download Whisper model: {e}")
    print("Install with: pip install openai-whisper")
    print("Please ensure you have an active internet connection for this one-time setup step.")