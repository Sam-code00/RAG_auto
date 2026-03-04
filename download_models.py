import os
import urllib.request
import zipfile
from pathlib import Path
from transformers import AutoModel, AutoProcessor

# SigLIP2 Model
model_name = "google/siglip2-base-patch16-224"

print(f"Downloading/Loading {model_name} to cache...")
try:
    AutoModel.from_pretrained(model_name)
    AutoProcessor.from_pretrained(model_name)
    print("Successfully cached SigLIP2 model.")
except Exception as e:
    print(f"Failed to download SigLIP2 model: {e}")
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