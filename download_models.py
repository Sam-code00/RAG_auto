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


# Vosk Model
vosk_model_name = "vosk-model-small-en-us-0.15"
model_dir = Path("models")
model_path = model_dir / vosk_model_name

print(f"\nDownloading/Loading {vosk_model_name}...")

if model_path.exists():
    print("Successfully cached Vosk model.")
else:
    model_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://alphacephei.com/vosk/models/{vosk_model_name}.zip"
    zip_path = model_dir / f"{vosk_model_name}.zip"
    
    try:
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(model_dir)
        zip_path.unlink()
        print("Successfully cached Vosk model.")
    except Exception as e:
        print(f"Failed to download Vosk model: {e}")
        print("Please ensure you have an active internet connection for this one-time setup step.")