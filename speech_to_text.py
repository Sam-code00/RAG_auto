import io
import numpy as np
import whisper

MODEL_NAME = "base"

# Module-level model cache
_model = None


def load_model():
    global _model
    if _model is None:
        print("Loading Whisper model...")
        _model = whisper.load_model(MODEL_NAME)
        print("Whisper model loaded successfully")
    return _model


def is_model_loaded():
    return _model is not None


def transcribe_audio(audio_bytes: bytes) -> str:
    if not audio_bytes:
        print("No audio bytes provided")
        return ""
    
    print(f"Received audio: {len(audio_bytes)} bytes")
    
    model = load_model()
    
    try:
        # Decode WAV bytes directly using scipy (no ffmpeg needed)
        import scipy.io.wavfile as wavfile
        
        # Read WAV from bytes
        audio_buffer = io.BytesIO(audio_bytes)
        sample_rate, audio_data = wavfile.read(audio_buffer)
        
        print(f"Sample rate: {sample_rate}, Audio shape: {audio_data.shape}, Dtype: {audio_data.dtype}")
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        # Convert to float32 and normalize to [-1, 1]
        if audio_data.dtype == np.int16:
            audio_float = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_float = audio_data.astype(np.float32) / 2147483648.0
        elif audio_data.dtype == np.float32:
            audio_float = audio_data
        else:
            audio_float = audio_data.astype(np.float32)
        
        # Resample to 16kHz if needed (Whisper expects 16kHz)
        if sample_rate != 16000:
            print(f"Resampling from {sample_rate}Hz to 16000Hz")
            # Simple resampling using numpy interpolation
            duration = len(audio_float) / sample_rate
            target_length = int(duration * 16000)
            audio_float = np.interp(
                np.linspace(0, len(audio_float), target_length),
                np.arange(len(audio_float)),
                audio_float
            ).astype(np.float32)
        
        print(f"Final audio shape: {audio_float.shape}")
        
        # Transcribe directly from numpy array
        result = model.transcribe(audio_float, fp16=False, language="en")
        text = (result.get("text") or "").strip()
        print(f"Transcription result: '{text}'")
        return text
        
    except Exception as e:
        print(f"Transcription error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return ""