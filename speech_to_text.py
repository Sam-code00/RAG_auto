"""
Speech-to-Text Module using Vosk
Click to start, click to stop recording
"""

import os
import json
import queue
import threading
from pathlib import Path

import sounddevice as sd
from vosk import Model, KaldiRecognizer

SAMPLE_RATE = 16000
MODEL_PATH = Path("models/vosk-model-small-en-us-0.15")


class Recorder:
    """Singleton recorder that persists across Streamlit reruns"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.model = None
        self.recognizer = None
        self.stream = None
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.transcript = ""
        self._initialized = True
    
    def load_model(self):
        if self.model is None:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Vosk model not found at {MODEL_PATH}. "
                    "Run 'python download_models.py' first."
                )
            self.model = Model(str(MODEL_PATH))
        return self.model
    
    def _audio_callback(self, indata, frames, time, status):
        if self.is_recording:
            self.audio_queue.put(bytes(indata))
    
    def start(self):
        if self.is_recording:
            return
        
        self.load_model()
        
        # Clear state
        self.transcript = ""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                pass
        
        # New recognizer
        self.recognizer = KaldiRecognizer(self.model, SAMPLE_RATE)
        self.recognizer.SetWords(True)
        
        # Start stream
        self.is_recording = True
        self.stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=8000,
            dtype='int16',
            channels=1,
            callback=self._audio_callback
        )
        self.stream.start()
    
    def stop(self):
        if not self.is_recording:
            return self.transcript
        
        self.is_recording = False
        
        # Stop stream
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        # Process all audio
        transcript = ""
        while not self.audio_queue.empty():
            try:
                data = self.audio_queue.get_nowait()
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "")
                    if text:
                        transcript += " " + text if transcript else text
            except:
                break
        
        # Final result
        if self.recognizer:
            final = json.loads(self.recognizer.FinalResult())
            text = final.get("text", "")
            if text:
                transcript += " " + text if transcript else text
        
        self.transcript = transcript.strip()
        return self.transcript


# Global singleton
_recorder = None


def get_recorder():
    global _recorder
    if _recorder is None:
        _recorder = Recorder()
    return _recorder


def load_model():
    return get_recorder().load_model()


def start_recording():
    get_recorder().start()


def stop_recording():
    return get_recorder().stop()


def is_recording():
    return get_recorder().is_recording


def is_model_loaded():
    return get_recorder().model is not None