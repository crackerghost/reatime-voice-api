"""OmniVoice model loading and audio generation service."""

import io
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import soundfile as sf
import torch
from omnivoice import OmniVoice

log = logging.getLogger("voice_api")


@dataclass(frozen=True)
class TTSConfig:
    model_name: str
    device: str
    dtype: object
    sample_rate: int
    temperature: float
    default_speed: float
    stream_max_chars: int
    first_window_step: int
    pause_seconds: dict[str, float]


class TTSEngine:
    def __init__(self, config: TTSConfig, pronunciation_fix: Callable[[str], str]):
        self.config = config
        self.pronunciation_fix = pronunciation_fix
        self.generate_lock = threading.Lock()
        self.gpu_warm = {"done": False}
        self.model = None
        self.voice_prompt = None

    def load(self, ref_audio, ref_text):
        self.model = OmniVoice.from_pretrained(
            self.config.model_name,
            device_map=self.config.device,
            dtype=self.config.dtype,
        )
        self.voice_prompt = self.model.create_voice_clone_prompt(
            ref_audio=str(ref_audio),
            ref_text=ref_text,
        )
        log.info("OmniVoice loaded + voice prompt cached from %s", ref_audio.name)
        return self.model, self.voice_prompt

    def warm(self):
        try:
            self.generate("नमस्ते", 2, 1.0)
            self.gpu_warm["done"] = True
            log.info("TTS GPU warm-up complete")
        except Exception as exc:
            log.warning("TTS GPU warm-up failed: %s", exc)

    def generate(self, text, num_step, speed, temperature=None):
        if self.model is None or self.voice_prompt is None:
            raise RuntimeError("TTS engine is not loaded")
        kwargs = {
            "text": self.pronunciation_fix(text),
            "voice_clone_prompt": self.voice_prompt,
            "num_step": num_step,
        }
        if speed is not None:
            kwargs["speed"] = speed
        if temperature is not None:
            kwargs["class_temperature"] = temperature
        if torch.cuda.is_available() and not self.gpu_warm["done"]:
            torch.cuda.empty_cache()
        with self.generate_lock:
            with torch.inference_mode():
                outputs = self.model.generate(**kwargs)
        if not outputs:
            raise RuntimeError("OmniVoice returned no audio")
        segments = []
        for segment in outputs:
            if hasattr(segment, "detach"):
                segment = segment.detach()
            if hasattr(segment, "cpu"):
                segment = segment.cpu()
            segments.append(np.asarray(segment, dtype=np.float32))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return segments[0] if len(segments) == 1 else np.concatenate(segments)

    def wav_bytes(self, samples):
        buffer = io.BytesIO()
        sf.write(buffer, samples, self.config.sample_rate, format="WAV")
        return buffer.getvalue()

    def insert_pauses(self, wav, text):
        total = len(wav)
        if total == 0 or not text:
            return wav
        n_chars = max(len(text), 1)
        parts = []
        start = 0
        for index, char in enumerate(text):
            pause = self.config.pause_seconds.get(char)
            if pause is None:
                continue
            estimate = int(total * (index + 1) / n_chars)
            if estimate <= start or estimate > total:
                continue
            segment = wav[start:estimate]
            if len(segment) > 0:
                parts.append(segment)
            parts.append(np.zeros(int(pause * self.config.sample_rate), dtype=wav.dtype))
            start = estimate
        if start < total:
            parts.append(wav[start:])
        return np.concatenate(parts) if parts else wav

    def stream_chunks(self, text):
        clauses = re.split(r"(?<=[।?!.])\s*", text)
        chunks, current = [], ""
        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue
            if len((current + clause).encode("utf-8")) <= self.config.stream_max_chars:
                current += clause
                continue
            if current:
                chunks.append(current)
                current = ""
            for word in clause.split(" "):
                word = word.strip()
                if not word:
                    continue
                candidate = (current + " " + word).strip() if current else word
                if len(candidate.encode("utf-8")) <= self.config.stream_max_chars:
                    current = candidate
                    continue
                if current:
                    chunks.append(current)
                    current = ""
                buffer = ""
                for char in word:
                    if len((buffer + char).encode("utf-8")) <= self.config.stream_max_chars:
                        buffer += char
                    else:
                        chunks.append(buffer)
                        buffer = char
                current = buffer
        if current:
            chunks.append(current)
        return chunks
