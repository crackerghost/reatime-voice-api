"""OmniVoice single-shot voice-cloning demo in Rahul's voice.

Local port of the official Kaggle realtime demo — same flow:

    OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=..., dtype=...)
    -> create_voice_clone_prompt(my_voice.wav, transcript)   # cached ONCE
    -> generate(text=..., voice_clone_prompt=<cached>, num_step=16, ...)

Run:
    cd Voice_Cloning && ./omnivoice-env/bin/python main.py ["text to speak"]

Device: MPS on Apple Silicon, else CUDA, else CPU (override with VOICE_API_DEVICE
and VOICE_API_DTYPE=fp16|fp32).
"""

import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# huggingface_hub's hf_xet chunked CDN stalls on this network (and cdn-lfs DNS
# is blocked), so force the classic HTTP download path for model weights.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from omnivoice import OmniVoice  # noqa: E402

HERE = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no dependency): KEY=VALUE lines, comments ignored.

    Trailing inline comments are stripped ("K=V  # note" -> "V"), and quoted
    values keep their content until the closing quote.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        raw = value.strip()
        if raw[:1] in ('"', "'"):
            q = raw[0]
            end = raw.find(q, 1)
            value = raw if end < 0 else raw[: end + 1]
        else:
            value = raw.split("#", 1)[0]
        value = value.strip().strip('"').strip("'")
        if key and value:  # blank values are treated as unset -> code defaults apply
            os.environ.setdefault(key, value)


_load_dotenv(HERE / ".env")

# Reference voice (clone source) + its exact transcript
REF_AUDIO = Path(os.environ.get("VOICE_REF_AUDIO", HERE / "my_voice.wav")).expanduser()
if not REF_AUDIO.is_absolute():
    REF_AUDIO = HERE / REF_AUDIO
REF_TEXT = os.environ.get(
    "VOICE_REF_TEXT",
    "कोडिंग में बहुत मज़ा आता है, बट समटाइम्स बग्स आर सो अनोइंग यार।",
)
OUTPUT = Path(os.environ.get("VOICE_OUTPUT", HERE / "output.wav")).expanduser()
if not OUTPUT.is_absolute():
    OUTPUT = HERE / OUTPUT
DEFAULT_TEXT = os.environ.get("VOICE_DEMO_TEXT", "अरे वाह! आपने तो बहुत अच्छा सवाल पूछा।")

MODEL_NAME = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
SAMPLE_RATE = int(os.environ.get("VOICE_SAMPLE_RATE", "24000"))  # OmniVoice always outputs 24 kHz
NUM_STEP = int(os.environ.get("VOICE_NUM_STEP", "16"))           # diffusion steps; lower = faster
TEMPERATURE = float(os.environ.get("VOICE_TEMPERATURE", "0.3"))

DEVICE = (
    os.environ.get("VOICE_API_DEVICE", "").strip().lower()
    or ("mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu")
)
# torch 2.14 segfaults in its MPS fp16 copy/cast kernel while loading weights on
# Apple Silicon ("Python quit unexpectedly") — fp32 is the safe MPS/CPU default;
# fp16 stays the CUDA default. Override with VOICE_API_DTYPE=fp16 at your own risk.
DTYPE_ENV = os.environ.get("VOICE_API_DTYPE", "").strip().lower()
DTYPE = (
    torch.float16 if DTYPE_ENV in ("fp16", "float16")
    else torch.float32 if DTYPE_ENV in ("fp32", "float32")
    else torch.float16 if DEVICE.startswith("cuda")
    else torch.float32
)


def run_optimized_realtime_tts(text_input: str, reference_audio_path: Path,
                               reference_text_string: str, output_audio_path: Path) -> None:
    """Generate speech in Rahul's voice and report the inference benchmark."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("🔄 Loading OmniVoice checkpoint...")
    ov_model = OmniVoice.from_pretrained(MODEL_NAME, device_map=DEVICE, dtype=DTYPE)

    # OPTIMIZATION 1: create + cache the voice profile prompt ONCE
    print("🧠 Extracting and caching your voice tokens...")
    cached_voice_prompt = ov_model.create_voice_clone_prompt(
        ref_audio=str(reference_audio_path),
        ref_text=reference_text_string,
    )

    print(f"\n🗣️ Target Output Text: '{text_input}'")
    print("⏳ Running voice generation...")

    # Start the high-precision inference timer
    start_time = time.perf_counter()

    with torch.no_grad():
        # OPTIMIZATION 2: pass the cached profile and drop num_step to 16
        # omnivoice 0.2.x maps extra kwargs into OmniVoiceGenerationConfig, so
        # the Kaggle demo's `temperature` is passed as `class_temperature`.
        audio_outputs = ov_model.generate(
            text=text_input,
            voice_clone_prompt=cached_voice_prompt,  # bypasses raw audio re-processing
            num_step=NUM_STEP,                       # cuts diffusion workload in half
            class_temperature=TEMPERATURE,           # 0 = greedy; 0.3 = demo default
        )

    end_time = time.perf_counter()
    inference_duration = end_time - start_time

    # generate() -> list of float32 np.ndarray at 24 kHz; merge if long text
    # was auto-chunked into several segments.
    segs = [np.asarray(s, dtype=np.float32) for s in audio_outputs if len(s) > 0]
    if not segs:
        raise SystemExit("OmniVoice returned no audio!")
    audio = segs[0] if len(segs) == 1 else np.concatenate(segs)

    # Save target audio to disk (soundfile — torchaudio 2.11 would need the
    # optional torchcodec extra just to write a wav)
    sf.write(str(output_audio_path), audio, SAMPLE_RATE)

    # Calculate output audio specifications
    generated_audio_duration_seconds = len(audio) / SAMPLE_RATE
    rtf = inference_duration / generated_audio_duration_seconds

    print("\n📊 --- OPTIMIZED BENCHMARK METRICS ---")
    print(f"⏱️ Model Inference Time : {inference_duration:.4f} seconds")
    print(f"🎵 Generated Audio Length: {generated_audio_duration_seconds:.4f} seconds")
    print(f"🚀 Real-Time Factor (RTF): {rtf:.4f}")
    print(f"📈 Processing Speed     : {1 / rtf:.2f}x faster than real-time")
    print(f"💾 Saved to             : {output_audio_path}")
    print("-----------------------------------------\n")

    # VRAM Cleanup
    del ov_model, cached_voice_prompt
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    target_text = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEXT
    if not REF_AUDIO.exists():
        raise SystemExit(f"Reference audio not found: {REF_AUDIO}")

    print(f"Device: {DEVICE} | dtype: {DTYPE}")
    run_optimized_realtime_tts(target_text, REF_AUDIO, REF_TEXT, OUTPUT)
