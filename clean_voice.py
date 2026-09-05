"""Clean background noise out of my_voice.wav (the OmniVoice reference clip).

Pipeline (all safe for a 3-10s voice-clone reference):
  1. Trim leading/trailing silence (keeps 60 ms margins).
  2. 60 Hz high-pass -> removes hum/rumble, keeps the voice intact.
  3. Spectral-gate denoise (noisereduce) using a noise profile measured from the
     quietest frames of the recording.
  4. Re-trim + normalize peak to -1 dBFS.

The first run backs the original up to my_voice.original.wav, then overwrites
my_voice.wav — no code changes needed, the transcript stays the same.

Run:
    cd Voice_Cloning && ./omnivoice-env/bin/python clean_voice.py
"""

import shutil
from pathlib import Path

import noisereduce as nr
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

HERE = Path(__file__).resolve().parent
SRC = HERE / "my_voice.wav"
BAK = HERE / "my_voice.original.wav"

FRAME = 0.02          # 20 ms frames for RMS analysis
MARGIN = 0.06         # silence margin kept after trimming (s)
HIGHPASS_HZ = 60.0    # rumble / mains-hum removal
GATE_DB = -50.0       # below this, a frame counts as silence
PROP_DECREASE = 0.85  # how aggressively the spectral gate removes noise


def to_mono(x):
    return x if x.ndim == 1 else x.mean(axis=1)


def trim_silence(x: np.ndarray, sr: int) -> np.ndarray:
    """Cut leading/trailing silence, keeping a short margin on both sides."""
    hop = int(FRAME * sr)
    n = len(x) // hop
    if n == 0:
        return x
    frames = x[: n * hop].reshape(n, hop)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    thr = max(10 ** (GATE_DB / 20), float(np.quantile(rms, 0.1)) * 4)
    above = np.where(rms > thr)[0]
    if len(above) == 0:
        return x
    margin = int(MARGIN / FRAME)
    start = max(0, above[0] - margin) * hop
    end = min(len(x), (above[-1] + 1 + margin) * hop)
    return x[start:end]


def estimate_noise(x: np.ndarray, sr: int) -> np.ndarray:
    """Build a noise-only clip from the quietest frames (fallback: synthetic)."""
    hop = int(FRAME * sr)
    n = len(x) // hop
    if n == 0:
        return np.zeros(int(0.5 * sr))
    frames = x[: n * hop].reshape(n, hop)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    floor = float(np.quantile(rms, 0.1))
    thr = max(floor * 3, 10 ** (GATE_DB / 20))
    quiet = frames[rms < thr]
    noise = quiet[: int(1.0 / FRAME)].reshape(-1)  # up to ~1 s of noise
    if len(noise) < int(0.25 * sr):                 # not enough real silence
        rng = np.random.default_rng(0)
        noise = rng.normal(0.0, max(floor, 1e-6), int(1.0 * sr))
    return noise


def main() -> None:
    data, sr = sf.read(str(SRC))
    x = to_mono(data).astype(np.float64)
    if not BAK.exists():
        shutil.copy2(SRC, BAK)
        print(f"📦 Original backed up to {BAK.name}")

    def report(tag: str) -> None:
        peak = np.abs(x).max()
        rms = np.sqrt((x ** 2).mean())
        hop = int(FRAME * sr)
        n = len(x) // hop
        fr = x[: n * hop].reshape(n, hop)
        fr_rms = np.sqrt((fr ** 2).mean(axis=1))
        quiet = np.sort(fr_rms)[: max(1, n // 10)].mean()
        print(f"{tag:>9}: {len(x) / sr:5.2f}s  peak={peak:6.3f}  rms={rms:7.4f}  "
              f"noise-floor10%={quiet:7.5f}  (-{20 * np.log10(max(quiet, 1e-9)):5.1f} dBFS)")

    report("before")
    x = trim_silence(x, sr)
    sos = butter(2, HIGHPASS_HZ, btype="highpass", fs=sr, output="sos")
    x = sosfiltfilt(sos, x)
    noise = estimate_noise(x, sr)
    x = nr.reduce_noise(
        y=x, sr=sr, y_noise=noise, stationary=True,
        prop_decrease=PROP_DECREASE, n_fft=512, use_tqdm=False,
    )
    x = trim_silence(x, sr)
    peak = np.abs(x).max()
    if peak > 0:
        x = x * (0.891 / peak)  # normalize to -1 dBFS
    report("after")
    sf.write(str(SRC), x.astype(np.float32), sr, subtype="PCM_16")
    print(f"✅ Wrote cleaned {SRC.name} ({len(x) / sr:.2f}s, {sr} Hz, PCM16)")


if __name__ == "__main__":
    main()
