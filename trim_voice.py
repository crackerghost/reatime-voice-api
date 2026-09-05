"""Trim my_voice.wav to the strongest ~5s of speech and clean it.

Finds a real silence gap near TARGET_SECONDS and cuts there, so the kept
segment ends on a complete phrase/sentence (never mid-word). The rest of the
pipeline is identical to clean_voice.py: high-pass, spectral-gate denoise,
edge silence trim, normalize to -1 dBFS.

The raw take is backed up to my_voice.raw8s.wav first (my_voice.original.wav
holds the *old* voice and is left untouched).

Run:
    cd Voice_Cloning && ./omnivoice-env/bin/python trim_voice.py
"""

import shutil
from pathlib import Path

import noisereduce as nr
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

HERE = Path(__file__).resolve().parent
SRC = HERE / "my_voice.wav"
BAK = HERE / "my_voice.raw8s.wav"

TARGET = 5.0        # aim for ~this many seconds of clean audio
MIN_GAP = 0.25      # a pause counts as a cut point if >= this long (s)
SEARCH = (3.0, 6.5) # only look for pauses inside this window (s)

FRAME = 0.02
MARGIN = 0.06
HIGHPASS_HZ = 60.0
GATE_DB = -50.0
PROP_DECREASE = 0.85


def to_mono(x):
    return x if x.ndim == 1 else x.mean(axis=1)


def find_cut(x: np.ndarray, sr: int) -> float:
    """Return the best cut time (s): the silence gap closest to TARGET."""
    hop = int(0.05 * sr)
    n = len(x) // hop
    rms = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
    dt = hop / sr
    sil = rms < np.quantile(rms, 0.25)
    best, best_score = None, float("inf")
    i = 0
    while i < n:
        if sil[i]:
            j = i
            while j < n and sil[j]:
                j += 1
            start, end = i * dt, j * dt
            if (end - start) >= MIN_GAP and SEARCH[0] <= start <= SEARCH[1]:
                cut = min(end - 0.05, TARGET + 1.0)  # just before speech resumes
                score = abs(cut - TARGET)
                if score < best_score:
                    best, best_score = cut, score
            i = j
        else:
            i += 1
    return best if best is not None else TARGET  # hard cut fallback


def trim_silence(x: np.ndarray, sr: int) -> np.ndarray:
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
    return x[max(0, above[0] - margin) * hop : min(len(x), (above[-1] + 1 + margin) * hop)]


def estimate_noise(x: np.ndarray, sr: int) -> np.ndarray:
    hop = int(FRAME * sr)
    n = len(x) // hop
    if n == 0:
        return np.zeros(int(0.5 * sr))
    frames = x[: n * hop].reshape(n, hop)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    floor = float(np.quantile(rms, 0.1))
    thr = max(floor * 3, 10 ** (GATE_DB / 20))
    quiet = frames[rms < thr]
    noise = quiet[: int(1.0 / FRAME)].reshape(-1)
    if len(noise) < int(0.25 * sr):
        rng = np.random.default_rng(0)
        noise = rng.normal(0.0, max(floor, 1e-6), int(1.0 * sr))
    return noise


def main() -> None:
    data, sr = sf.read(str(SRC))
    x = to_mono(data).astype(np.float64)
    print(f"before: {len(x) / sr:.2f}s  peak={np.abs(x).max():.3f}")

    cut = find_cut(x, sr)
    x = x[: int(cut * sr)]
    print(f"cut at {cut:.2f}s (pause near {TARGET:.0f}s) -> {len(x) / sr:.2f}s kept")

    shutil.copy2(SRC, BAK)
    print(f"📦 raw take backed up to {BAK.name}")

    x = trim_silence(x, sr)
    sos = butter(2, HIGHPASS_HZ, btype="highpass", fs=sr, output="sos")
    x = sosfiltfilt(sos, x)
    noise = estimate_noise(x, sr)
    x = nr.reduce_noise(y=x, sr=sr, y_noise=noise, stationary=True,
                        prop_decrease=PROP_DECREASE, n_fft=512, use_tqdm=False)
    x = trim_silence(x, sr)
    peak = np.abs(x).max()
    if peak > 0:
        x = x * (0.891 / peak)  # -1 dBFS
    sf.write(str(SRC), x.astype(np.float32), sr, subtype="PCM_16")
    print(f"✅ wrote {SRC.name}: {len(x) / sr:.2f}s @ {sr} Hz, peak {np.abs(x).max():.3f}")


if __name__ == "__main__":
    main()
