"""Benchmark OmniVoice across num_step values on Kaggle / cloud GPUs.

Kaggle notebooks do not get your local .env, so this script injects every
knob directly into the environment of each main.py run (main.py reads env at
import time; defaults match .env.example otherwise).

Run from the Voice_Cloning repo root:
    !python kaggle/benchmark_kaggle.py

Tunables (set as Kaggle env vars or edit below):
    BENCH_TEXT   - text to synthesize (default a Hindi demo sentence)
    BENCH_STEPS  - comma-separated num_step values (default "16,24,32")
    VOICE_REF_AUDIO / VOICE_REF_TEXT - reference clip + its exact transcript
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT = os.environ.get("BENCH_TEXT", "अरे वाह! आपने तो बहुत अच्छा सवाल पूछा।")
STEPS = [int(x) for x in os.environ.get("BENCH_STEPS", "16,24,32").split(",") if x.strip()]
REF_AUDIO = os.environ.get("VOICE_REF_AUDIO", os.path.join(ROOT, "my_voice.wav"))
REF_TEXT = os.environ.get(
    "VOICE_REF_TEXT",
    "कोडिंग में बहुत मज़ा आता है, बट समटाइम्स बग्स आर सो अनोइंग यार।",
)

# main.py prints, e.g.:  Real-Time Factor (RTF): 0.2431
RTF_RE = re.compile(r"Real-Time Factor \(RTF\): ([\d.]+)")


def cuda_summary() -> str:
    try:
        import torch
    except Exception:
        return "torch unavailable"
    if not torch.cuda.is_available():
        return "CUDA unavailable — this will run on CPU (slow)!"
    name = torch.cuda.get_device_name(0)
    gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    return f"{name} ({gb:.0f} GB)"


def main() -> int:
    print("=" * 64)
    print("Voice_Cloning benchmark (k2-fsa/OmniVoice)")
    print("GPU   :", cuda_summary())
    print("Text  :", TEXT[:60])
    print("Steps :", STEPS)
    print("=" * 64)
    if not os.path.exists(REF_AUDIO):
        print(f"\nERROR: reference audio not found: {REF_AUDIO}\n"
              "Upload a clean 3-10 s clip as my_voice.wav (or set VOICE_REF_AUDIO).")
        return 1

    results = []
    for step in STEPS:
        out_wav = os.path.join(ROOT, f"output_step{step}.wav")
        env = dict(os.environ)
        env.update({
            "VOICE_NUM_STEP": str(step),
            "VOICE_FIRST_STEP": "8",
            "VOICE_REF_AUDIO": REF_AUDIO,
            "VOICE_REF_TEXT": REF_TEXT,
            "VOICE_OUTPUT": out_wav,
            # Kaggle GPUs are CUDA: fp16 is the auto default, keep explicit.
            "VOICE_API_DEVICE": os.environ.get("VOICE_API_DEVICE", "cuda"),
            "VOICE_API_DTYPE": os.environ.get("VOICE_API_DTYPE", "fp16"),
        })
        print(f"\n>>> num_step = {step}")
        proc = subprocess.run(
            [sys.executable, os.path.join(ROOT, "main.py"), TEXT],
            cwd=ROOT, env=env, capture_output=True, text=True,
        )
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        m = RTF_RE.search(proc.stdout)
        results.append((step, float(m.group(1)) if m else None))

    print("\n" + "=" * 64)
    print("SUMMARY")
    for step, rtf in results:
        if rtf is None:
            print(f"  num_step={step:>3}  (no RTF parsed — run failed?)")
        else:
            print(f"  num_step={step:>3}  RTF={rtf:.3f}  "
                  f"({1 / rtf:.1f}x faster than real-time)")
    print("-" * 64)
    print("Pick the highest num_step whose RTF < ~1 (still faster than "
          "real-time);")
    print("that is your quality/speed sweet spot on this GPU.")
    print("Listen in the notebook, e.g.:")
    print("  from IPython.display import Audio")
    print("  Audio('output_step16.wav')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
