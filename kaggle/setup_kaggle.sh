#!/usr/bin/env bash
# Kaggle / cloud-GPU bootstrap for Voice_Cloning — no .env needed.
# Run from anywhere inside the repo (it cd's to the repo root):
#     !python kaggle/setup_kaggle.sh
set -e
cd "$(dirname "$0")/.."

echo "== Python =="
python --version

echo "== PyTorch / CUDA =="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("vram_gb:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
else:
    print("gpu: none (CPU only)")
PY

echo "== Installing requirements (preinstalled torch is left untouched) =="
python -m pip install -q -r requirements.txt

echo "== Reference voice =="
if [ ! -f my_voice.wav ]; then
    echo "WARNING: my_voice.wav is missing (it is git-ignored)."
    echo "Upload a clean 3-10 s clip as my_voice.wav, or set VOICE_REF_AUDIO."
fi

echo
echo "Ready. Next step:"
echo "    python kaggle/benchmark_kaggle.py"
