# Voice_Cloning — OmniVoice Hindi Voice Assistant (राहुल)

A realtime, **echo-free**, voice-cloning Hindi voice assistant built on
[k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) (~613M params, 600+
languages). It clones a fixed reference voice once at startup, then answers
chat turns by speaking in that voice.

- **TTS / voice cloning:** OmniVoice, fp16 on CUDA (~1.3 GB VRAM — fits any
  8 GB NVIDIA GPU), diffusion `num_step` as the quality/speed dial.
- **Conversation brain:** any OpenAI-compatible chat LLM (default
  `ministral-8b-latest` via Mistral), streamed sentence-by-sentence.
- **Voice input — OpenAI/Gemini-style, echo-free:** the browser streams the
  **AEC-processed** `getUserMedia` mic (16 kHz) to a local
  [faster-whisper](https://github.com/SYSTRAN/faster-whisper) server
  (`/ws/asr`). Chrome's acoustic echo canceller subtracts the assistant's own
  speaker output *before* the audio is transcribed, so the assistant can never
  "hear" itself. *(No Web Speech API — its separate capture path has no echo
  reference, which caused self-conversation loops.)*
- **Web UI:** React/Vite/Tailwind panel (mic, live captions, chat, barge-in).

```
Voice_Cloning/
├── voice_api.py          # FastAPI server: TTS WS, chat, streaming ASR, static UI
├── main.py               # single-shot CLI benchmark (prints RTF)
├── clean_voice.py        # denoise/normalize my_voice.wav
├── trim_voice.py         # trim reference to ~5 s of speech
├── start.sh              # macOS helper: create venv + run server + open UI
├── requirements.txt
├── .env.example          # every tunable, documented (copy to .env)
├── web/ui/               # React frontend (build with npm)
└── kaggle/               # Kaggle / cloud-GPU helpers (no .env needed)
```

---

## 1. Hardware & device selection

The code auto-selects the device/dtype at startup (override in `.env`):

| Host                    | device | dtype |
| ----------------------- | ------ | ----- |
| NVIDIA GPU (8 GB is plenty) | `cuda` | `fp16` |
| Apple Silicon (M-series)    | `mps`  | `fp32` |
| CPU-only                    | `cpu`  | `fp32` |

Quick CUDA sanity check on the target machine:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> ⚠️ **Never copy `omnivoice-env/` from a Mac to an NVIDIA machine.** The venv
> holds a CPU/MPS torch build; torch silently falls back to CPU (very slow).
> Create the venv fresh on the GPU machine so pip installs the CUDA wheel.

## 2. Setup (local)

```bash
cd Voice_Cloning

# 1) Python env (uv recommended; or python3.11 -m venv omnivoice-env)
uv venv --python 3.11 omnivoice-env
uv pip install -r requirements.txt

# 2) Optional: web UI needs node once (only if you want the chat panel)
cd web/ui && npm install && npm run build && cd ../..

# 3) Config — copy and edit (reference voice path/text, LLM key, tuning)
cp .env.example .env
#   - MISTRAL_API_KEY=...        (required for chat; or export the env var)
#   - VOICE_REF_AUDIO=my_voice.wav
#   - VOICE_REF_TEXT=<EXACT transcript of the clip>
```

> Inline comments in `.env` are supported: `KEY=value  # note` is parsed as
> `value`.

### Reference voice

- `my_voice.wav` must be a **3–10 s clean mono clip** (ideally with the exact
  transcript in `VOICE_REF_TEXT`). Run `clean_voice.py` (denoise) and/or
  `trim_voice.py` (auto-trim to ~5 s) to prepare it.
- It is git-ignored (`*.wav`) — drop your own after cloning.

### Run

```bash
# CLI benchmark — prints inference time, audio length and RTF
./omnivoice-env/bin/python main.py "अरे वाह! आपने तो बहुत अच्छा सवाल पूछा।"

# Full server (UI at http://127.0.0.1:8000) — macOS helper:
./start.sh
# …or manually:
./omnivoice-env/bin/python voice_api.py
```

**First run downloads models:** OmniVoice weights and (on first mic use) the
faster-whisper model (`ASR_MODEL`, default `small` ≈ 460 MB) from Hugging Face.

## 3. Quality ↔ speed tuning

The main dial is diffusion `num_step` (`VOICE_NUM_STEP`):

- **32** — highest quality (official default) — roughly real-time on a
  4060/3060-class card at fp16
- **16** — default here; ~2–4× real-time on an 8 GB NVIDIA card
- **8–12** — fastest, used for the *first* audio window (`VOICE_FIRST_STEP=6`)
  so a reply starts almost instantly

Run `main.py` at a few values on your GPU and pick the highest `num_step`
whose RTF feels comfortable — that is your real quality/speed sweet spot.
Every knob is documented in `.env.example` (chat windows, VAD, pauses, pacing,
ASR cadence, …) and is served to the UI by `/api/config`.

**Known upstream issue:** OmniVoice leaks GPU memory across repeated
`generate()` calls (`k2-fsa/OmniVoice#199`). This repo applies the upstream
workaround — `torch.cuda.empty_cache()` around every call plus pulling outputs
to CPU — so long-running servers stay flat.

## 4. Voice input (mic) — requirements & tuning

- Works in **Chrome / Edge** only (AudioWorklet + WebRTC AEC).
- The assistant's audio and the mic must share the browser's audio context
  (they do). **Headphones or a lower speaker volume** help Chrome's AEC lock.
- First mic use initializes the ASR backend (one-time load + model download;
  ~460 MB for `small`). **Keep `ASR_MODEL=small` and `ASR_LANG=hi`** — `small`
  is the smallest size that transcribes Hindi correctly (`base`/`tiny` mangle
  it), and forcing `hi` is both faster and accurate for Hindi/Hinglish.
- **Switchable backend (`ASR_BACKEND`)** — `auto` (default) uses **mlx-whisper
  on Apple Silicon** (Neural Engine, ~10x faster than CPU Whisper on an M4;
  install: `uv pip install -p omnivoice-env/bin/python mlx-whisper`) and
  **faster-whisper everywhere else** — CUDA fp16 on Kaggle/NVIDIA, CPU int8 on
  other machines. The same repo therefore runs fast on the M4 AND on Kaggle's
  Linux GPU with no edits. `ASR_BACKEND=mlx` or `=faster-whisper` forces one.
  Note mlx-whisper has no beam decoder; `ASR_FINAL_BEAM` applies to the
  faster-whisper backend only.
- Want true multilingual STT? Set `ASR_LANG=` (empty = auto-detect per
  utterance). Auto-detect is great for English but routinely mislabels SHORT
  Hindi clips (as es/ru/ur/si) — that is why Hindi seemed broken.

```ini
# .env — voice input
ASR_BACKEND=auto      # auto | mlx (Apple Silicon only) | faster-whisper
ASR_MODEL=small       # tiny | base | small (default) | medium
ASR_LANG=hi           # hi = Hindi/Hinglish (fast, accurate); empty = auto-detect any language
ASR_DEVICE=           # auto: cuda on NVIDIA, cpu elsewhere (fw backend)
ASR_COMPUTE=          # auto: float16 (CUDA) / int8 (CPU) (fw backend)
ASR_FINAL_BEAM=5      # fw only: beam width for the final transcript
VOICE_ASR_PARTIAL_NEW_SECONDS=0.45   # live-caption cadence
VOICE_ASR_PARTIAL_GAP_SECONDS=0.9
VOICE_ASR_SPECULATIVE=1              # 1 = fast greedy transcript fires the reply before the beam final (lower latency)
VOICE_SPECULATIVE_MS=5000            # how long the client waits for the final before releasing the mic anyway
VOICE_AUTO_SEND_MS=280               # end-of-speech silence tail before the utterance is sent (lower = snappier turns)
```

## 5. Testing on a GPU cloud / Kaggle

Kaggle notebooks give a free NVIDIA GPU (T4/P100, 16 GB) — great for
benchmarking the CUDA path. Kaggle has no microphone or browser, so test
**TTS generation + speed** there (not the live mic chat).

Because a Kaggle notebook does **not** ship your `.env`, use the bundled
helpers which set every knob inside the process:

1. Push this repo to GitHub (public) or upload a zip as a Kaggle **Dataset**.
2. Create a Notebook → `File ▸ Add input` (the repo / dataset) or clone:
   ```bash
   !git clone https://github.com/<you>/<repo>.git
   %cd /kaggle/working/Voice_Cloning
   ```
3. Install + verify CUDA:
   ```bash
   !python kaggle/setup_kaggle.sh
   ```
4. Benchmark several `num_step` values and hear the outputs:
   ```bash
   !python kaggle/benchmark_kaggle.py
   ```
   (`kaggle/Voice_Cloning_Kaggle.ipynb` is a ready-made notebook with these
   steps — upload it and run all cells.)

Notes:
- `requirements.txt` leaves the preinstalled Kaggle torch untouched.
- `MISTRAL_API_KEY` (for chat replies) can be injected via Kaggle
  **Secrets** (`Add-ons ▸ Secrets`) or `os.environ` in a cell — the ASR/TTS
  benchmarks above don't need it.
- Outputs are saved as `output_step<num>.wav` next to `main.py`; listen inline
  with `IPython.display.Audio` or download them.

## 6. Troubleshooting

| Symptom | Fix |
| ------- | --- |
| `torch.cuda.is_available()` is `False` on the GPU box | venv was copied from another machine; recreate it there (`uv pip install -r requirements.txt`) |
| Voice input does nothing | Chrome/Edge only; check `/api/config` shows `"asr_model"`; watch the server log for the whisper load line; allow a minute on first use |
| Assistant still hears itself | Use headphones / lower volume (AEC needs a clear echo reference); confirm the reply is playing through the same tab's speakers |
| No web UI after `git clone` | `web/ui/dist` is git-ignored — run `cd web/ui && npm install && npm run build` |
| Chat says "MISTRAL_API_KEY not configured" | Add the key to `.env` (or export it) and restart the server |
| Generated speech sounds flat/robotic | Raise `VOICE_NUM_STEP`, and/or re-record a cleaner `my_voice.wav` with clear pauses |

## License & ethics

OmniVoice is Apache-2.0. Voice cloning can be misused: only clone voices you
own or have permission to use, and comply with local law.
