"""Realtime Hindi TTS API in Rahul's voice (OmniVoice).

Loads the k2-fsa/OmniVoice checkpoint once at startup, encodes my_voice.wav
into a cached voice-clone prompt, then answers every request by voice cloning
through that prompt — the same flow as the official Kaggle realtime demo
(OmniVoice.from_pretrained -> create_voice_clone_prompt -> generate).

Run:
    cd Voice_Cloning && ./omnivoice-env/bin/python voice_api.py

Example:
    curl -X POST http://127.0.0.1:8000/tts \\
         -H "Content-Type: application/json" \\
         -d '{"text": "आपका दिन शुभ हो।"}' -o speech.wav
"""

import asyncio
import io
import json
import logging
import os
import queue
import re
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

# huggingface_hub's hf_xet chunked CDN stalls on this network (and cdn-lfs DNS
# is blocked), so force the classic HTTP download path for model weights.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import httpx
import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from omnivoice import OmniVoice
from server.schemas import make_schemas
from server.runtime import VoiceRuntime
from server.settings import Settings

from server.llm.diagrams import generate as _generate_diagram, should_generate as _should_generate_diagram

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("voice_api")
# faster-whisper (and its vad/silero deps) log an INFO line per decode
# ("Processing audio with duration 00:01.250") — with streaming ASR that is a
# firehose 2-3x per second while you talk. Our own logger reports the same
# events usefully, so quiet theirs down.
logging.getLogger("faster_whisper").setLevel(logging.WARNING)

HERE = Path(__file__).resolve().parent.parent


# ---------- Persistent HTTP(S) clients (connection reuse) ----------
# Thread-local HTTP clients managed by ProviderClients (server/http_clients.py).
# Reusing connections across turns removes the per-call TLS handshake + connect
# (~50-200 ms) from the reply path.


# ---------- Single-GPU guard: TTS generate() vs VLM generate() ----------
# Shared primitives live in server/gpu.py so app.py AND vision/service.py use
# the SAME lock/counter/event (a second copy would silently break exclusion).
from server.gpu import gpu_generate_lock as GPU_GENERATE_LOCK
from server.gpu import screen_pause_begin as _screen_pause_begin
from server.gpu import screen_pause_end as _screen_pause_end


@asynccontextmanager
async def _screen_paused():
    """Pause background screen warm-ups for the duration of one streaming
    turn (context-manager form — unwinds on exceptions AND client
    disconnects, so the pause can never stick)."""
    _screen_pause_begin()
    try:
        yield
    finally:
        _screen_pause_end()


# True once the boot warm-up JIT'd the OmniVoice kernels — before that the
# pre-generate empty_cache() device sync stays (cold-start safety).
_GPU_WARM = {"done": False}


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


# Everything below is overridable via env vars (or a .env file next to this
# script) — see .env.example for the full documented list.
_load_dotenv(HERE / ".env")
SETTINGS = Settings.from_environment(HERE)

# ---------- Fixed voice: reference clip + its exact transcript ----------
REF_AUDIO = Path(os.environ.get("VOICE_REF_AUDIO", HERE / "my_voice.wav")).expanduser()
if not REF_AUDIO.is_absolute():
    REF_AUDIO = HERE / REF_AUDIO
REF_TEXT = os.environ.get(
    "VOICE_REF_TEXT",
    "कोडिंग में बहुत मज़ा आता है, बट समटाइम्स बग्स आर सो अनोइंग यार।",
)

# ---------- Model: OmniVoice (k2-fsa/OmniVoice), 24 kHz output ----------
OMNIVOICE_MODEL = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
SAMPLE_RATE = int(os.environ.get("VOICE_SAMPLE_RATE", "24000"))  # OmniVoice always outputs 24 kHz
NUM_STEP = int(os.environ.get("VOICE_NUM_STEP", "8"))            # diffusion steps; lower = faster (8 ≈ realtime)
TEMPERATURE = float(os.environ.get("VOICE_TEMPERATURE", "0.3"))  # Kaggle demo default
DEFAULT_SPEED = float(os.environ.get("VOICE_SPEED", "1.0"))      # rate when a request omits speed
# Allowed diffusion-step range and the first-window / greeting caps below are
# all env-tunable (used by /tts, the WebSocket, and TTSRequest validation).
STEP_MIN = int(os.environ.get("VOICE_STEP_MIN", "4"))
STEP_MAX = int(os.environ.get("VOICE_STEP_MAX", "64"))
GREETING_MAX = int(os.environ.get("VOICE_GREETING_MAX", "2"))  # sentences for small-talk replies
FIRST_WINDOW_CHARS = int(os.environ.get("VOICE_FIRST_WINDOW_CHARS", "40"))  # chars in the 1st audio window (T4: 30-45 covers a full clause so the reply starts mid-flow, not mid-word)
JITTER_FRAMES = max(0, int(os.environ.get("VOICE_JITTER_FRAMES", "1")))  # client audio frames buffered before playback
# T4 fluency: language id + per-window edge trims passed to every OmniVoice generate()
TTS_LANGUAGE = os.environ.get("VOICE_TTS_LANGUAGE", "hi").strip().lower() or "hi"
TTS_PAD_S = float(os.environ.get("VOICE_TTS_PAD_S", "0.02"))   # edge padding per streamed window (upstream default 0.1s = dead air each frame)
TTS_FADE_S = float(os.environ.get("VOICE_TTS_FADE_S", "0.02"))  # edge fades per streamed window (same per-window cost)

# MPS (Apple Silicon) / CUDA run best in fp16; CPU falls back to fp32.
# Override with VOICE_API_DEVICE=cpu|mps|cuda and VOICE_API_DTYPE=fp16|fp32.
DEVICE = (
    os.environ.get("VOICE_API_DEVICE", "").strip().lower()
    or ("mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu")
)
# IMPORTANT: torch 2.14 segfaults in its MPS fp16 copy/cast kernel while
# loading weights on Apple Silicon ("Python quit unexpectedly"). fp32 is the
# safe MPS/CPU default; fp16 stays the CUDA default. Override at your own risk
# with VOICE_API_DTYPE=fp16.
DTYPE_ENV = os.environ.get("VOICE_API_DTYPE", "").strip().lower()
DTYPE = (
    torch.float16 if DTYPE_ENV in ("fp16", "float16")
    else torch.float32 if DTYPE_ENV in ("fp32", "float32")
    else torch.float16 if DEVICE.startswith("cuda")
    else torch.float32
)

# Reject ASCII letters: this model speaks Devanagari only
# Greeting/small-talk inputs -> server enforces a short natural reply
GREETING_RE = re.compile(
    r"^(नमस्ते|हेलो|हाय|हैलो|नमस्कार|कैसे\s+हो|क्या\s+हाल|क्या\s+चल|हाय\s+|तुम\s+कौन|तुम्हारा\s+नाम|तुम्हारे\s+बारे|क्या\s+कर\s+सकते|क्या\s+कर\s+सकता|क्या\s+कर\s+सकती|hello|hi|hey|good\s+(morning|afternoon|evening|night)|how\s+are\s+you|who\s+are\s+you|what\s+can\s+you\s+do)",
    re.IGNORECASE,
)

# Sentence enders used to cut the STREAMED LLM reply into clean TTS chunks
# (the LLM decides where sentences break, instead of byte-splitting after the
# fact). The reply is never fully generated before audio starts.
SENT_END_RE = re.compile(r"[।?!.\n]")


def _short_greeting(reply: str, max_sentences: int | None = None) -> str:
    """Cut a greeting reply to the first few sentences (no rambling)."""
    limit = max_sentences if max_sentences is not None else GREETING_MAX
    if limit <= 1:
        return reply
    stops = ["?", "!", "।", "."]
    positions = sorted(p for s in stops if s in reply for p in [reply.find(s)] if p >= 0)
    if not positions:
        return reply
    last = positions[min(limit - 1, len(positions) - 1)]
    return reply[: last + 1].strip()


# ------- expressive pacing (OmniVoice v1 has no real emotion control, so the
# delivery "life" comes from text energy + per-sentence speed variation) -------
_EXCITED_RE = re.compile(r"[!]{1,}|(?:वाह|अरे|ओहो|कमाल|हा\s*हा)")
_DRAMATIC_RE = re.compile(r"\.\.\.|…")
# Multipliers applied when a sentence is excited / dramatic / long, plus the
# length threshold and the clamp range. All env-tunable.
SPEED_EXCITED = float(os.environ.get("VOICE_EXCITED_SPEED", "1.12"))
SPEED_DRAMATIC = float(os.environ.get("VOICE_DRAMATIC_SPEED", "0.92"))
SPEED_LONG = float(os.environ.get("VOICE_LONG_SPEED", "0.96"))
SPEED_LONG_CHARS = int(os.environ.get("VOICE_LONG_CHARS", "110"))
SPEED_MIN = float(os.environ.get("VOICE_SPEED_MIN", "0.3"))
SPEED_MAX = float(os.environ.get("VOICE_SPEED_MAX", "2.0"))


def _pick_speed(sent: str, base: float = 1.0) -> float:
    """Per-sentence pace: excited lines speed up, thoughtful lines slow down.

    Combined with the LLM's punctuation/expression cues ('!', '...', 'वाह!')
    this makes replies sound less flat — OmniVoice v1 clones one prosody from
    the reference clip, so energy must come from text + pacing.
    """
    s = float(base)
    if _EXCITED_RE.search(sent):
        s *= SPEED_EXCITED
    if _DRAMATIC_RE.search(sent):
        s *= SPEED_DRAMATIC
    if len(sent) > SPEED_LONG_CHARS:
        s *= SPEED_LONG  # long sentences a touch slower for clarity
    return max(SPEED_MIN, min(SPEED_MAX, round(s, 3)))


# ---------- LLM: OpenAI-compatible chat endpoint (default Groq, gpt-oss-20b) ----------
# gpt-oss-20b is the fastest model still available on Groq's developer tier
# (1000 tps). llama-3.3-70b-versatile moved to the Enterprise tier, so it now
# returns 404 for developer keys — do not use it as the default.
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-20b")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.6"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "550"))
# gpt-oss models "think" before answering — reasoning tokens count against
# max_tokens, so a small budget can end with EMPTY content (silent no-reply).
# "low" keeps first-audio fast; set LLM_REASONING_EFFORT="" to omit the param.
LLM_REASONING_EFFORT = os.environ.get("LLM_REASONING_EFFORT", "low")
MISTRAL_URL = os.environ.get("MISTRAL_URL", "https://api.groq.com/openai/v1/chat/completions")
LLM_STREAM_TIMEOUT = float(os.environ.get("VOICE_LLM_TIMEOUT", "120.0"))  # httpx stream read timeout (s)
VISION_TIMEOUT = float(os.environ.get("VOICE_VISION_TIMEOUT", "90.0"))
# Extra attempts for TRANSIENT LLM failures (429 rate-limit, 5xx, network
# blips, empty content). 0 = try once. Retries only happen before the first
# sentence is spoken, so a retry can never interrupt a playing reply.
LLM_RETRIES = int(os.environ.get("LLM_RETRIES", "2"))
DIAGRAM_ENABLED = os.environ.get("DIAGRAM_EVENTS", "1") == "1"

from server.llm.prompts import (
    LLM_SYSTEM_PROMPT,
    LLM_SYSTEM_SHORT,
    _LLM_PERSONA_CUSTOM,
    _screen_system_prompt,
    _system_prompt_for,
)


from server.speech.normalization import (
    _devanagari_only,
    _fix_pronunciation,
    _speechify,
)
from server.speech.tts_engine import TTSConfig, TTSEngine


from server.vision import service as vision_service
from server.vision.service import (
    SCREEN_CACHE_TTL,
    SCREEN_STALE_DESC_S,
    SCREEN_WAIT_MAX_S,
    VISION_BACKEND,
    VISION_BASE_URL,
    VISION_LOCAL_MODEL,
    VISION_MODEL,
    _load_local_vlm,
    _load_ocr,
    _ocr_cache,
    _ocr_lock,
    _screen_backend_choice,
    _screen_cached_desc,
    _screen_cache,
    _screen_context,
    _screen_layers,
    _screen_lock,
    _vision_ready,
    _vision_ready_backend,
    _warm_screen_cache,
    _warm_screen_ocr,
)
SCREEN_CONTEXT_TMPL = (
    "स्क्रीन कॉन्टेक्स्ट — यूज़र अभी अपनी स्क्रीन शेयर कर रहा है और तुम उसे देख सकते हो।\n"
    "नियम:\n"
    "1) OCR text सबसे सटीक ground truth है — उसे १००% सच मानो। visual summary सिर्फ़ "
    "layout/मोटा अंदाज़ा है और ग़लत हो सकता है; अगर दोनों में टकराव हो तो OCR text को "
    "मानो और visual summary के हिसाब से चीज़ें मत गढ़ो।\n"
    "2) Active tutor बनो: स्क्रीन पर दिख रही चीज़ों को सीधे reference करो — "
    "'आपके कोड में ये undefined दिख रहा है', 'ये लाइन गलत है'। OCR text से exact "
    "शब्द/एरर quote करो।\n"
    "3) जो दिख नहीं रहा, यूज़र से action मांगो — 'ये file खोलकर दिखाओ', 'उस component "
    "तक scroll करो', 'terminal का output दिखाओ'।\n"
    "4) यूज़र fix करके दिखाए तो बदलाव notice करके confirm करो — 'अब सही दिख रहा है'।\n"
    "5) OCR text में menus/notifications का noise हो सकता है — सिर्फ relevant हिस्सा उठाओ।\n"
    "6) 'स्क्रीन कॉन्टेक्स्ट' जैसे शब्द कभी यूज़र से मत बोलो — सीधे 'आपकी स्क्रीन पर …' कहो।\n"
    "7) कोई दिक्कत दिखे तो पहले बताओ क्या गड़बड़ है, फिर २-३ आसान कदम।"
)


# Used when a screen frame ARRIVES but neither layer has data yet (first
# share in a session: VLM still loading, OCR engine cold). The user IS
# sharing — the tutor must never ask them to share again, and the wait must
# never become the whole reply (the 'एक पल रुको' loop).
SCREEN_PENDING_TMPL = (
    "स्क्रीन स्थिति — यूज़र अभी स्क्रीन शेयर कर रहा है, पर स्क्रीन का विश्लेषण अभी "
    "तैयार नहीं हुआ (कुछ सेकंड लगेंगे)।\n"
    "नियम:\n"
    "1) कभी मत बोलो कि स्क्रीन शेयर नहीं हुई या शेयर बटन दबाओ — स्क्रीन शेयर हो रही है।\n"
    "2) स्क्रीन शेयर की पुष्टि यूज़र से कभी माँगो नहीं — 'स्क्रीन शेयर बटन दबाया है?', "
    "'शेयर चालू करो', 'स्क्रीन दिखाओ' जैसा कुछ भी नहीं। शेयर पहले से चालू है, बस "
    "विश्लेषण लोड हो रहा है।\n"
    "3) स्क्रीन देखने की बात सिर्फ एक छोटी सी लाइन में निपटाओ — जैसे 'मैं स्क्रीन "
    "लोड कर रहा हूँ' — और उसके तुरंत बाद यूज़र के सवाल से जुड़ी कोई दूसरी बात जोड़ो "
    "या एक छोटा सवाल पूछो। इंतज़ार की बात कभी पूरा जवाब नहीं होनी चाहिए।\n"
    "4) अगर यूज़र का सवाल स्क्रीन के बिना भी answer हो सकता है तो पहले पूरा answer दो।\n"
    "5) स्क्रीन विश्लेषण अगले कुछ सेकंड में तैयार हो जाएगा — यूज़र दोबारा पूछे तो "
    "तब स्क्रीन पूरी तरह दिखेगी।"
)


def _screen_context_block(layers: dict) -> str | None:
    """Combine the two screen layers into one system-prompt block."""
    parts = []
    if layers.get("desc"):
        parts.append("स्क्रीन का visual summary:\n" + layers["desc"])
    if layers.get("ocr"):
        parts.append("स्क्रीन पर अभी दिख रहा text (OCR):\n" + layers["ocr"][:1500])
    if not parts:
        return None
    return SCREEN_CONTEXT_TMPL + "\n\n" + "\n\n".join(parts)


# A chat turn without a frame is treated as an active sharing session only if
# we saw screen activity this recently (warm-ups or a frame turn).
SCREEN_RECENT_S = float(os.environ.get("SCREEN_RECENT_S", "25"))


def _recent_screen_block() -> str | None:
    """Best-effort context from the WARM CACHES when a WS turn arrives WITHOUT
    a screen frame (stale client bundle, dropped field, any client bug).

    Warm-ups keep hitting /api/vision while the user shares, so the latest
    desc/OCR is normally a few seconds old — good enough to keep the tutor
    from ever claiming blindness mid-share. Frame-bearing turns always take
    precedence (fresh OCR wins over this).
    """
    if time.monotonic() - vision_service._last_screen_activity > SCREEN_RECENT_S:
        return None
    layers: dict = {"ocr": "", "desc": ""}
    with _screen_lock:
        if _screen_cache["desc"] and time.monotonic() - _screen_cache["ts"] < SCREEN_CACHE_TTL:
            layers["desc"] = _screen_cache["desc"]
    with _ocr_lock:
        if _ocr_cache["text"] and time.monotonic() - _ocr_cache["ts"] < SCREEN_CACHE_TTL:
            layers["ocr"] = _ocr_cache["text"]
    return _screen_context_block(layers)


# Screen context routing mode: "auto" (intent-based, saves tokens) or "always" (legacy)
SCREEN_ROUTING_MODE = os.environ.get("VOICE_SCREEN_ROUTING", "auto").strip().lower()

# Regex to detect when a user utterance actually refers to the screen, code, errors, or visual state
_SCREEN_INTENT_RE = re.compile(
    r"("
    r"स्क्रीन|screen|विंडो|window|डिस्प्ले|display|टैब|tab\b|"
    r"देख|देखो|देखना|दिखा|दिख रहा|दिखाई|देखा|देखकर|look|see|show|visible|watch\b|"
    r"dekho|dekhiye|dekh|dikhao|dikh\s+raha|"
    r"ये\s+क्या|यह\s+क्या|यहाँ|इधर|इसमें|इसपर|is\s+par|isme|what\s+is\s+this|what\'?s\s+this|look\s+at\s+this|"
    r"यहाँ\s+क्या|idhar|yahan|here\b|"
    r"एरर|error|बग|bug|इशू|issue|प्रॉब्लम|problem|दिक्कत|dikkat|गड़बड़|gadbad|मिस्टेक|mistake|गलत|galat|wrong|"
    r"एक्सेप्शन|exception|क्रैश|crash|वार्निंग|warning|fail|"
    r"चल\s+नहीं\s+रहा|काम\s+नहीं\s+कर\s+रहा|nahi\s+chal\s+raha|chal\s+nahi\s+raha|kam\s+nahi\s+kar\s+raha|not\s+working|अटक\s+गया|stuck|"
    r"कोड|code|लाइन|line\b|सिंटैक्स|syntax|फ़ाइल|file\b|टर्मिनल|terminal|कंसोल|console|आउटपुट|output|लॉग|logs?\b|"
    r"चेक|check|इंस्पेक्ट|inspect|रिव्यू|review|फिक्स|fix|सॉल्व|solve|सुधार|सुधारो|पढ़|read|"
    r"बटन|button|फॉर्म|form|कंपोनेंट|component|वेबसाइट|website|पेज|page\b|यूआई|ui\b"
    r")",
    re.IGNORECASE,
)

# Short follow-ups in an ongoing debugging / screen discussion
_SCREEN_FOLLOWUP_RE = re.compile(
    r"^(तो\s+फिर|अब\s+क्या|आगे\s+क्या|कैसे\s+करूँ|कैसे\s+होगा|क्या\s+करूँ|फिक्स\s+कैसे|how\s+to\s+fix|what\s+next|and\s+now\??$)",
    re.IGNORECASE,
)


def _should_include_screen_context(text: str, history: list[dict] | None = None) -> bool:
    """Determine if screen context should be injected into the LLM system prompt.

    Avoids wasting hundreds of tokens on greetings, general theory, or unrelated chat
    while screen share is active.
    """
    if SCREEN_ROUTING_MODE == "always":
        return True

    clean_text = (text or "").strip()
    if not clean_text:
        return False

    # If it directly matches screen/code intent, include it
    if _SCREEN_INTENT_RE.search(clean_text):
        return True

    # If it's pure greeting or casual small-talk without screen keywords, skip
    if GREETING_RE.search(clean_text):
        return False

    # Check if this is a short follow-up to a recent screen-related question
    if history and len(clean_text) < 40 and _SCREEN_FOLLOWUP_RE.search(clean_text):
        user_msgs = [m.get("content", "") for m in history if m.get("role") == "user"]
        if user_msgs and _SCREEN_INTENT_RE.search(user_msgs[-1]):
            return True

    return False




# Serve the built React/Tailwind UI (web/ui/dist). Rebuild with:
#   cd web/ui && npm run build
WEB_DIR = HERE / "web" / "ui" / "dist"


def _llm_api_key() -> str:
    """LLM API key: GROQ_API_KEY -> LLM_API_KEY -> MISTRAL_API_KEY.

    .env is loaded once at startup by _load_dotenv(); read from os.environ only.
    """
    for name in ("GROQ_API_KEY", "LLM_API_KEY", "MISTRAL_API_KEY"):
        key = os.environ.get(name, "").strip()
        if key:
            return key
    return ""


class _LLMRetryable(Exception):
    """Transient LLM API failure (429/5xx/network) worth retrying."""


def _cut_phrase(buf: str, max_chars: int) -> tuple[str, str]:
    """Cut `buf` at a word boundary within the first `max_chars` characters.

    Returns (phrase, rest). Used to flush a partial phrase to TTS before the
    LLM has finished the sentence.
    """
    window = buf[:max_chars]
    cut = window.rfind(" ")
    if cut > 0:
        return window[:cut].strip(), (window[cut + 1:] + buf[max_chars:]).lstrip()
    return window.strip(), buf[max_chars:].lstrip()


def _llm_stream_phrases(
    key: str,
    messages: list[dict],
    temperature: float,
    http_client: httpx.Client,
    max_tokens: int | None = None,
    phrase_chars: int = FIRST_WINDOW_CHARS,
):
    """Stream LLM tokens and yield speakable phrases as soon as they're ready.

    Each item is (text, complete). `complete` is True when the phrase ends at a
    sentence boundary (। ? ! . \\n); False when it was flushed early at a word
    boundary because phrase_chars was reached. The first phrase can be fed to
    TTS before the LLM finishes the sentence — removing the "wait for a full
    sentence" latency from the first audio window.

    Retry behaviour is unchanged: transient failures are retried with backoff
    while NOTHING has been yielded yet, and an empty-success attempt is retried
    once with double max_tokens.
    """
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    total_attempts = 1 + max(0, LLM_RETRIES)
    attempt = 0
    tok_budget = max_tokens or LLM_MAX_TOKENS
    while True:
        payload = {
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": tok_budget,
            "stream": True,
        }
        if LLM_REASONING_EFFORT and "gpt-oss" in LLM_MODEL:
            payload["reasoning_effort"] = LLM_REASONING_EFFORT  # cap thinking time
        buf = ""
        yielded = False
        retry_wait = 0.8
        try:
            with http_client.stream("POST", MISTRAL_URL, headers=headers, json=payload) as r:
                if r.status_code != 200:
                    body = r.read()[:300]
                    if r.status_code == 429 or r.status_code >= 500:
                        ra = r.headers.get("retry-after", "")
                        try:
                            retry_wait = max(retry_wait, min(float(ra), 8.0))
                        except ValueError:
                            pass
                        raise _LLMRetryable(f"LLM API {r.status_code}: {body!r}")
                    raise RuntimeError(f"LLM API {r.status_code}: {body!r}")
                for line in r.iter_lines():
                    if not line:
                        continue
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        delta = json.loads(data)["choices"][0]["delta"].get("content") or ""
                    except (KeyError, IndexError, ValueError):
                        continue
                    if not delta:
                        continue
                    buf += delta
                    while True:
                        m = SENT_END_RE.search(buf)
                        if m:
                            sent = buf[: m.end()].strip()
                            buf = buf[m.end():]
                            if sent:
                                yielded = True
                                yield sent, True
                            continue
                        # No sentence boundary yet: flush a partial phrase once
                        # we have enough text so the first TTS window can start.
                        if len(buf) >= phrase_chars:
                            phrase, buf = _cut_phrase(buf, phrase_chars)
                            if phrase:
                                yielded = True
                                yield phrase, False
                            continue
                        break
            tail = buf.strip()
            if tail:
                yielded = True
                yield tail, True
            if not yielded:
                # Stream "succeeded" but produced nothing speakable — treat as
                # transient (the next attempt gets double tokens) instead of
                # leaving the user with silence.
                raise _LLMRetryable("LLM returned no content (token budget exhausted by reasoning?)")
            return
        except _LLMRetryable as e:
            if yielded or attempt + 1 >= total_attempts:
                raise RuntimeError(str(e)) from e
            attempt += 1
            tok_budget = tok_budget * 2  # reasoning models: give the answer room
            log.warning("LLM attempt %d/%d failed (%s) — retrying in %.1fs",
                        attempt, total_attempts, e, retry_wait)
            time.sleep(retry_wait)
            retry_wait = min(retry_wait * 2, 8.0)
        except httpx.HTTPError as e:
            if yielded or attempt + 1 >= total_attempts:
                raise RuntimeError(f"Mistral stream failed: {e}") from e
            attempt += 1
            log.warning("LLM attempt %d/%d network error (%s) — retrying in %.1fs",
                        attempt, total_attempts, e, retry_wait)
            time.sleep(retry_wait)
            retry_wait = min(retry_wait * 2, 8.0)


_HINDI_NUMS = {
    0: "शून्य", 1: "एक", 2: "दो", 3: "तीन", 4: "चार", 5: "पांच", 6: "छह", 7: "सात", 8: "आठ", 9: "नौ",
    10: "दस", 11: "ग्यारह", 12: "बारह", 13: "तेरह", 14: "चौदह", 15: "पंद्रह", 16: "सोलह", 17: "सत्रह",
    18: "अठारह", 19: "उन्नीस", 20: "बीस", 21: "इक्कीस", 22: "बाईस", 23: "तेईस", 24: "चौबीस", 25: "पच्चीस",
    26: "छब्बीस", 27: "सत्ताईस", 28: "अट्ठाइस", 29: "उनतीस", 30: "तीस", 31: "इकत्तीस", 32: "बत्तीस",
    33: "तैंतीस", 34: "चौंतीस", 35: "पैंतीस", 36: "छत्तीस", 37: "सैंतीस", 38: "अड़तीस", 39: "उनतालीस",
    40: "चालीस", 50: "पचास", 60: "साठ", 70: "सत्तर", 80: "अस्सी", 90: "नब्बे", 100: "सौ",
}


def _convert_numbers_to_hindi(text: str) -> str:
    """Convert numeric digits to spoken Hindi words so OmniVoice never fails on ASCII numbers."""
    def _repl(m: re.Match) -> str:
        s = m.group(0)
        try:
            val = int(s)
            if val in _HINDI_NUMS:
                return _HINDI_NUMS[val]
            if val < 100:
                tens = (val // 10) * 10
                ones = val % 10
                return f"{_HINDI_NUMS.get(tens, '')} {_HINDI_NUMS.get(ones, '')}".strip()
            # Multi-digit numbers (like 404, 2024): pronounce digit-by-digit
            return " ".join(_HINDI_NUMS.get(int(d), d) for d in s)
        except Exception:
            return s

    return re.sub(r"\d+", _repl, text)


def _speech_sentence(sent: str, complete: bool = True) -> str:
    """Make one streamed phrase speakable (strip markup, convert numbers, Devanagari accent).

    complete=False leaves off the terminal danda/full-stop so a phrase that was
    flushed early (mid-sentence) doesn't get an artificial full stop.
    """
    sent = _speechify(sent)
    sent = _convert_numbers_to_hindi(sent)
    sent = _devanagari_only(sent)
    if not sent:
        return ""
    if complete and sent[-1] not in "।?!.":
        sent += "।" if any("\u0900" <= ch <= "\u097F" for ch in sent) else "."
    return sent


def _clause_units(sent: str) -> list[str]:
    """Split text into TTS-window-sized pieces (hard cap, word boundaries).

    Every returned piece is <= _PIECE_MAX chars, so no single audio window can
    grow into a long uninterruptible frame (the #1 thing that kills the
    realtime feel — one giant run-on sentence used to stall TTS for 10s+).
    Cuts land at clause punctuation first, then at spaces; short sentences
    pass through untouched.
    """
    if len(sent) <= _PIECE_MAX:
        return [sent]
    out: list[str] = []
    # first cut at clause punctuation so seams sit at natural pauses
    seps = [m.start() for m in re.finditer(r"[,;—–]", sent)]
    if seps:
        last = 0
        for p in seps:
            if p - last > _PIECE_MAX:
                out.append(sent[last : p + 1].strip())
                last = p + 1
        tail = sent[last:].strip()
        if tail:
            out.append(tail)
    else:
        out.append(sent)
    # then hard-split anything still too long at word boundaries
    final: list[str] = []
    for u in out:
        if len(u) <= _PIECE_MAX:
            final.append(u)
            continue
        words = u.split(" ")
        cur = ""
        for w in words:
            cand = ((cur + " " + w) if cur else w).strip()
            if len(cand) <= _PIECE_MAX:
                cur = cand
                continue
            if cur:
                final.append(cur)
                cur = w
        if cur:
            final.append(cur)
    return [p for p in final if p]


def _chat_worker(state, key, messages, temperature, num_step, speed, out_q, stop_evt, t0=None,
                 http_client: httpx.Client | None = None):
    """3-thread pipeline: LLM producer -> text/windowing -> audio synth.

    Pipeline (all three run concurrently, so nothing serializes):

      1. llm_producer thread  — streams Mistral, yields cleaned sentences
         onto sent_q. Pure network I/O, never blocked by the GPU.
      2. main thread (here)   — pulls sentences, emits a ("text", sent)
         event to the browser IMMEDIATELY, and groups sentences into audio
         windows on win_q. Text typing is never stalled behind TTS.
      3. audio thread         — pulls finished windows off win_q and runs the
         OmniVoice generate() calls. The GPU works while the LLM is still
         writing later sentences.

    Text events therefore stream live at LLM speed (first one lands ~0.5-1s),
    while the first audio window is synthesized at FIRST_WINDOW_STEP (fast
    start) and later windows at the requested num_step, each window a single
    continuous intonation arc instead of choppy per-sentence restarts.

    Events on out_q, in order: ("text", sentence)... then ("audio", wav) per
    window, ending with ("done", None) or ("error", msg).
    """
    if http_client is None:
        http_client = state.runtime.provider_clients.llm()
    # Small talk stays short (same rule as /api/chat): stop after 2 sentences
    last_user = messages[-1]["content"] if messages else ""
    is_greeting = bool(GREETING_RE.search(last_user))
    sent_count = 0
    if t0 is None:
        t0 = time.perf_counter()  # request start (set by the WS handler normally)
    # latency/RTF accounting, filled by audio_synth, reported at the end
    timing = {"first_audio": 0.0, "windows": 0, "total_gen": 0.0, "total_dur": 0.0}

    # ---- producer: read the LLM stream continuously ---------------------
    sent_q: queue.Queue = queue.Queue()
    win_q: queue.Queue = queue.Queue()
    llm_error: list[str] = []
    audio_done = threading.Event()
    producer_stop = threading.Event()  # stop the LLM producer only (graceful small-talk/max cut)

    def llm_producer():
        try:
            for raw, complete in _llm_stream_phrases(key, messages, temperature, http_client):
                if stop_evt is not None and stop_evt.is_set():
                    return
                if producer_stop.is_set():
                    return
                sent = _speech_sentence(raw, complete)
                if not sent or len(sent) <= 2:  # junk like "." or ")." from stray punctuation
                    continue
                sent_q.put((sent, complete))
        except Exception as e:  # noqa: BLE001 — reported at the consumer end
            llm_error.append(f"{type(e).__name__}: {e}")
        finally:
            sent_q.put(None)  # end-of-stream sentinel

    def audio_synth():
        """Drain win_q and generate audio. Keeps the GPU busy in parallel
        with the LLM stream and the browser text events."""
        try:
            while True:
                win = win_q.get()
                if win is None:
                    return
                if stop_evt is not None and stop_evt.is_set():
                    continue  # drop windows queued after an interrupt
                t_gen = time.perf_counter()
                w = TTS_ENGINE.generate(win["text"], win["steps"], _pick_speed(win["text"], speed), TEMPERATURE)
                gen_s = time.perf_counter() - t_gen
                w = TTS_ENGINE.insert_pauses(w, win["text"])
                dur_s = w.shape[-1] / SAMPLE_RATE
                rtf = gen_s / dur_s if dur_s > 0 else 0.0
                timing["windows"] += 1
                timing["total_gen"] += gen_s
                timing["total_dur"] += dur_s
                # Debug event: show the client exactly how the LLM stream was
                # divided into TTS windows, with per-window gen cost + RTF.
                out_q.put(("window", {
                    "n": timing["windows"],
                    "chars": len(win["text"]),
                    "steps": win["steps"],
                    "speed": _pick_speed(win["text"], speed),
                    "audio_s": round(dur_s, 2),
                    "gen_s": round(gen_s, 2),
                    "rtf": round(rtf, 2) if rtf else 0,
                    "text": win["text"],
                }))
                if timing["first_audio"] == 0.0:
                    timing["first_audio"] = time.perf_counter() - t0
                    log.info(
                        "TTS window #1: %d ch, step %d -> %.2fs audio in %.2fs (RTF %.2f) | first audio %.2fs after request",
                        len(win["text"]), win["steps"], dur_s, gen_s, rtf, timing["first_audio"],
                    )
                else:
                    log.info(
                        "TTS window #%d: %d ch, step %d -> %.2fs audio in %.2fs (RTF %.2f)",
                        timing["windows"], len(win["text"]), win["steps"], dur_s, gen_s, rtf,
                    )
                out_q.put(("audio", _wav_bytes(w, SAMPLE_RATE)))
        except Exception as e:  # noqa: BLE001 — reported by the main thread
            llm_error.append(f"audio: {type(e).__name__}: {e}")
        finally:
            audio_done.set()

    threading.Thread(target=llm_producer, daemon=True).start()
    threading.Thread(target=audio_synth, daemon=True).start()

    try:
        window = []          # phrases buffered for the next audio window
        window_chars = 0
        emitted_audio = False
        emitted_text = False  # did the LLM produce ANY speakable sentence?
        text_buf = ""        # partial phrases pending a complete sentence (caption)
        while True:
            item = sent_q.get()
            if item is None:
                break  # LLM stream ended (or was stopped)
            if stop_evt is not None and stop_evt.is_set():
                break
            text, complete = item
            if complete:
                sent_count += 1
                if sent_count > MAX_CHAT_SENTENCES:
                    break  # hard ceiling — never let one turn become a monologue
                caption = (text_buf + " " + text).strip() if text_buf else text
                out_q.put(("text", caption))  # text streams live, never behind TTS
                emitted_text = True
                text_buf = ""
            else:
                text_buf = (text_buf + " " + text).strip()
            # Bound window sizes with clause pieces so one long run-on sentence
            # never delays the first frame (units get re-joined inside a window).
            # Complete and partial phrases both feed the audio windows; windows
            # ship on CHARACTER thresholds, so the first window starts as soon as
            # enough text has streamed instead of waiting for a full sentence.
            for piece in _clause_units(text):
                if window and window_chars + len(piece) > WINDOW_CHAR_CAP:
                    steps = min(num_step, FIRST_WINDOW_STEP) if not emitted_audio else num_step
                    win_q.put({"text": " ".join(window), "steps": steps})
                    window, window_chars = [], 0
                    emitted_audio = True
                window.append(piece)
                window_chars += len(piece)
                if not emitted_audio:
                    if window_chars >= FIRST_WINDOW_CHARS:
                        text_to_speak = " ".join(window)
                        if len(text_to_speak) > FIRST_WINDOW_CHARS:
                            words = text_to_speak.split()
                            cut_text = ""
                            for w in words:
                                if len(cut_text) + len(w) + 1 <= FIRST_WINDOW_CHARS:
                                    cut_text = (cut_text + " " + w).strip()
                                else:
                                    break
                            if not cut_text:
                                cut_text = words[0]
                            remainder = text_to_speak[len(cut_text):].strip()
                            window = remainder.split() if remainder else []
                            window_chars = sum(len(w) for w in window)
                            text_to_speak = cut_text
                        else:
                            window, window_chars = [], 0
                        win_q.put({"text": text_to_speak, "steps": min(num_step, FIRST_WINDOW_STEP)})
                        emitted_audio = True
                else:
                    if window_chars >= MIN_WINDOW_CHARS * 2:
                        win_q.put({"text": " ".join(window), "steps": num_step})
                        window, window_chars = [], 0
                        emitted_audio = True
            if (is_greeting and sent_count >= GREETING_MAX) or sent_count >= MAX_CHAT_SENTENCES:
                # Graceful cut: stop the LLM producer but still speak what is
                # already buffered. stop_evt is reserved for barge-in (drop).
                producer_stop.set()
                break
        if (not emitted_text or (llm_error and not emitted_audio)) and not (stop_evt is not None and stop_evt.is_set()):
            # Always answer out loud — silence reads as "the assistant is broken".
            fallback = "अरे, आवाज़ साफ़ नहीं आ पाई। एक बार फिर से बोल दो।"
            out_q.put(("text", fallback))
            win_q.put({"text": fallback, "steps": min(num_step, FIRST_WINDOW_STEP)})
            emitted_audio = True
        if window and not (stop_evt is not None and stop_evt.is_set()):
            steps = min(num_step, FIRST_WINDOW_STEP) if not emitted_audio else num_step
            win_q.put({"text": " ".join(window), "steps": steps})  # final tail window
        win_q.put(None)  # stop the audio thread
        audio_done.wait(timeout=180)
        if llm_error and not emitted_audio:
            raise RuntimeError(llm_error[0])
        rtf = timing["total_gen"] / timing["total_dur"] if timing["total_dur"] else 0.0
        log.info(
            "TTS total: %d window(s), %.2fs audio in %.2fs gen (avg RTF %.2f) | first audio %.2fs after request",
            timing["windows"], timing["total_dur"], timing["total_gen"], rtf, timing["first_audio"],
        )
        out_q.put(("done", {"first_audio": round(timing["first_audio"], 2), "rtf": round(rtf, 2)}))
    except Exception as e:  # noqa: BLE001 — report to the client
        log.exception("WS chat synthesis failed")
        out_q.put(("error", f"{type(e).__name__}: {e}"))


TTSRequest, ChatMsg, ChatRequest, VisionRequest = make_schemas(
    default_speed=DEFAULT_SPEED,
    step_min=STEP_MIN,
    step_max=STEP_MAX,
    default_step=NUM_STEP,
)


# ---------- Load once at startup ----------
@asynccontextmanager
async def lifespan(_app: FastAPI):
    log.info("Device: %s | dtype: %s", DEVICE, DTYPE)

    ov_model, voice_prompt = TTS_ENGINE.load(REF_AUDIO, REF_TEXT)
    threading.Thread(target=TTS_ENGINE.warm, daemon=True).start()

    _app.state.ov_model = ov_model
    _app.state.voice_prompt = voice_prompt
    _app.state.tts_engine = TTS_ENGINE
    _app.state.gen_lock = asyncio.Lock()
    _app.state.runtime.readiness.tts = True
    _app.state.runtime.readiness.provider = bool(_llm_api_key())

    # Preload the ASR model in the background so the first utterance isn't
    # delayed by the model download/load (up to minutes on slow links).
    threading.Thread(target=_warmup_asr, daemon=True).start()
    # Preload RapidOCR too — without this, the FIRST screen-share turn runs
    # before the OCR engine exists, _screen_layers sees no layer at all, and
    # the tutor wrongly claims it can't see the screen.
    threading.Thread(
        target=lambda: (_load_ocr(), None)[-1] if not vision_service._ocr_disabled else None,
        daemon=True,
    ).start()
    # Preload the local VLM as well. Lazily it loads on the FIRST screen share
    # (30-70 s on Colab/Kaggle), so the first "मेरी स्क्रीन पर क्या है?" turn
    # lands while the describe is still running — no context, and the tutor
    # asks about the share button even though sharing is on. Warm it at boot
    # instead (background thread, after the TTS model, so startup is unchanged
    # in sequence and the weights are ready before the user ever shares).
    if _vision_ready_backend() == "local":
        threading.Thread(target=_load_local_vlm, daemon=True).start()

    yield


app = FastAPI(title="Voice API — Hindi TTS (Rahul's voice)", lifespan=lifespan)
app.state.runtime = VoiceRuntime(
    llm_timeout=LLM_STREAM_TIMEOUT,
    vision_timeout=VISION_TIMEOUT,
)


def health():
    return {"status": "ok", "device": DEVICE}


def ready():
    runtime = app.state.runtime
    runtime.readiness.asr = bool(_asr_ready())
    runtime.readiness.vision = bool(_vision_ready())
    runtime.readiness.provider = bool(_llm_api_key())
    return runtime.readiness.public()


def api_config():
    """Every env-tunable knob, in one JSON blob.

    The web UI fetches this once at startup and drives its chat step, history
    length, reconnect/restart timers, VAD (barge-in) thresholds, and the
    auto-send delay from it — so tuning happens in .env, never in code.
    """
    return {
        # model / quality
        "model": OMNIVOICE_MODEL,
        "num_step": NUM_STEP,
        "step_min": STEP_MIN,
        "step_max": STEP_MAX,
        "temperature": TEMPERATURE,
        "speed": DEFAULT_SPEED,
        "sample_rate": SAMPLE_RATE,
        "device": DEVICE,
        "dtype": str(DTYPE).replace("torch.", ""),
        # local streaming ASR (voice input)
        "asr_backend": _pick_asr_backend(),
        "asr_model": ASR_MODEL,
        "asr_lang": ASR_LANG or "",
        "asr_device": ASR_DEVICE,
        "asr_final_beam": ASR_FINAL_BEAM,
        "asr_initial_prompt": ASR_INITIAL_PROMPT or "",
        "asr_vad_mode": VAD_BACKEND,  # "server" = Silero VAD authority; client opts in per connection
        "silero_silence_ms": SILERO_SILENCE_MS,  # pause length that closes an utterance
        "speaker_gate": bool(_get_speaker_ref() is not None and SPEAKER_GATE not in ("0", "off")),
        "speaker_sim_min": SPEAKER_SIM_MIN,
        "asr_speculative": ASR_SPECULATIVE,
        "spec_chat": ASR_SPECULATIVE,  # the UI reads this key (client-side speculative-turn toggle)
        "speculative_ms": ASR_SPECULATIVE_MS,
        "asr_ready": _asr_ready(),  # True once the warmup finished loading Whisper
        # chat reply shaping
        "first_step": FIRST_WINDOW_STEP,
        "stream_window": STREAM_WINDOW,
        "window_chars": WINDOW_CHAR_CAP,
        "min_window_chars": MIN_WINDOW_CHARS,
        "max_sentences": MAX_CHAT_SENTENCES,
        "greeting_max": GREETING_MAX,
        # vision (screen understanding) — read by web/ui/src/App.jsx
        "vision_enabled": _vision_ready(),
        "vision_backend": _vision_ready_backend() or "off",
        "vision_model": (VISION_MODEL if _vision_ready_backend() == "api" else VISION_LOCAL_MODEL),
        # screen-share capture cadence + warm-up spacing (client-side knobs)
        "screen_tick_ms": int(os.environ.get("SCREEN_TICK_MS", "1200")),
        "screen_prefetch_ms": int(os.environ.get("SCREEN_PREFETCH_MS", "4000")),
        # Push-to-see mode (new default): hold the button, the client warms the
        # vision cache for THIS screen DURING the hold, then sends the turn.
        # 0 = continuous background screen-share (legacy behaviour).
        "screen_push_mode": os.environ.get("SCREEN_PUSH_MODE", "1") == "1",
        "screen_push_max_ms": int(os.environ.get("SCREEN_PUSH_MAX_MS", "5000")),  # auto-send cap
        "screen_push_tick_ms": int(os.environ.get("SCREEN_PUSH_TICK_MS", "400")),  # hold capture cadence
        # LLM
        "llm_model": LLM_MODEL,
        "llm_temperature": LLM_TEMPERATURE,
        "llm_max_tokens": LLM_MAX_TOKENS,
        "diagram_enabled": DIAGRAM_ENABLED,
        # browser-side conversation behaviour (read by web/ui/src/App.jsx)
        "chat_step": int(os.environ.get("VOICE_CHAT_STEP", "8")),  # nfe_step the UI sends
        "jitter_frames": JITTER_FRAMES,
        "max_history": int(os.environ.get("VOICE_MAX_HISTORY", "12")),
        "ws_reconnect_ms": int(os.environ.get("VOICE_WS_RECONNECT_MS", "1500")),
        "rec_restart_ms": int(os.environ.get("VOICE_REC_RESTART_MS", "400")),
        "vad_tick_ms": int(os.environ.get("VOICE_VAD_TICK_MS", "50")),
        "auto_send_ms": int(os.environ.get("VOICE_AUTO_SEND_MS", "750")),
        "send_min_chars": int(os.environ.get("VOICE_SEND_MIN_CHARS", "2")),
        # VAD energy gates (ms of sustained energy etc.)
        "vad_noise_floor": float(os.environ.get("VOICE_VAD_NOISE", "0.005")),
        "vad_threshold_min": float(os.environ.get("VOICE_VAD_THRESHOLD_MIN", "0.014")),
        "vad_gate_mult": float(os.environ.get("VOICE_VAD_GATE_MULT", "3.4")),
        "vad_sustain_ms": int(os.environ.get("VOICE_VAD_SUSTAIN_MS", "250")),  # real voice = ~250ms energy
        "vad_text_ms": int(os.environ.get("VOICE_VAD_TEXT_MS", "150")),  # …or words heard + ~150ms
        "vad_failsafe_ms": int(os.environ.get("VOICE_VAD_FAILSAFE_MS", "900")),
        "speak_tail_ms": int(os.environ.get("VOICE_SPEAK_TAIL_MS", "700")),  # ignore recognition this long after our speaker audio stops
        "vad_rec_active_ms": int(os.environ.get("VOICE_VAD_REC_ACTIVE_MS", "400")),
        "barge_idle_ms": int(os.environ.get("VOICE_BARGE_IDLE_MS", "900")),  # recognizer-idle send safety net
    }


def tts(req: TTSRequest, request: Request):
    start = time.perf_counter()
    try:
        wav = TTS_ENGINE.generate(req.text, req.nfe_step, req.speed, TEMPERATURE)
    except RuntimeError as e:
        log.exception("TTS inference failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    wav = TTS_ENGINE.insert_pauses(wav, req.text)
    elapsed = time.perf_counter() - start
    log.info("Generated %.1fs of audio in %.2fs", wav.shape[-1] / SAMPLE_RATE, elapsed)

    buf = io.BytesIO()
    sf.write(buf, wav, SAMPLE_RATE, format="WAV")
    return Response(
        content=buf.getvalue(),
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="speech.wav"'},
    )


# ---------- WebSocket realtime TTS ----------
#
# Protocol (one connection = one conversation):
#   client ->  {"type": "chat", "text": ..., "history": [...], "nfe_step"?: 8}
#              streams the LLM reply: one {"type":"text", "text": <sentence>}
#              per sentence (the LLM does the chunking), followed by a WAV
#              frame per sentence as it is synthesized — first audio arrives
#              while the model is still writing the rest of the reply.
#   client ->  {"text": "आपका दिन शुभ हो।", "nfe_step"?: 16, "speed"?: 1.0, ...}
#              plain TTS (no LLM) with server-side sentence chunking.
#   server ->  {"type": "start", "sample_rate": 24000}
#              {"type": "text", "text": <sentence>}   (chat only)
#              <binary>  one or more WAV frames (playable, in order)
#              {"type": "done", "frames": n, "elapsed": s}
#   errors  ->  {"type": "error", "message": ...}
#   client may send {"type": "stop"} anytime to abort the current utterance.


def _wav_bytes(samples: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# Punctuation -> natural speech breaks (pause durations in seconds). Each
# pause can be tuned via env (VOICE_PAUSE_COMM A/SEMI/FULL/QUESTION/EXCLAM/
# DANDA) and VOICE_PAUSE_SCALE multiplies them all.


def _pause_for(key: str, dflt: float) -> float:
    return float(os.environ.get(f"VOICE_PAUSE_{key}", str(dflt)))


PAUSE_SECONDS = {
    ",": _pause_for("COMMA", 0.15),
    ";": _pause_for("SEMI", 0.2),
    ".": _pause_for("FULL", 0.3),
    "?": _pause_for("QUESTION", 0.35),
    "!": _pause_for("EXCLAM", 0.4),
    "।": _pause_for("DANDA", 0.35),
}
PAUSE_SCALE = float(os.environ.get("VOICE_PAUSE_SCALE", "1.0"))
PAUSE_SECONDS = {k: round(v * PAUSE_SCALE, 3) for k, v in PAUSE_SECONDS.items()}

# Streaming chunk size in utf-8 bytes (~25 Devanagari chars = one sentence).
STREAM_MAX_CHARS = int(os.environ.get("VOICE_STREAM_MAX_CHARS", "75"))
# Chat audio is synthesized in windows of this many sentences per generate()
# call so prosody flows across the window (1 = old choppy per-sentence mode).
STREAM_WINDOW = max(1, int(os.environ.get("VOICE_STREAM_WINDOW", "2")))
# Max text chars per audio window and per clause piece — on the T4 the GPU
# outruns the audio clock at num_step<=6, so wider windows mean FEWER
# prosody restarts (more natural flow) with no extra wait between frames.
WINDOW_CHAR_CAP = int(os.environ.get("VOICE_WINDOW_CHARS", "110"))
# Never ship a TTS audio window smaller than this (except the final tail).
# Stops the LLM's short sentences from becoming tiny 2-3 word audio chunks
# that keep breaking the flow — windows only go out once they're worth speaking.
# T4: 28 chars ≈ one full clause; smaller values fragment the reply audibly.
MIN_WINDOW_CHARS = int(os.environ.get("VOICE_MIN_WINDOW_CHARS", "28"))
_PIECE_MAX = max(40, min(120, WINDOW_CHAR_CAP))  # single unit fed to TTS
# Hard ceiling on sentences per chat reply (the LLM is told 3-4 but can ramble;
# this bounds worst-case latency so a turn never turns into a monologue).
MAX_CHAT_SENTENCES = int(os.environ.get("VOICE_MAX_SENTENCES", "6"))
# The FIRST audio window uses at most this many diffusion steps (faster start,
# like a human replying quickly); later windows use the requested num_step.
# Set equal to the normal num_step to disable.
FIRST_WINDOW_STEP = max(2, min(32, int(os.environ.get("VOICE_FIRST_STEP", "2"))))  # floor 2 = snappiest first window
# Conversation history kept per turn (older messages dropped). Kept small so
# the LLM prefill stays tiny - the #1 lever for first-audio latency.
MAX_HISTORY = int(os.environ.get("VOICE_MAX_HISTORY", "12"))

TTS_ENGINE = TTSEngine(
    TTSConfig(
        model_name=OMNIVOICE_MODEL,
        device=DEVICE,
        dtype=DTYPE,
        sample_rate=SAMPLE_RATE,
        temperature=TEMPERATURE,
        default_speed=DEFAULT_SPEED,
        stream_max_chars=STREAM_MAX_CHARS,
        first_window_step=FIRST_WINDOW_STEP,
        pause_seconds=PAUSE_SECONDS,
        language=TTS_LANGUAGE,
        pad_duration=TTS_PAD_S,
        fade_duration=TTS_FADE_S,
    ),
    _fix_pronunciation,
)




def _stream_batches(state, text, num_step, speed, stop_evt, t0=None):
    """Yield playable WAV bytes per generated sentence chunk, in order.

    The worker thread (see _synth_worker) pulls from this generator as fast as
    the model produces frames, while the websocket sender streams each frame
    to the client — so chunk n+1 is already being generated while chunk n is
    playing (true prefetch).

    Batched generate([...]) was tried for the tail sentences but returns only
    when the whole batch finishes (no per-item speedup on one device), which
    clumps frames and leaves a long silence after frame 1 — so streaming stays
    one sentence per call for a steady cadence.
    """
    batches = TTS_ENGINE.stream_chunks(text)
    if not batches:
        return
    first = True
    n_win, total_gen, total_dur = 0, 0.0, 0.0
    for gen_text in batches:
        if stop_evt is not None and stop_evt.is_set():
            return
        steps = min(num_step, FIRST_WINDOW_STEP) if first else num_step
        t_gen = time.perf_counter()
        w = TTS_ENGINE.generate(gen_text, steps, _pick_speed(gen_text, speed), TEMPERATURE)
        gen_s = time.perf_counter() - t_gen
        w = TTS_ENGINE.insert_pauses(w, gen_text)
        dur_s = w.shape[-1] / SAMPLE_RATE
        n_win += 1
        total_gen += gen_s
        total_dur += dur_s
        rtf = gen_s / dur_s if dur_s > 0 else 0.0
        if first:
            first_audio = (time.perf_counter() - t0) if t0 else 0.0
            log.info(
                "TTS window #1: %d ch, step %d -> %.2fs audio in %.2fs (RTF %.2f)%s",
                len(gen_text), steps, dur_s, gen_s, rtf,
                f" | first audio {first_audio:.2f}s after request" if t0 else "",
            )
        else:
            log.info(
                "TTS window #%d: %d ch, step %d -> %.2fs audio in %.2fs (RTF %.2f)",
                n_win, len(gen_text), steps, dur_s, gen_s, rtf,
            )
        first = False
        yield _wav_bytes(w, SAMPLE_RATE)
    avg_rtf = total_gen / total_dur if total_dur else 0.0
    log.info(
        "TTS total: %d window(s), %.2fs audio in %.2fs gen (avg RTF %.2f)",
        n_win, total_dur, total_gen, avg_rtf,
    )


def _synth_worker(state, text, num_step, speed, out_q, stop_evt, t0=None):
    try:
        for chunk in _stream_batches(state, text, num_step, speed, stop_evt, t0):
            out_q.put(("audio", chunk))
        out_q.put(("done", None))
    except Exception as e:  # noqa: BLE001 — report to the client
        log.exception("WS synthesis failed")
        out_q.put(("error", f"{type(e).__name__}: {e}"))


def _generate_diagram_for_turn(key, text, history, stop_evt, http_client: httpx.Client):
    return _generate_diagram(
        key,
        text,
        history,
        stop_evt,
        client=http_client,
        url=MISTRAL_URL,
        model=LLM_MODEL,
        reasoning_effort=LLM_REASONING_EFFORT,
    )


async def _send_diagram_when_ready(
    websocket: WebSocket,
    key: str,
    text: str,
    history: list[dict],
    stop_evt: threading.Event,
    turn_id: str,
    client_turn_id: str,
    http_client: httpx.Client,
) -> None:
    try:
        diagram = await asyncio.to_thread(
            _generate_diagram_for_turn,
            key,
            text,
            history,
            stop_evt,
            http_client,
        )
        if not diagram or stop_evt.is_set():
            return
        await websocket.send_text(json.dumps({
            "type": "diagram",
            "turn_id": turn_id,
            "client_turn_id": client_turn_id,
            "diagram": diagram,
        }))
    except Exception as exc:
        log.warning("Diagram generation skipped: %s", exc)


async def ws_tts(websocket: WebSocket):
    await websocket.accept()
    state = websocket.app.state
    stop_evt = threading.Event()
    ctrl: asyncio.Queue = asyncio.Queue()
    busy = [False]  # a generation/chat is currently streaming to this client
    turn_started = [0.0]  # monotonic time the latest turn started (stop-grace guard)
    diagram_tasks: set[asyncio.Task] = set()

    def cancel_diagram_tasks() -> None:
        for task in diagram_tasks:
            task.cancel()
        diagram_tasks.clear()

    async def reader():
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    data = json.loads(raw)
                except (ValueError, AttributeError):
                    data = None
                mtype = data.get("type") if isinstance(data, dict) else None
                if mtype == "ping":  # client heartbeat — reply to keep it alive
                    try:
                        await websocket.send_text(json.dumps({"type": "pong"}))
                    except Exception:
                        pass
                    continue
                if mtype == "stop":
                    # Ignore a stale self-barge that fires immediately after a
                    # fresh turn was submitted (the browser's previous VAD burst
                    # re-sends hardStop and would otherwise kill this reply).
                    if busy[0] and time.monotonic() - turn_started[0] < STOP_GRACE_S:
                        continue
                    stop_evt.set()  # explicit user interrupt
                    cancel_diagram_tasks()
                    continue
                if busy[0] and (mtype == "chat" or data.get("text")):
                    # barge-in: a new request while one is streaming cuts the
                    # current reply short (it will run next, in order)
                    stop_evt.set()
                    cancel_diagram_tasks()
                await ctrl.put(raw)
        except Exception:
            await ctrl.put(None)

    reader_task = asyncio.create_task(reader())
    try:
        while True:
            raw = await ctrl.get()
            if raw is None:
                break
            try:
                data = json.loads(raw)
            except ValueError:
                await websocket.send_text(json.dumps({"type": "error", "message": "Invalid JSON"}))
                continue

            if data.get("type") == "chat":
                text = str(data.get("text", "")).strip()
                if not text:
                    await websocket.send_text(json.dumps({"type": "error", "message": "text is required"}))
                    continue
                try:
                    nfe = int(data.get("nfe_step", data.get("nstep", NUM_STEP)))
                    num_step = max(STEP_MIN, min(STEP_MAX, nfe))
                    speed = float(data.get("speed", DEFAULT_SPEED))
                    temperature = float(data.get("temperature", LLM_TEMPERATURE))
                except (TypeError, ValueError):
                    await websocket.send_text(json.dumps({"type": "error", "message": "bad numeric params"}))
                    continue
                key = _llm_api_key()
                if not key:
                    await websocket.send_text(
                        json.dumps({"type": "error", "message": "LLM API key not configured (set GROQ_API_KEY or MISTRAL_API_KEY in .env)"})
                    )
                    continue
                history = [
                    {"role": m["role"], "content": str(m["content"])}
                    for m in (data.get("history") or [])
                    if isinstance(m, dict) and m.get("role") in ("user", "assistant")
                ]
                if not history or history[-1]["role"] != "user":
                    history.append({"role": "user", "content": text})
                need_screen = _should_include_screen_context(text, history)
                # Push-to-see: a turn that CARRIES a frame is an explicit
                # "look at my screen" — the user held the button for it. Never
                # let the intent regex downgrade it to no-context (a push turn
                # that lost its context is the #1 source of blind replies).
                screen = data.get("screen") if isinstance(data.get("screen"), dict) else None
                if screen and str(screen.get("image") or screen.get("b64") or "").strip():
                    need_screen = True
                # Sub-second latency: SHORT system prompt by default (~0.5 KB vs
                # ~2.5 KB -> ~0.3-1 s less LLM prefill per turn). Screen turns
                # also use the compact persona (the screen rules are appended in
                # the context block), so only a custom VOICE_PROMPT_FILE keeps
                # the full persona.
                if _LLM_PERSONA_CUSTOM:
                    system_prompt = LLM_SYSTEM_PROMPT
                else:
                    system_prompt = LLM_SYSTEM_SHORT
                messages = [{"role": "system", "content": system_prompt}, *history]

                # Screen understanding: a shared screen arrives as an optional
                # {"screen": {"image": <b64>, "hash": <change-detection id>}}.
                # The description is cache-first (hash-keyed), so an unchanged
                # screen adds ZERO vision latency — only a fresh screen pays
                # one Qwen2.5-VL call before the LLM starts writing.
                if screen:
                    vision_service._last_screen_activity = time.monotonic()
                    img = str(screen.get("image") or screen.get("b64") or "").strip()
                    if img.startswith("data:") and "," in img:
                        img = img.split(",", 1)[1]
                    # Push-to-see turns may ask the server to wait a bit longer
                    # for the describe that the hold-time warm-up started.
                    try:
                        screen_wait_s = min(max(float(screen.get("wait_ms", 0) or 0) / 1000.0, 0.0), SCREEN_WAIT_MAX_S)
                    except (TypeError, ValueError):
                        screen_wait_s = 0.0
                    if img:
                        if need_screen:
                            # TWO-LAYER context (executor thread, never the event loop):
                            #   OCR text  — peek cache, else INLINE (~100-300 ms)
                            #               → text is never more than ~1 turn old
                            #   VLM summary — cache peek with bounded wait ONLY;
                            #               background warmer fills it (7 s/screen)
                            _screen_t0 = time.perf_counter()
                            layers = await asyncio.get_running_loop().run_in_executor(
                                None,
                                _screen_layers,
                                str(screen.get("hash") or ""),
                                img,
                                screen_wait_s if screen_wait_s > 0 else None,
                            )
                            log.info(
                                "WS chat: screen context ready in %.2fs (OCR %d chars, VLM %d chars%s)",
                                time.perf_counter() - _screen_t0,
                                len(layers.get("ocr") or ""), len(layers.get("desc") or ""),
                                ", waited for in-flight describe" if screen_wait_s > 0 else "",
                            )
                            block = _screen_context_block(layers)
                            if block:
                                messages[0] = {
                                    "role": "system",
                                    "content": _screen_system_prompt(block),
                                }
                            else:
                                # A frame ARRIVED, so the user IS sharing — never let
                                # the tutor say "share your screen". First try the
                                # recent cached context (a describe from seconds ago
                                # is far better than a pending dead-end — and for
                                # push auto-ask turns it is the CORRECT context: the
                                # screen usually hasn't changed since the last
                                # describe). Only when there is truly nothing do we
                                # tell the LLM analysis is still warming.
                                recent = await asyncio.get_running_loop().run_in_executor(
                                    None, _recent_screen_block
                                )
                                if recent:
                                    log.info(
                                        "WS chat: frame layers empty — using recent cached screen context instead of pending dead-end"
                                    )
                                    messages[0] = {
                                        "role": "system",
                                        "content": _screen_system_prompt(recent),
                                    }
                                else:
                                    log.info(
                                        "WS chat: screen frame received but analysis pending — "
                                        "using pending-context block"
                                    )
                                    messages[0] = {
                                        "role": "system",
                                        "content": _screen_system_prompt(SCREEN_PENDING_TMPL),
                                    }
                            # VLM summary cold → describe in background for the
                            # next turn (reply already has fresh OCR text).
                            # NOT force=True: a per-turn forced re-describe burns
                            # 7-15 s of GPU on the SAME screen and the result is
                            # a near-duplicate. The gated worker skips the turn
                            # anyway if the client has since sent a newer frame.
                            threading.Thread(
                                target=_warm_screen_cache,
                                args=(img, str(screen.get("hash") or "")),
                                daemon=True,
                            ).start()
                        else:
                            log.info("WS chat: screen sharing active, but query does not require screen context (saved tokens)")
                            # Keep background warmer active so cache stays hot for
                            # when the user asks about the screen. Warm OCR TOO —
                            # without it the next screen-intent turn pays full OCR
                            # latency for this screen (the old gap: only the VLM
                            # was warmed here, OCR cache stayed cold).
                            threading.Thread(
                                target=_warm_screen_cache,
                                args=(img, str(screen.get("hash") or "")),
                                daemon=True,
                            ).start()
                            threading.Thread(
                                target=_warm_screen_ocr,
                                args=(img, str(screen.get("hash") or "")),
                                daemon=True,
                            ).start()
                else:
                    # No frame this turn. If screen activity is recent AND query requires screen:
                    if need_screen:
                        recent = await asyncio.get_running_loop().run_in_executor(None, _recent_screen_block)
                        if recent:
                            log.info(
                                "WS chat: using recent cached screen context (activity %.1fs ago)",
                                time.monotonic() - vision_service._last_screen_activity,
                            )
                            messages[0] = {
                                "role": "system",
                                "content": _screen_system_prompt(recent),
                            }

                turn_started[0] = time.monotonic()
                stop_evt.clear()
                cancel_diagram_tasks()
                start = time.perf_counter()
                turn_id = uuid.uuid4().hex
                client_turn_id = str(data.get("client_turn_id", ""))[:80]
                should_diagram = _should_generate_diagram(text, history, DIAGRAM_ENABLED)
                log.info("WS chat request: %s%s", text[:50], " [diagram candidate]" if should_diagram else "")
                busy[0] = True
                # _screen_paused pauses background screen warm-ups until this
                # reply finishes streaming (covers disconnects/exceptions too)
                async with state.gen_lock, _screen_paused():
                    out_q: queue.Queue = queue.Queue()
                    llm_client = state.runtime.provider_clients.llm()
                    threading.Thread(
                        target=_chat_worker,
                        args=(state, key, messages, temperature, num_step, speed, out_q, stop_evt, start, llm_client),
                        daemon=True,
                    ).start()
                    if should_diagram:
                        diagram_event = asyncio.create_task(
                            _send_diagram_when_ready(
                                websocket,
                                key,
                                text,
                                history,
                                stop_evt,
                                turn_id,
                                client_turn_id,
                                llm_client,
                            )
                        )
                        diagram_tasks.add(diagram_event)
                        diagram_event.add_done_callback(diagram_tasks.discard)

                    await websocket.send_text(
                        json.dumps({"type": "start", "sample_rate": SAMPLE_RATE, "text": text})
                    )
                    frames = 0
                    while True:
                        kind, payload = await asyncio.get_running_loop().run_in_executor(None, out_q.get)
                        if kind == "text":
                            await websocket.send_text(json.dumps({"type": "text", "text": payload}))
                        elif kind == "window":
                            await websocket.send_text(
                                json.dumps({"type": "window", **payload})
                            )
                        elif kind == "diagram":
                            if not stop_evt.is_set():
                                await websocket.send_text(json.dumps({"type": "diagram", **payload}))
                        elif kind == "audio":
                            frames += 1
                            if frames == 1:
                                log.info("WS chat first audio frame sent %.2fs after request", time.perf_counter() - start)
                            try:
                                await websocket.send_bytes(payload)
                            except Exception:
                                stop_evt.set()
                                return
                        elif kind == "done":
                            elapsed = round(time.perf_counter() - start, 2)
                            extra = payload if isinstance(payload, dict) else {}
                            log.info(
                                "WS chat done: %d frame(s) in %.2fs | first audio %.2fs | TTS RTF %.2f",
                                frames, elapsed, extra.get("first_audio", 0), extra.get("rtf", 0),
                            )
                            await websocket.send_text(json.dumps({
                                "type": "done", "frames": frames, "elapsed": elapsed,
                                "first_audio": extra.get("first_audio", 0), "rtf": extra.get("rtf", 0),
                            }))
                            break
                        else:  # error
                            log.error("WS chat error: %s", payload)
                            await websocket.send_text(json.dumps({"type": "error", "message": payload}))
                            break
                busy[0] = False
                continue

            text = str(data.get("text", "")).strip()
            if not text:
                await websocket.send_text(json.dumps({"type": "error", "message": "text is required"}))
                continue
            try:
                nfe = int(data.get("nfe_step", data.get("nstep", NUM_STEP)))
                num_step = max(STEP_MIN, min(STEP_MAX, nfe))
                speed = float(data.get("speed", DEFAULT_SPEED))
            except (TypeError, ValueError):
                await websocket.send_text(json.dumps({"type": "error", "message": "bad numeric params"}))
                continue

            turn_started[0] = time.monotonic()
            stop_evt.clear()
            start = time.perf_counter()
            log.info("WS synth request: %s (num_step=%d, speed=%.2f)", text[:50], num_step, speed)
            busy[0] = True
            async with state.gen_lock, _screen_paused():
                out_q: queue.Queue = queue.Queue()
                threading.Thread(
                    target=_synth_worker,
                    args=(state, text, num_step, speed, out_q, stop_evt, start),
                    daemon=True,
                ).start()

                await websocket.send_text(
                    json.dumps({"type": "start", "sample_rate": SAMPLE_RATE, "text": text})
                )
                frames = 0
                while True:
                    kind, payload = await asyncio.get_running_loop().run_in_executor(None, out_q.get)
                    if kind == "audio":
                        frames += 1
                        if frames == 1:
                            log.info("WS synth first audio frame sent %.2fs after request", time.perf_counter() - start)
                        try:
                            await websocket.send_bytes(payload)
                        except Exception:
                            stop_evt.set()
                            return
                    elif kind == "done":
                        elapsed = round(time.perf_counter() - start, 2)
                        log.info("WS synth done: %d frame(s) in %.2fs", frames, elapsed)
                        await websocket.send_text(json.dumps({"type": "done", "frames": frames, "elapsed": elapsed}))
                        break
                    else:  # error
                        log.error("WS synth error: %s", payload)
                        await websocket.send_text(json.dumps({"type": "error", "message": payload}))
                        break
            busy[0] = False
    except WebSocketDisconnect:
        pass
    finally:
        reader_task.cancel()


from server.asr.service import (
    ASR_FINAL_BEAM,
    ASR_INITIAL_PROMPT,
    ASR_LANG,
    ASR_MODEL,
    ASR_SR,
    ASR_SPECULATIVE,
    ASR_SPECULATIVE_MS,
    ASR_BACKEND,
    ASR_DEVICE,
    ASR_COMPUTE,
    ASR_NO_REPEAT_NGRAM,
    ASR_MAX_DUP,
    ASR_PARTIAL_MIN_NEW,
    ASR_PARTIAL_MIN_GAP,
    ASR_CPU_THREADS,
    VAD_BACKEND,
    SILERO_SILENCE_MS,
    SILERO_SILENCE_MIN_MS,
    SILERO_HOLD_THRESH,
    SILERO_ON_THRESH,
    SILERO_ON_MS,
    SERVER_PRE_ROLL_S,
    MIN_UTT_MS,
    SPEAKER_SIM_MIN,
    STOP_GRACE_S,
    _asr_ready,
    _collapse_repeats,
    _get_asr,
    _get_silero,
    _get_speaker_ref,
    _pick_asr_backend,
    _pick_vad_mode,
    _speaker_similarity,
    _warmup_asr,
    _whisper_text,
)


async def ws_asr(websocket: WebSocket):
    """Streaming local transcription of the AEC-cleaned browser mic.

    client -> {"type": "start"}             begin an utterance (clears the buffer)
    client -> {"type": "end"}               end it -> authoritative final transcript
    client -> {"type": "cancel"}            discard the current utterance
    client -> {"type": "early_end"}         silence started: pre-decode while the end-of-speech tail counts down
    client -> {"type": "resume"}            user kept talking: discard the early decode and continue
    client -> {"type": "mode", "vad": "server"}   opt into server-side Silero VAD (client becomes a dumb streamer)
    client -> {"type": "assistant", "active": true|false}  our TTS is playing (gate server VAD opens)
    server -> {"type": "vad_start"}         Silero opened an utterance (client mirrors state / may barge)
    server -> {"type": "vad_end"}           Silero closed it (then speculative/final follow)
    client -> <binary>                        float32 mono PCM @ 16 kHz while talking
    server -> {"type": "partial", "text"}   live caption (throttled by new-audio / gap)
    server -> {"type": "speculative", "text"}  fast greedy transcript at "end" (if enabled)
    server -> {"type": "final", "text"}     authoritative beam transcript after "end"
    """
    await websocket.accept()
    ctrl_q: queue.Queue = queue.Queue()   # "start" / "end" / "cancel" / "__close__"
    pcm_q: queue.Queue = queue.Queue()    # float32 sample arrays from the reader
    out_q: queue.Queue = queue.Queue()    # (kind, text) results -> websocket
    stop_evt = threading.Event()

    async def reader():
        try:
            while True:
                raw = await websocket.receive()
                if raw.get("bytes") is not None:
                    arr = np.frombuffer(raw["bytes"], dtype=np.float32).copy()
                    if arr.size:
                        pcm_q.put(arr)
                elif raw.get("text") is not None:
                    try:
                        msg = json.loads(raw["text"])
                    except (ValueError, AttributeError):
                        msg = None
                    mtype = msg.get("type") if isinstance(msg, dict) else None
                    if mtype == "ping":  # client heartbeat — reply to keep the socket alive
                        try:
                            await websocket.send_text(json.dumps({"type": "pong"}))
                        except Exception:
                            pass
                        continue
                    if mtype == "mode":
                        # client opts into server-side Silero VAD per connection
                        want = str((msg or {}).get("vad", "")).strip().lower()
                        ctrl_q.put("__mode_server__" if want == "server" else "__mode_client__")
                        continue
                    if mtype == "assistant":
                        a = bool((msg or {}).get("active"))
                        ctrl_q.put("__assistant_on__" if a else "__assistant_off__")
                        continue
                    ctrl_q.put(mtype or "__close__" if mtype in ("start", "end", "cancel", "early_end", "resume") else "__noop__")
        except Exception:
            ctrl_q.put("__close__")

    def worker():
        """Owns all blocking Whisper work; results go to out_q for the sender."""
        buf: list = []       # np arrays of the current utterance
        total = 0            # total samples buffered
        new_since = 0.0      # seconds of audio since the last partial run
        last_partial = 0.0   # monotonic time of the last partial run
        open_utt = False
        early = False        # early decode ran during silence confirmation
        held = None          # that pre-decoded transcript (reused at finish)
        total_at_early = 0   # buffer size when the early decode ran (reuse check)
        early_thread = None   # background greedy decode started at silence onset
        in_trailing_silence = False  # silence began: stop buffering trailing silence
        spec_emitted = False  # speculative transcript already sent for this utterance
        # --- server-side Silero VAD state (used when server_mode is on) ---
        server_mode = False  # flipped by the client's {"type":"mode","vad":"server"}
        vad = None           # lazily created _Silero for this connection
        speech_run = 0.0     # ms of consecutive speech (open decision)
        silence_run = 0.0    # ms of consecutive silence (close decision)
        pre_roll: deque = deque(maxlen=int(SERVER_PRE_ROLL_S * ASR_SR / 512))
        pending512 = np.zeros(0, dtype=np.float32)
        assistant_active = False  # our TTS is playing: server VAD must not open

        def finish_utterance(reason: str):
            """Close the open utterance and run the decode pipeline.

            Shared by the legacy client-VAD path ('end') and the server-VAD
            path (Silero silence timeout) so both get: early-decode reuse,
            speculative greedy, and the authoritative beam final.
            """
            nonlocal open_utt, early, held, total_at_early, buf, total, new_since, spec_emitted
            open_utt = False
            samples = np.concatenate(buf) if total else np.zeros(0, dtype=np.float32)
            dur_s = total / ASR_SR
            if early and held is None and early_thread is not None:
                early_thread.join(timeout=0.5)  # wait for the in-flight greedy decode
            reused = early and held is not None and 0 <= total - total_at_early <= int(0.05 * ASR_SR)
            held_text = held or ""
            early = False; held = None; total_at_early = 0
            buf = []; total = 0; new_since = 0.0
            if len(samples) < int(max(0.25, MIN_UTT_MS / 1000.0) * ASR_SR):
                log.info("ASR utterance discarded (%s): %.2fs audio too short", reason, dur_s)
                out_q.put(("rejected", "blip"))   # UI shows a dismiss chip
                out_q.put(("final", ""))
                return
            # Speaker-identity gate: check only on sufficiently long speech (>= 1.8s)
            # because short phrases ("हाँ", "नमस्ते", "ओके") produce noisy embeddings that cause false rejects.
            if len(samples) >= int(1.8 * ASR_SR):
                sim = _speaker_similarity(samples)
                if sim is not None and sim < SPEAKER_SIM_MIN:
                    log.info(
                        "Speaker gate: utterance REJECTED (%s): sim=%.2f < %.2f (not the primary speaker)",
                        reason, sim, SPEAKER_SIM_MIN,
                    )
                    out_q.put(("rejected", "speaker"))  # UI shows a dismiss chip
                    out_q.put(("final", ""))
                    return
            if ASR_FINAL_BEAM <= 1:
                # Fast single-pass decode: greedy Whisper on large-v3-turbo
                if reused:
                    final = held_text
                    log.info("ASR: reused early decode (0.00s on critical path)")
                else:
                    try:
                        t = time.perf_counter()
                        final = _whisper_text(samples)
                        log.info("ASR: %.2fs for %.2fs audio (%s)", time.perf_counter() - t, dur_s, reason)
                    except Exception as e:  # noqa: BLE001 — report; never kill the worker
                        log.exception("ASR decode failed")
                        out_q.put(("error", f"ASR decode failed: {type(e).__name__}: {e}"))
                        return
                if ASR_SPECULATIVE and final and not spec_emitted:
                    spec_emitted = True
                    out_q.put(("speculative", final))
                out_q.put(("final", final))
                return

            if ASR_SPECULATIVE and not spec_emitted:
                if reused:
                    spec = held_text
                    log.info("ASR greedy: reused early decode (0.00s on critical path)")
                else:
                    try:
                        t = time.perf_counter()
                        spec = _whisper_text(samples)
                        log.info("ASR greedy: %.2fs for %.2fs audio", time.perf_counter() - t, dur_s)
                    except Exception:  # noqa: BLE001 — greedy is best-effort
                        spec = ""
                if spec:
                    spec_emitted = True
                    out_q.put(("speculative", spec))
            try:
                t = time.perf_counter()
                final = _whisper_text(samples, beam=ASR_FINAL_BEAM, vad=True)
                log.info("ASR beam: %.2fs for %.2fs audio (%s)", time.perf_counter() - t, dur_s, reason)
            except Exception as e:  # noqa: BLE001 — report; never kill the worker
                log.exception("ASR final decode failed")
                out_q.put(("error", f"ASR decode failed: {type(e).__name__}: {e}"))
                return
            out_q.put(("final", final))

        def start_early_decode():
            """Snapshot the speech so far and decode it greedily in the background.

            Runs during the silence confirmation so finish_utterance() can reuse
            the transcript instead of paying for a greedy decode on the critical
            path. The VAD loop keeps running while this thread works.
            """
            nonlocal early, held, total_at_early, early_thread
            if early or total < int(0.25 * ASR_SR):
                return
            early = True
            held = None  # a previous (stale) early decode must never leak into this run
            snap = np.concatenate(buf)
            total_at_early = total

            def _run():
                nonlocal held
                t0 = time.perf_counter()
                try:
                    result = _whisper_text(snap)
                except Exception:  # noqa: BLE001 — best-effort pre-decode
                    result = None
                held = result
                if result:
                    log.info("ASR early decode: %.2fs for %.2fs audio (overlapped)",
                             time.perf_counter() - t0, total_at_early / ASR_SR)

            early_thread = threading.Thread(target=_run, daemon=True)
            early_thread.start()

        while not stop_evt.is_set():
            try:
                ctl = ctrl_q.get(timeout=0.05)
            except queue.Empty:
                ctl = None
            # always drain whatever audio arrived (may accompany a control msg)
            while True:
                try:
                    arr = pcm_q.get_nowait()
                except queue.Empty:
                    break
                if server_mode:
                    # Buffer to 512-sample frames for Silero; the VAD decides
                    # whether audio feeds the utterance buffer or the pre-roll.
                    pending512 = np.concatenate([pending512, arr])
                    while pending512.size >= 512:
                        frame, pending512 = pending512[:512], pending512[512:]
                        if vad is None:
                            try:
                                vad = _get_silero()
                            except Exception as e:  # noqa: BLE001 — fall back to client VAD
                                log.error("Silero unavailable (%s) — falling back to client VAD", e)
                                server_mode = False
                                break
                        if open_utt and not in_trailing_silence:
                            buf.append(frame)
                            total += len(frame)
                            new_since += len(frame) / ASR_SR
                        p_voice = vad.p(frame)
                        if p_voice >= SILERO_ON_THRESH:
                            speech_run += 32.0; silence_run = 0.0
                            if open_utt and in_trailing_silence:
                                # speech resumed — the early decode is now stale
                                in_trailing_silence = False
                                early = False
                                held = None
                                total_at_early = 0
                        elif p_voice < SILERO_HOLD_THRESH:
                            speech_run = 0.0
                            if open_utt:
                                silence_run += 32.0
                                if not in_trailing_silence:
                                    # first silence frame after speech: stop
                                    # buffering trailing silence and pre-decode
                                    # the speech we have
                                    in_trailing_silence = True
                                    start_early_decode()
                        else:  # hysteresis band: keep current state, no counters
                            if open_utt:
                                silence_run = 0.0
                        # Speculative turn: ship the overlapped greedy transcript as
                        # soon as it's ready so the client can start the LLM while
                        # the remaining silence confirmation + beam decode run.
                        if open_utt and not spec_emitted and early and held:
                            spec_emitted = True
                            out_q.put(("speculative", held))
                            log.info("ASR speculative sent at silence onset (%.0fms)", silence_run)
                        if not open_utt:
                            if speech_run >= SILERO_ON_MS:
                                if assistant_active:
                                    speech_run = 0.0  # gated: our TTS is playing
                                    continue
                                # OPEN: replay pre-roll so the first words (heard
                                # before the decision) are never lost
                                buf.extend(pre_roll)
                                total += sum(len(f) for f in pre_roll)
                                pre_roll.clear()
                                open_utt = True
                                in_trailing_silence = False
                                early = False
                                held = None
                                total_at_early = 0
                                spec_emitted = False
                                speech_run = 0.0; silence_run = 0.0
                                last_partial = time.monotonic()
                                vad.reset()
                                out_q.put(("vad_start", ""))
                                log.info("Server VAD: utterance OPENED")
                            else:
                                pre_roll.append(frame)
                        elif silence_run >= SILERO_SILENCE_MIN_MS and held:
                            log.info("Server VAD: utterance CLOSING early after %.0fms silence (early decode ready)", silence_run)
                            silence_run = 0.0
                            out_q.put(("vad_end", ""))
                            finish_utterance("silence")
                        elif silence_run >= SILERO_SILENCE_MS:
                            log.info("Server VAD: utterance CLOSING after %.0fms silence", silence_run)
                            silence_run = 0.0
                            out_q.put(("vad_end", ""))
                            finish_utterance("silence")
                    continue
                if open_utt:
                    buf.append(arr)
                    total += len(arr)
                    new_since += len(arr) / ASR_SR
            if ctl == "__close__":
                return
            if ctl == "__mode_server__":
                server_mode = True
                log.info("/ws/asr: server-side Silero VAD engaged for this connection")
                continue
            if ctl == "__mode_client__":
                server_mode = False
                continue
            if ctl == "__assistant_on__":
                assistant_active = True
                continue
            if ctl == "__assistant_off__":
                assistant_active = False
                continue
            if ctl == "start":
                buf.clear(); total = 0; new_since = 0.0; open_utt = True
                last_partial = time.monotonic()
                early = False; held = None; total_at_early = 0
                in_trailing_silence = False
                spec_emitted = False
                continue
            if ctl == "cancel":
                buf.clear(); total = 0; new_since = 0.0; open_utt = False
                early = False; held = None; total_at_early = 0
                in_trailing_silence = False
                spec_emitted = False
                continue
            if ctl == "early_end" and open_utt:
                # Overlap trick (client VAD path): the client reports the moment
                # silence starts (its end-of-speech tail begins counting). Run
                # the speculative greedy decode NOW, in parallel with that tail,
                # so at "end" the transcript is already ready. The server-VAD
                # path gets the same overlap for free: decode starts the moment
                # silence is confirmed, while 'vad_end' races to the client.
                if total >= int(0.25 * ASR_SR):
                    snap = np.concatenate(buf)
                    t = time.perf_counter()
                    try:
                        held = _whisper_text(snap)
                    except Exception:  # noqa: BLE001 — best-effort pre-decode
                        held = None
                    log.info(
                        "ASR early decode: %.2fs for %.2fs audio (overlapped)",
                        time.perf_counter() - t, total / ASR_SR,
                    )
                    total_at_early = total
                    early = True
                continue
            if ctl == "resume" and open_utt:
                early = False; held = None; total_at_early = 0  # pre-decode is stale now
                continue
            if ctl == "end":
                finish_utterance("client-vad end")
                continue
            if (ctl == "__noop__" or ctl is None) and open_utt and total >= int(0.5 * ASR_SR) \
                    and new_since >= ASR_PARTIAL_MIN_NEW \
                    and time.monotonic() - last_partial >= ASR_PARTIAL_MIN_GAP:
                new_since = 0.0
                last_partial = time.monotonic()
                samples = np.concatenate(buf)
                try:
                    out_q.put(("partial", _whisper_text(samples)))
                except Exception:  # noqa: BLE001 — captions are best-effort
                    log.exception("ASR partial decode failed")

    reader_task = asyncio.create_task(reader())
    threading.Thread(target=worker, daemon=True).start()
    try:
        while not reader_task.done():
            try:
                kind, text = out_q.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.05)
                continue
            try:
                # error frames carry the reason in "message" (the client reads
                # m.message); transcript frames keep the "text" field.
                frame = (
                    {"type": kind, "message": text}
                    if kind == "error" else {"type": kind, "text": text}
                )
                await websocket.send_text(json.dumps(frame))
            except Exception:  # noqa: BLE001 — client vanished mid-decode; stop the worker too
                break
    except WebSocketDisconnect:
        pass
    finally:
        stop_evt.set()
        reader_task.cancel()


# ---------- Vision: screen understanding ----------
def api_vision(req: VisionRequest):
    """Describe one screenshot with Qwen2.5-VL (cache-first by client hash).

    The browser UI calls this when the screen CHANGES (change detection runs
    client-side), so an idle screen costs nothing and the latest description
    is always warm before the user asks about it.

    Reply-priority: warm-ups are routed through the single vision drain
    worker, which PAUSES whenever a chat turn is synthesizing — a describe
    that lands mid-reply used to steal the GPU from TTS window #2 (seen as
    RTF 1.58 spikes in the logs). The synchronous path is kept ONLY for
    cache hits and `force` (explicit re-describe) requests.
    """
    vision_service._last_screen_activity = time.monotonic()
    img = req.image.strip()
    if img.startswith("data:") and "," in img:
        img = img.split(",", 1)[1]
    if not req.force:
        # Cache peek first — an already-described screen answers instantly.
        with _screen_lock:
            if (
                _screen_cache["hash"] == req.hash
                and _screen_cache["desc"]
                and time.monotonic() - _screen_cache["ts"] < SCREEN_CACHE_TTL
            ):
                return {
                    "description": _screen_cache["desc"], "cached": True,
                    "model": _screen_cache["model"], "backend": _screen_backend_choice(),
                }
        # Not cached: enqueue as the worker's newest job and report
        # accepted-async. The turn's bounded wait (screen.wait_ms) catches the
        # result when it lands; the reply NEVER competes with the describe.
        _warm_screen_cache(img, req.hash, force=False)
        return {
            "description": "", "cached": False, "queued": True,
            "model": "", "backend": _screen_backend_choice(),
        }
    try:
        vision_client = app.state.runtime.provider_clients.vision()
        desc, cached, model = _screen_context(img, req.hash, force=True, http_client=vision_client)
    except Exception as e:  # noqa: BLE001 — log the WHY, not just the 503
        log.exception("/api/vision failed: %s", e)
        raise HTTPException(status_code=503, detail=str(e)) from e
    return {"description": desc, "cached": cached, "model": model, "backend": _screen_backend_choice()}


# ---------- LLM chat (keeps the API key server-side) ----------
def chat(req: ChatRequest, request: Request):
    key = _llm_api_key()
    if not key:
        raise HTTPException(
            status_code=503,
            detail="LLM API key not configured. Create Voice_Cloning/.env with GROQ_API_KEY=... (or legacy MISTRAL_API_KEY=...)",
        )

    http_client = request.app.state.runtime.provider_clients.llm()

    def _call(messages: list[dict]) -> str:
        payload = {
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": req.temperature,
            "max_tokens": LLM_MAX_TOKENS,
        }
        if LLM_REASONING_EFFORT and "gpt-oss" in LLM_MODEL:
            payload["reasoning_effort"] = LLM_REASONING_EFFORT  # cap thinking time
        try:
            r = http_client.post(
                MISTRAL_URL,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload,
            )
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"LLM request failed: {e}") from e
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Mistral API {r.status_code}: {r.text[:300]}")
        try:
            return r.json()["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, ValueError) as e:
            raise HTTPException(status_code=502, detail=f"Unexpected LLM response: {e}") from e

    history = [{"role": m.role, "content": m.content} for m in req.messages]
    messages = [{"role": "system", "content": LLM_SYSTEM_PROMPT}, *history]
    reply = _call(messages)

    reply = _speechify(reply)
    reply = _devanagari_only(reply)
    # Never speak a half sentence — cut at the last complete sentence if truncated
    if reply and reply[-1] not in "।?!.":
        cut = max(reply.rfind("।"), reply.rfind("?"), reply.rfind("!"), reply.rfind("."))
        if cut > 0:
            reply = reply[: cut + 1]
    # Small talk stays short — no examples/rambling, ever
    last_user = history[-1]["content"] if history else ""
    if GREETING_RE.search(last_user):
        reply = _short_greeting(reply)
    log.info("LLM reply: %s", reply[:80])
    return {"reply": reply, "model": LLM_MODEL}


from server.routes import asr as asr_routes
from server.routes import chat as chat_routes
from server.routes import system as system_routes
from server.routes import tts as tts_routes
from server.routes import vision as vision_routes

tts_routes.register(app, tts=tts, ws_tts=ws_tts)
asr_routes.register(app, ws_asr=ws_asr)
vision_routes.register(app, api_vision=api_vision)
chat_routes.register(app, chat=chat)
system_routes.register(app, health=health, ready=ready, api_config=api_config, web_dir=WEB_DIR)


if __name__ == "__main__":
    uvicorn.run(app, host=SETTINGS.host, port=SETTINGS.port, log_level="info")
