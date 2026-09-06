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
import sys
import threading
import time
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
from pydantic import BaseModel, Field, model_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("voice_api")
# faster-whisper (and its vad/silero deps) log an INFO line per decode
# ("Processing audio with duration 00:01.250") — with streaming ASR that is a
# firehose 2-3x per second while you talk. Our own logger reports the same
# events usefully, so quiet theirs down.
logging.getLogger("faster_whisper").setLevel(logging.WARNING)

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


# Everything below is overridable via env vars (or a .env file next to this
# script) — see .env.example for the full documented list.
_load_dotenv(HERE / ".env")

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
NUM_STEP = int(os.environ.get("VOICE_NUM_STEP", "16"))           # diffusion steps; lower = faster
TEMPERATURE = float(os.environ.get("VOICE_TEMPERATURE", "0.3"))  # Kaggle demo default
DEFAULT_SPEED = float(os.environ.get("VOICE_SPEED", "1.0"))      # rate when a request omits speed
# Allowed diffusion-step range and the first-window / greeting caps below are
# all env-tunable (used by /tts, the WebSocket, and TTSRequest validation).
STEP_MIN = int(os.environ.get("VOICE_STEP_MIN", "4"))
STEP_MAX = int(os.environ.get("VOICE_STEP_MAX", "64"))
GREETING_MAX = int(os.environ.get("VOICE_GREETING_MAX", "2"))  # sentences for small-talk replies
FIRST_WINDOW_CHARS = int(os.environ.get("VOICE_FIRST_WINDOW_CHARS", "30"))  # chars in 1st audio window (small = fast first audio)

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


# ---------- LLM: OpenAI-compatible chat endpoint (default Groq, gpt-oss-120b) ----------
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "2000"))
# gpt-oss models "think" before answering — reasoning tokens count against
# max_tokens, so a small budget can end with EMPTY content (silent no-reply).
# "low" keeps first-audio fast; set LLM_REASONING_EFFORT="" to omit the param.
LLM_REASONING_EFFORT = os.environ.get("LLM_REASONING_EFFORT", "low")
MISTRAL_URL = os.environ.get("MISTRAL_URL", "https://api.groq.com/openai/v1/chat/completions")
LLM_STREAM_TIMEOUT = float(os.environ.get("VOICE_LLM_TIMEOUT", "120.0"))  # httpx stream read timeout (s)

LLM_SYSTEM_PROMPT = (
    "तुम एक मस्त और चिल हिंदी टीचर हो जो हर चीज़ एकदम सिंपल तरीके से और "
    "एग्ज़ाम्पल देकर समझाता है — जैसे कोई जेन-ज़ी दोस्त बात कर रहा हो। "
    "सबसे ज़रूरी नियम: शुद्ध / फॉर्मल संस्कृतनिष्ठ हिंदी कभी मत लिखो। "
    "जवाब हमेशा पूरी तरह देवनागरी लिपि में लिखो — एक भी अंग्रेज़ी अक्षर "
    "(a-z, A-Z) मत लिखो। अंग्रेज़ी शब्दों को भी देवनागरी में लिखो, जैसे: एआई, "
    "ऐप, गेम, कूल, बेसिकली, लाइक, सिंपल, फन, डेटा, क्लिक, कोड, वेबसाइट, बटन, "
    "मैसेज, टेंशन, सीन, चिल, स्मार्ट, थिंक, लर्न। "
    "टेक के छोटे नाम (HTML, CSS, API, GPU) देवनागरी अक्षर-नामों में लिखो जैसे: "
    "एचटीएमएल, सीएसएस, एपीआई, जीपीयू — ताकि उच्चारण एकदम साफ़ रहे। "
    "ज़रूरी: यूज़र का बोला हुआ आप तक आवाज़-पहचान (ASR) के ज़रिए आता है, जो अक्सर "
    "नाम और प्रॉपर नाउन ग़लत लिखता है। जवाब लिखने से पहले यूज़र के टेक्स्ट में "
    "नामों को ध्यान से पढ़ो — अगर कोई शब्द असली चीज़/इंसान/किरदार/ब्रांड के नाम "
    "से लगभग मिलता है (जैसे बोला-सुना में फ़र्क़ की वजह से), तो उसे असली सही "
    "नाम से सुधार कर उसी के बारे में जवाब दो। अगर बातचीत की हिस्ट्री में वो "
    "नाम/चीज़ पहले सही रूप में आ चुकी है, तो हिस्ट्री वाला सही नाम ही इस्तेमाल "
    "करो — यूज़र की आख़िरी ग़लत स्पेलिंग कभी मत दोहराओ। यूज़र के सही किए नाम का "
    "कोई ज़िक्र मत करो — सीधे सही नाम इस्तेमाल करो, जैसे कोई दोस्त सुनकर समझ "
    "गया हो। "
    "टोन: एक दोस्त जैसा, आसान और नैचुरल — नकली जेन-ज़ी नकल मत बनो। कैज़ुअल "
    "शब्द (मतलब, यार, भाई, चिल, टेंशन, एकदम) केवल तब इस्तेमाल करो जब वो सच में "
    "फिट हों, बाकी वाक्य साफ़ सीधी बोलचाल की हिंदी में लिखो। "
    "एग्ज़ाम्पल — सुध है ❌: 'कंप्यूटर को मानव की तरह सोचने और काम करने की क्षमता देना।' "
    "एग्ज़ाम्पल — जेन-ज़ी है ✅: 'एआई का मतलब है कंप्यूटर को इतना स्मार्ट बनाना "
    "कि वो खुद सोच सके — बेसिकली एक सुपर-स्मार्ट दोस्त जैसा।' "
    "ऐसे ही बात करो: छोटे वाक्य, आसान शब्द, जैसे व्हाट्सऐप पर दोस्त को लिखते हो। "
    "कठिन या औपचारिक शब्द मत लिखो — 'क्षमता' नहीं, 'पावर या कैपेबिलिटी'; "
    "'आवश्यकता' नहीं, 'ज़रूरत'; 'उदाहरण' की जगह 'एग्ज़ाम्पल'; 'जानकारी' की जगह "
    "'इंफो या जानकारी'। "
    "जरूरी नियम (सबसे पहले याद रखो): अगर उपयोगकर्ता सिर्फ अभिवादन या हालचाल पूछ "
    "रहा है (जैसे 'नमस्ते', 'हेलो', 'हाय', 'कैसे हो', 'क्या चल रहा है', 'क्या हाल', "
    "'hello', 'hi', 'how are you'), तो सिर्फ १-२ वाक्य का सीधा, स्वाभाविक जवाब दो "
    "जैसे 'मैं बहुत अच्छा हूँ यार, तुम कैसे हो?' — कोई एग्ज़ाम्पल मत दो, कोई सवाल "
    "मत जोड़ो, बहुत ज़्यादा मत बोलो। एग्ज़ाम्पल सिर्फ तब दो जब कोई चीज़ या विषय "
    "समझाना हो, और सवाल सिर्फ तब पूछो जब उपयोगकर्ता कुछ सीख रहा हो। "
    "जवाब की शैली: पहले एक लाइन में सीधा जवाब, फिर सिर्फ एक आसान रोज़मर्रा "
    "एग्ज़ाम्पल जैसे 'जैसे-...', और अंत में पूछो कि 'क्या और समझना चाहते हो?' "
    "हाँ/नहीं वाले या छोटे सवालों (जैसे 'मज़ा आता है क्या?', 'तुम कौन हो?') का जवाब "
    "सिर्फ १-२ वाक्य में दो — लंबा लेक्चर मत दो। लंबा समझाना सिर्फ तब जब वो सिखाने के "
    "लिए पूछे (जैसे 'समझाओ', 'क्या है', 'कैसे', 'सिखाओ') — तब भी सिर्फ ३-४ छोटे वाक्य। "
    "कभी कोष्ठक ( ) या डैश (— या -) मत लिखो, और एक वाक्य में एक ही विराम चिह्न हो — "
    "'!!' या '?!' जैसा दोहरा विराम मत लिखो। "
    "आवाज़ में जान लाने के लिए जहाँ सही लगे वहाँ '!' या '...' या '?' इस्तेमाल करो। "
    "खुशी, उत्साह या हैरानी दिखानी हो तो बोलचाल वाले एक्सप्रेशन जोड़ सकते हो — "
    "जैसे 'अरे वाह!', 'वाह!', 'ओहो!', 'हा हा', 'सच में?', 'कमाल है!', 'ज़रूर!' — "
    "लेकिन ज़्यादा मत करो: हर जवाब में ज़्यादा से ज़्यादा १-२ ही जगह, वहीं जहाँ "
    "सच में फिट बैठे। कुछ गंभीरता या सोच-समझकर बताना हो तो 'देख यार...', 'सुन...' "
    "जैसी शुरुआत कर सकते हो। ऐसा नहीं कि हर वाक्य एक्साइटेड लगे — बीच-बीच में "
    "नॉर्मल टोन भी रखो, वरना रोबोटिक लगेगा। "
    "फिलर शब्दों का बैलेंस रखो — 'यार', 'मतलब', 'एकदम' जैसे शब्द पूरे जवाब में "
    "ज़्यादा से ज़्यादा एक-दो बार ही आएँ, वहीं जहाँ असली दोस्त बोलता तो बोलता। हर "
    "वाक्य में 'यार' नहीं आता — वो नकली और चिड़चिड़ा करने वाला लगता है। "
    "कोई चिह्न या फ़ॉर्मेटिंग मत लिखो — न तारांकन (*), न रेखा (---), न क्रमांक "
    "(1., २.), न मोटा अक्षर (**), न बुलेट (-), न इमोजी — पूरी तरह साधारण बोलचाल "
    "के वाक्यों में लिखो, जैसे कोई सुनकर समझे। "
    "आखिरी नियम — आवाज़ के लिए: हर वाक्य कम से कम ६-८ शब्दों का लिखो; "
    "बहुत छोटे २-३ शब्दों के वाक्य मत लिखो, वरना आवाज़ कट-कट जाती है। "
    "हर वाक्य को नई लाइन से शुरू करो (एक लाइन में एक पूरा वाक्य) ताकि बोलते "
    "वक्त सही जगह रुके और फिर से सहज शुरू हो। "
)
# Optional override: point VOICE_PROMPT_FILE at a text file whose contents
# replace the whole system prompt above (tune the persona without editing code).
_PROMPT_FILE = os.environ.get("VOICE_PROMPT_FILE", "")
if _PROMPT_FILE:
    _pf = Path(_PROMPT_FILE).expanduser()
    if _pf.is_file():
        LLM_SYSTEM_PROMPT = _pf.read_text(encoding="utf-8").strip()


def _speechify(text: str) -> str:
    """Strip markdown/symbols/emoji and flatten to plain spoken sentences."""
    text = re.sub(r"<[^>]+>", " ", text)                    # <tag> leftovers
    text = re.sub(r"[\"'\u201c\u201d\u2018\u2019]+", "", text)  # quotes
    text = re.sub(r"[()\[\]{}]+", " ", text)                 # ( ) [ ] { } break the spoken flow
    text = re.sub(r"\*+|_+|`+|#+", "", text)                # *, _, `, #
    text = re.sub(r"^\s*[-=~]{3,}\s*$", " ", text, flags=re.M)  # --- lines
    text = re.sub(r"^\s*(?:[-•]|\d+[.)]|[१२३४५६७८९०]+[.)])\s*", " ", text, flags=re.M)  # bullets/numbers
    text = re.sub(
        r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u2190-\u21FF\u2B00-\u2BFF]+",
        " ",
        text,
    )  # emoji/symbols
    text = re.sub(r"[\r\n]+", " ", text)                   # newlines -> space
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


# English words the LLM may still slip into -> Devanagari, so the TTS keeps the
# Hindi voice/accent everywhere (Latin letters would be read English-accented).
HINGLISH_TO_DEVANAGARI = {
    "ai": "एआई", "app": "ऐप", "apps": "ऐप्स", "game": "गेम", "games": "गेम्स",
    "cool": "कूल", "basically": "बेसिकली", "like": "लाइक", "simple": "सिंपल",
    "fun": "फन", "data": "डेटा", "click": "क्लिक", "code": "कोड",
    "website": "वेबसाइट", "button": "बटन", "message": "मैसेज", "messages": "मैसेज",
    "tension": "टेंशन", "scene": "सीन", "chill": "चिल", "smart": "स्मार्ट",
    "think": "थिंक", "learn": "लर्न", "decision": "डिसीज़न", "example": "एग्ज़ाम्पल",
    "friend": "फ्रेंड", "google": "गूगल", "siri": "सिरी", "assistant": "असिस्टेंट",
    "question": "क्वेश्चन", "answer": "आंसर", "computer": "कंप्यूटर",
    "internet": "इंटरनेट", "online": "ऑनलाइन", "photo": "फोटो", "type": "टाइप",
    "search": "सर्च", "program": "प्रोग्राम", "programming": "प्रोग्रामिंग",
    "best": "बेस्ट", "easy": "ईज़ी", "yaar": "यार", "bhai": "भाई",
    "na": "ना", "bas": "बस", "matlab": "मतलब", "info": "इंफो",
    "power": "पावर", "capability": "कैपेबिलिटी", "super": "सुपर",
    # whole-word tech terms — exact pronunciations the letter-name fallback
    # can't know (acronyms said as words, brand names, etc.)
    "html": "एचटीएमएल", "css": "सीएसएस", "api": "एपीआई", "gpu": "जीपीयू",
    "cpu": "सीपीयू", "ui": "यूआई", "ux": "यूएक्स", "sql": "एसक्यूएल",
    "usb": "यूएसबी", "pdf": "पीडीएफ", "url": "यूआरएल", "wifi": "वाईफाई",
    "json": "जेसन", "java": "जावा", "python": "पायथन",
    "javascript": "जावास्क्रिप्ट", "linux": "लिनक्स", "whatsapp": "व्हाट्सऐप",
    # everyday words that slip through as Latin — say them, don't spell them
    "markup": "मार्कअप", "language": "लैंग्वेज", "languages": "लैंग्वेजेस",
    "software": "सॉफ्टवेयर", "hardware": "हार्डवेयर", "browser": "ब्राउज़र",
    "file": "फाइल", "folder": "फ़ोल्डर", "download": "डाउनलोड",
    "update": "अपडेट", "version": "वर्ज़न", "server": "सर्वर",
    "keyboard": "कीबोर्ड", "screen": "स्क्रीन", "laptop": "लैपटॉप",
    "mobile": "मोबाइल", "developer": "डेवलपर", "design": "डिज़ाइन",
    "practice": "प्रैक्टिस", "english": "इंग्लिश", "science": "साइंस",
    "teacher": "टीचर", "student": "स्टूडेंट", "college": "कॉलेज",
    "class": "क्लास", "time": "टाइम", "video": "वीडियो",
}

# Letter NAMES (not bare consonants) so ANY leftover Latin text spells out
# correctly: "vpn" -> "वी पी एन", not "वपन". This is the root-cause fix for
# TTS reading acronyms as words ("HTML" -> "हटमल").
LATIN_TO_DEVANAGARI = {
    "a": "ए", "b": "बी", "c": "सी", "d": "डी", "e": "ई", "f": "एफ", "g": "जी",
    "h": "एच", "i": "आई", "j": "जे", "k": "के", "l": "एल", "m": "एम", "n": "एन",
    "o": "ओ", "p": "पी", "q": "क्यू", "r": "आर", "s": "एस", "t": "टी", "u": "यू",
    "v": "वी", "w": "डब्ल्यू", "x": "एक्स", "y": "वाई", "z": "ज़ी",
}


HAS_LATIN = re.compile(r"[A-Za-z]")


def _devanagari_only(text: str) -> str:
    """Rewrite leftover Latin words as Devanagari so speech keeps the Hindi accent."""
    def repl(m):
        low = m.group(0).lower()
        if low in HINGLISH_TO_DEVANAGARI:
            return HINGLISH_TO_DEVANAGARI[low]
        return "".join(LATIN_TO_DEVANAGARI.get(ch, "") for ch in low)

    return re.sub(r"[A-Za-z]+", repl, text) if HAS_LATIN.search(text) else text


# Serve the built React/Tailwind UI (web/ui/dist). Rebuild with:
#   cd web/ui && npm run build
WEB_DIR = HERE / "web" / "ui" / "dist"


def _llm_api_key() -> str:
    """LLM API key: GROQ_API_KEY (default provider now) -> LLM_API_KEY ->
    MISTRAL_API_KEY (legacy fallback), from env or the .env file."""
    for name in ("GROQ_API_KEY", "LLM_API_KEY", "MISTRAL_API_KEY"):
        key = os.environ.get(name, "").strip()
        if key:
            return key
        env_file = HERE / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.startswith(f"{name}="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if key:
                        return key
    return ""


def _llm_stream_sentences(key: str, messages: list[dict], temperature: float, max_tokens: int | None = None):
    """Stream LLM tokens and yield complete Devanagari sentences as they finish.

    The LLM effectively does the chunking: each yielded sentence is a TTS
    chunk, so the first sentence can be spoken while the model is still
    writing the rest of the reply (kills the "whole reply first, then audio"
    lag).
    """
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens or LLM_MAX_TOKENS,
        "stream": True,
    }
    if LLM_REASONING_EFFORT and "gpt-oss" in LLM_MODEL:
        payload["reasoning_effort"] = LLM_REASONING_EFFORT  # cap thinking time
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    try:
        with httpx.stream("POST", MISTRAL_URL, headers=headers, json=payload, timeout=LLM_STREAM_TIMEOUT) as r:
            if r.status_code != 200:
                raise RuntimeError(f"LLM API {r.status_code}: {r.read()[:300]!r}")
            buf = ""
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
                # flush every complete sentence that has finished streaming
                while True:
                    m = SENT_END_RE.search(buf)
                    if not m:
                        break
                    sent = buf[: m.end()].strip()
                    buf = buf[m.end():]
                    if sent:
                        yield sent
                if len(buf) > 200:  # pathological run with no punctuation yet
                    yield buf.strip()
                    buf = ""
        tail = buf.strip()
        if tail:
            yield tail
    except httpx.HTTPError as e:
        raise RuntimeError(f"Mistral stream failed: {e}") from e


def _speech_sentence(sent: str) -> str:
    """Make one streamed sentence speakable (strip markup, Devanagari accent)."""
    sent = _speechify(sent)
    sent = _devanagari_only(sent)
    if not sent:
        return ""
    if sent[-1] not in "।?!.":
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


def _chat_worker(state, key, messages, temperature, num_step, speed, out_q, stop_evt, t0=None):
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

    def llm_producer():
        try:
            for raw in _llm_stream_sentences(key, messages, temperature):
                if stop_evt is not None and stop_evt.is_set():
                    return
                sent = _speech_sentence(raw)
                if not sent or len(sent) <= 2:  # junk like "." or ")." from stray punctuation
                    continue
                sent_q.put(sent)
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
                w = _generate(state.ov_model, state.voice_prompt, win["text"],
                              win["steps"], _pick_speed(win["text"], speed),
                              TEMPERATURE)
                gen_s = time.perf_counter() - t_gen
                w = _insert_pauses(w, SAMPLE_RATE, win["text"])
                dur_s = w.shape[-1] / SAMPLE_RATE
                rtf = gen_s / dur_s if dur_s > 0 else 0.0
                timing["windows"] += 1
                timing["total_gen"] += gen_s
                timing["total_dur"] += dur_s
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
        window = []          # sentences buffered for the next audio window
        window_chars = 0
        emitted_audio = False
        emitted_text = False  # did the LLM produce ANY speakable sentence?
        while True:
            sent = sent_q.get()
            if sent is None:
                break  # LLM stream ended (or was stopped)
            if stop_evt is not None and stop_evt.is_set():
                break
            sent_count += 1
            if sent_count > MAX_CHAT_SENTENCES:
                break  # hard ceiling — never let one turn become a monologue
            out_q.put(("text", sent))  # text streams live, never behind TTS
            emitted_text = True
            # Bound window sizes with clause pieces so one long run-on sentence
            # never delays the first frame (units get re-joined inside a window).
            # Whole sentences feed the audio windows — the LLM decides where
            # a sentence ends and we never cut mid-thought. Windows ship on
            # CHARACTER thresholds, never on piece-counts, so short sentences
            # group into one continuous utterance instead of tiny 2-3 word
            # chunks that keep interrupting the flow.
            for piece in _clause_units(sent):
                if window and window_chars + len(piece) > WINDOW_CHAR_CAP:
                    # would overflow -> ship what we have first
                    steps = min(num_step, FIRST_WINDOW_STEP) if not emitted_audio else num_step
                    win_q.put({"text": " ".join(window), "steps": steps})
                    window, window_chars = [], 0
                    emitted_audio = True
                window.append(piece)
                window_chars += len(piece)
                if not emitted_audio:
                    # first window: wait until it's a real sentence big enough
                    # to speak (fast start, but never a fragment)
                    ready = window_chars >= FIRST_WINDOW_CHARS
                else:
                    # later windows: group until there's enough text for a
                    # smooth frame (~2-4 short sentences), never a scrap
                    ready = window_chars >= MIN_WINDOW_CHARS * 2
                if ready:
                    steps = min(num_step, FIRST_WINDOW_STEP) if not emitted_audio else num_step
                    win_q.put({"text": " ".join(window), "steps": steps})
                    window, window_chars = [], 0
                    emitted_audio = True
            if is_greeting and sent_count >= GREETING_MAX:
                if stop_evt is not None:
                    stop_evt.set()  # tell the producer to stop too
                break
            if sent_count >= MAX_CHAT_SENTENCES:
                if stop_evt is not None:
                    stop_evt.set()  # tell the producer to stop too
                break
        if not emitted_text and not (stop_evt is not None and stop_evt.is_set()):
            # The LLM returned no speakable text (reasoning models can burn the
            # whole token budget thinking before saying anything). Always answer
            # out loud — silence reads as "the assistant is broken".
            fallback = "माफ़ कीजिए, एक बार फिर से पूछिए।"
            out_q.put(("text", fallback))
            win_q.put({"text": fallback, "steps": min(num_step, FIRST_WINDOW_STEP)})
            emitted_audio = True
        if window and not (stop_evt is not None and stop_evt.is_set()):
            steps = min(num_step, FIRST_WINDOW_STEP) if not emitted_audio else num_step
            win_q.put({"text": " ".join(window), "steps": steps})  # final tail window
        win_q.put(None)  # stop the audio thread
        audio_done.wait(timeout=180)
        if llm_error:
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


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to speak (Devanagari, Hinglish or English)")
    speed: float = Field(DEFAULT_SPEED, ge=0.3, le=2.0)
    nfe_step: int | None = Field(None, ge=STEP_MIN, le=STEP_MAX,
                                 description=f"Diffusion steps / num_step ({STEP_MIN}-{STEP_MAX}). Lower = faster, lower quality.")
    nstep: int | None = Field(None, ge=STEP_MIN, le=STEP_MAX, description="Alias for nfe_step")

    @model_validator(mode="after")
    def _resolve_nfe_step(self):
        # Accept either nfe_step or its alias nstep; nfe_step wins if both given
        if self.nstep is not None:
            if self.nfe_step is not None and self.nfe_step != self.nstep:
                raise ValueError("nfe_step and nstep disagree; send only one")
            self.nfe_step = self.nstep
        if self.nfe_step is None:
            self.nfe_step = NUM_STEP
        return self


class ChatMsg(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    messages: list[ChatMsg] = Field(..., min_length=1)
    temperature: float = Field(0.7, ge=0.0, le=2.0)


# ---------- Load once at startup ----------
@asynccontextmanager
async def lifespan(_app: FastAPI):
    log.info("Device: %s | dtype: %s", DEVICE, DTYPE)

    # Kaggle demo load: OmniVoice.from_pretrained(..., device_map=..., dtype=...)
    ov_model = OmniVoice.from_pretrained(OMNIVOICE_MODEL, device_map=DEVICE, dtype=DTYPE)

    # Kaggle demo "OPTIMIZATION 1": encode the reference voice once and cache
    # the prompt — every request bypasses raw audio re-processing.
    voice_prompt = ov_model.create_voice_clone_prompt(ref_audio=str(REF_AUDIO), ref_text=REF_TEXT)
    log.info("OmniVoice loaded + voice prompt cached from %s", REF_AUDIO.name)

    _app.state.ov_model = ov_model
    _app.state.voice_prompt = voice_prompt
    _app.state.gen_lock = asyncio.Lock()  # serialize heavy generation across clients

    # Preload the ASR model in the background so the first utterance isn't
    # delayed by the model download/load (up to minutes on slow links).
    threading.Thread(target=_warmup_asr, daemon=True).start()

    yield


app = FastAPI(title="Voice API — Hindi TTS (Rahul's voice)", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.get("/api/config")
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
        "asr_ready": _asr_ready,  # True once the warmup finished loading Whisper
        # chat reply shaping
        "first_step": FIRST_WINDOW_STEP,
        "stream_window": STREAM_WINDOW,
        "window_chars": WINDOW_CHAR_CAP,
        "min_window_chars": MIN_WINDOW_CHARS,
        "max_sentences": MAX_CHAT_SENTENCES,
        "greeting_max": GREETING_MAX,
        # LLM
        "llm_model": LLM_MODEL,
        "llm_temperature": LLM_TEMPERATURE,
        "llm_max_tokens": LLM_MAX_TOKENS,
        # browser-side conversation behaviour (read by web/ui/src/App.jsx)
        "chat_step": int(os.environ.get("VOICE_CHAT_STEP", "12")),  # nfe_step the UI sends
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


@app.post("/tts")
def tts(req: TTSRequest, request: Request):
    state = request.app.state
    start = time.perf_counter()
    try:
        wav = _generate(state.ov_model, state.voice_prompt, req.text, req.nfe_step, req.speed, TEMPERATURE)
    except Exception as e:  # noqa: BLE001 — surface model errors to the client
        log.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    wav = _insert_pauses(wav, SAMPLE_RATE, req.text)
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
    sf.write(buf, samples, sr, format="WAV")
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
# Max text chars per audio window and per clause piece — keeps every frame to
# ~4-5s of speech so a reply starts fast and an interrupt tail stays tiny.
WINDOW_CHAR_CAP = int(os.environ.get("VOICE_WINDOW_CHARS", "90"))
# Never ship a TTS audio window smaller than this (except the final tail).
# Stops the LLM's short sentences from becoming tiny 2-3 word audio chunks
# that keep breaking the flow — windows only go out once they're worth speaking.
MIN_WINDOW_CHARS = int(os.environ.get("VOICE_MIN_WINDOW_CHARS", "24"))
_PIECE_MAX = max(40, min(120, WINDOW_CHAR_CAP))  # single unit fed to TTS
# Hard ceiling on sentences per chat reply (the LLM is told 3-4 but can ramble;
# this bounds worst-case latency so a turn never turns into a monologue).
MAX_CHAT_SENTENCES = int(os.environ.get("VOICE_MAX_SENTENCES", "8"))
# The FIRST audio window uses at most this many diffusion steps (faster start,
# like a human replying quickly); later windows use the requested num_step.
# Set equal to the normal num_step to disable.
FIRST_WINDOW_STEP = max(4, min(32, int(os.environ.get("VOICE_FIRST_STEP", "6"))))


def _stream_chunks(text: str, max_bytes: int) -> list[str]:
    """Split text into small speakable chunks at sentence boundaries (incl. ।),
    hard-splitting anything longer than max_bytes at word/char level."""
    clauses = re.split(r"(?<=[।?!.])\s*", text)
    chunks, cur = [], ""
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        if len((cur + clause).encode("utf-8")) <= max_bytes:
            cur += clause
            continue
        if cur:
            chunks.append(cur)
            cur = ""
        for word in clause.split(" "):
            word = word.strip()
            if not word:
                continue
            cand = (cur + " " + word).strip() if cur else word
            if len(cand.encode("utf-8")) <= max_bytes:
                cur = cand
                continue
            if cur:
                chunks.append(cur)
                cur = ""
            buf = ""
            for ch in word:
                if len((buf + ch).encode("utf-8")) <= max_bytes:
                    buf += ch
                else:
                    chunks.append(buf)
                    buf = ch
            cur = buf
    if cur:
        chunks.append(cur)
    return chunks


def _insert_pauses(wav: np.ndarray, sr: int, text: str) -> np.ndarray:
    """Insert short silences at punctuation marks so speech isn't flat/robotic."""
    total = len(wav)
    if total == 0 or not text:
        return wav
    n_chars = max(len(text), 1)
    parts = []
    start = 0
    for i, ch in enumerate(text):
        pause = PAUSE_SECONDS.get(ch)
        if pause is None:
            continue
        est = int(total * (i + 1) / n_chars)
        if est <= start or est > total:
            continue
        seg = wav[start:est]
        if len(seg) > 0:
            parts.append(seg)
        parts.append(np.zeros(int(pause * sr), dtype=wav.dtype))
        start = est
    if start < total:
        parts.append(wav[start:])
    if not parts:
        return wav
    return np.concatenate(parts)


def _generate(model, voice_prompt, text, num_step, speed, temperature):
    """One OmniVoice voice-clone call -> float32 mono samples at SAMPLE_RATE.

    Mirrors the Kaggle demo: generate(text=..., voice_clone_prompt=<cached>,
    num_step=..., temperature=...). The installed omnivoice (0.2.x) merges
    extra kwargs into OmniVoiceGenerationConfig and silently drops unknown
    keys, so the demo's `temperature` is mapped to `class_temperature` (token-
    sampling temperature; 0 = greedy) to stay effective.
    """
    kwargs = {"text": text, "voice_clone_prompt": voice_prompt, "num_step": num_step}
    if speed is not None:
        kwargs["speed"] = speed
    if temperature is not None:
        kwargs["class_temperature"] = temperature
    # Workaround for the upstream VRAM leak (k2-fsa/OmniVoice issue #199):
    # each generate() leaks/fragments GPU memory that the caching allocator
    # doesn't reclaim on its own, so a long-running server eventually OOMs
    # (fastest on small-VRAM cards). Draining the allocator right before AND
    # after every call, plus pulling outputs onto the CPU, keeps usage flat.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with torch.inference_mode():
        outs = model.generate(**kwargs)
    if not outs:
        raise RuntimeError("OmniVoice returned no audio")
    # Move outputs off the GPU before converting so nothing GPU-side lingers.
    segs = []
    for s in outs:
        if hasattr(s, "detach"):
            s = s.detach()
        if hasattr(s, "cpu"):
            s = s.cpu()
        segs.append(np.asarray(s, dtype=np.float32))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return segs[0] if len(segs) == 1 else np.concatenate(segs)


def _stream_batches(state, text, num_step, speed, stop_evt, t0=None):
    """Yield playable WAV bytes per generated sentence chunk, in order.

    The worker thread (see _synth_worker) pulls from this generator as fast as
    the model produces frames, while the websocket sender streams each frame
    to the client — so chunk n+1 is already being generated while chunk n is
    playing (true prefetch). The only bottleneck is single-device inference
    speed: on this M4/MPS at num_step=8 one sentence takes ~2.5s to make, so
    frames arrive every ~2.5s and the browser plays them back-to-back.

    Batched generate([...]) was tried for the tail sentences but returns only
    when the whole batch finishes (no per-item speedup on one device), which
    clumps frames and leaves a long silence after frame 1 — so streaming stays
    one sentence per call for a steady cadence.
    """
    batches = _stream_chunks(text, STREAM_MAX_CHARS)
    if not batches:
        return
    first = True
    n_win, total_gen, total_dur = 0, 0.0, 0.0
    for gen_text in batches:
        if stop_evt is not None and stop_evt.is_set():
            return
        steps = min(num_step, FIRST_WINDOW_STEP) if first else num_step
        t_gen = time.perf_counter()
        w = _generate(state.ov_model, state.voice_prompt, gen_text, steps, _pick_speed(gen_text, speed), TEMPERATURE)
        gen_s = time.perf_counter() - t_gen
        w = _insert_pauses(w, SAMPLE_RATE, gen_text)
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


@app.websocket("/ws/tts")
async def ws_tts(websocket: WebSocket):
    await websocket.accept()
    state = websocket.app.state
    stop_evt = threading.Event()
    ctrl: asyncio.Queue = asyncio.Queue()
    busy = [False]  # a generation/chat is currently streaming to this client

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
                    stop_evt.set()  # explicit user interrupt
                    continue
                if busy[0] and (mtype == "chat" or data.get("text")):
                    # barge-in: a new request while one is streaming cuts the
                    # current reply short (it will run next, in order)
                    stop_evt.set()
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
                messages = [{"role": "system", "content": LLM_SYSTEM_PROMPT}, *history]

                stop_evt.clear()
                start = time.perf_counter()
                log.info("WS chat request: %s", text[:50])
                busy[0] = True
                async with state.gen_lock:
                    out_q: queue.Queue = queue.Queue()
                    threading.Thread(
                        target=_chat_worker,
                        args=(state, key, messages, temperature, num_step, speed, out_q, stop_evt, start),
                        daemon=True,
                    ).start()

                    await websocket.send_text(
                        json.dumps({"type": "start", "sample_rate": SAMPLE_RATE, "text": text})
                    )
                    frames = 0
                    while True:
                        kind, payload = await asyncio.get_running_loop().run_in_executor(None, out_q.get)
                        if kind == "text":
                            await websocket.send_text(json.dumps({"type": "text", "text": payload}))
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

            stop_evt.clear()
            start = time.perf_counter()
            log.info("WS synth request: %s (num_step=%d, speed=%.2f)", text[:50], num_step, speed)
            busy[0] = True
            async with state.gen_lock:
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


# ---------- Streaming ASR — the voice-input path ----------
# The browser streams the AEC-processed 16 kHz mic PCM here over /ws/asr and
# we transcribe it locally — the same pattern OpenAI/Gemini realtime use:
# recognize the getUserMedia stream AFTER the browser's acoustic echo
# canceller has removed the assistant's own output, instead of the Web Speech
# API (whose separate capture path never gets that echo reference). Partial
# transcripts stream back as live captions; the client signals utterance
# boundaries with "start" / "end" JSON messages.
#
# Two switchable local backends (ASR_BACKEND):
#   * "mlx"            — mlx-whisper on Apple Silicon (Neural Engine). ~10x
#                        faster than CPU Whisper on an M4. Apple-only.
#   * "faster-whisper" — CTranslate2: CUDA fp16 (realtime — the Kaggle/NVIDIA
#                        GPU path) or CPU int8.
# ASR_BACKEND=auto picks mlx when running on Apple Silicon with mlx-whisper
# installed, else faster-whisper — so this exact code runs on the M4 Mac AND
# on Kaggle's Linux GPU with no edits.
ASR_BACKEND = (os.environ.get("ASR_BACKEND", "") or "auto").strip().lower()
# faster-whisper tuning (ignored by the mlx backend): device auto-selects CUDA
# when available, else CPU; compute auto-follows.
ASR_DEVICE = (os.environ.get("ASR_DEVICE", "").strip().lower()
              or ("cuda" if torch.cuda.is_available() else "cpu"))
ASR_COMPUTE = (os.environ.get("ASR_COMPUTE", "").strip().lower()
               or ("float16" if ASR_DEVICE.startswith("cuda") else "int8"))
# Model size (shared by both backends): "small" is the smallest Whisper size
# that transcribes Hindi well (base/tiny mangle it). On CUDA small is
# realtime; on the M4 the mlx backend makes it ~10x realtime.
ASR_MODEL = os.environ.get("ASR_MODEL", "") or "small"
# Spoken language: "hi" = Hindi/Hinglish, accurate and fast (no detection
# pass). LEAVE EMPTY only for true multilingual mode — auto-detect is great
# for English but routinely mislabels SHORT Hindi clips (es/ru/ur/si).
ASR_LANG = os.environ.get("ASR_LANG", "hi") or None
ASR_SR = 16000
# faster-whisper only: the AUTHORITATIVE final after "end" gets a beam-search
# decode + VAD trimming (better Hinglish accuracy for ~1.5x the cost of one
# greedy pass, once per utterance). Live captions stay greedy/beam-1.
# mlx-whisper 0.4.x has no beam decoder, so it always decodes greedily.
ASR_FINAL_BEAM = int(os.environ.get("ASR_FINAL_BEAM", "5"))
# Anti-repetition loops ("अगर अगर अगर…"): every transcript passes through
# _collapse_repeats(), the mlx decoder keeps Whisper's temperature fallback
# ladder (we used to force temperature=0.0, which DISABLED loop detection),
# and faster-whisper additionally blocks repeated 3-word n-grams.
ASR_NO_REPEAT_NGRAM = int(os.environ.get("VOICE_ASR_NO_REPEAT_NGRAM", "3"))  # fw only; 0 = off
ASR_MAX_DUP = int(os.environ.get("VOICE_ASR_MAX_DUP", "2"))  # keep at most N identical neighbours
# Whisper partials cost ~linear in buffered audio, so only re-run when this
# much NEW audio arrived since the last partial AND at least this long passed.
ASR_PARTIAL_MIN_NEW = float(os.environ.get("VOICE_ASR_PARTIAL_NEW_SECONDS", "0.45"))
ASR_PARTIAL_MIN_GAP = float(os.environ.get("VOICE_ASR_PARTIAL_GAP_SECONDS", "0.9"))
# Speculative turn trigger: at "end" we first decode GREEDY (fast) and send it
# to the browser as a "speculative" transcript so the LLM reply starts right
# away — the slower beam+VAD decode then follows as the authoritative "final"
# and the client reconciles the two (replace only if they meaningfully differ).
# This removes the beam decode (~0.3-1.5 s on CPU) from the front of the
# reply latency. VOICE_SPECULATIVE_MS bounds how long the client waits for the
# final before releasing the mic anyway.
ASR_SPECULATIVE = os.environ.get("VOICE_ASR_SPECULATIVE", "1").strip().lower() not in ("0", "false", "no")
ASR_SPECULATIVE_MS = int(os.environ.get("VOICE_SPECULATIVE_MS", "5000"))
# Vocabulary hint passed to Whisper as initial_prompt: proper nouns / brand
# names the ASR mangles ("नोबिता" heard as "Nakota"). Keep it SHORT (one
# line); empty = off.
ASR_INITIAL_PROMPT = os.environ.get("ASR_INITIAL_PROMPT", "").strip() or None

# ---------- Server-side neural VAD (Silero) — the noise-immune turn-taker ----------
#
# The browser's energy VAD cannot tell VOICE from NOISE — a fan, door slam or
# keyboard burst crosses the loudness gate, opens an utterance, and Whisper
# transcribes garbage into a phantom reply. Silero is a tiny neural net that
# classifies speech vs non-speech per 32 ms frame, so noise never opens a turn.
#
# Architecture (what LiveKit/Pipecat-class stacks do):
#   browser  -> streams AEC mic PCM continuously (a thin dumb streamer)
#   server   -> Silero decides open/close; pre-roll ring keeps first words;
#               utterance close triggers the decode pipeline (early decode
#               overlapped while silence is being confirmed, as before)
#   legacy   -> ASR_VAD_MODE=client keeps the old browser-energy behavior
#
# "auto": Silero if the package is installed, else client VAD (graceful).
VAD_MODE = (os.environ.get("ASR_VAD_MODE", "auto").strip().lower() or "auto")
SILERO_ON_THRESH = float(os.environ.get("VOICE_SILERO_ON", "0.55"))     # p(voice) to OPEN (high = noise-proof)
SILERO_HOLD_THRESH = float(os.environ.get("VOICE_SILERO_HOLD", "0.35"))  # p(voice) to STAY open (hysteresis)
SILERO_ON_MS = int(os.environ.get("VOICE_SILERO_ON_MS", "150"))          # speech this long opens the turn
# Silence this long closes it. 1200ms (not 700ms) on purpose: humans pause
# 700-1100ms while thinking mid-sentence, and a shorter window split one
# sentence into two turns — the first fragment got submitted alone ("short
# text, text skipped") and the rest barge-in'd as a second turn.
SILERO_SILENCE_MS = int(os.environ.get("VOICE_SILERO_SILENCE_MS", "1200"))
SERVER_PRE_ROLL_S = float(os.environ.get("VOICE_SERVER_PRE_ROLL_S", "0.4"))  # kept before the open decision
MIN_UTT_MS = int(os.environ.get("VOICE_MIN_UTT_MS", "300"))              # shorter utterances are discarded as blips


def _pick_vad_mode() -> str:
    if VAD_MODE == "server":
        return "server"
    if VAD_MODE in ("client", "browser"):
        return "client"
    try:
        import silero_vad  # noqa: F401
        return "server"
    except ImportError:
        return "client"


VAD_BACKEND = _pick_vad_mode()  # resolved once at boot; logged + served via /api/config


class _Silero:
    """Lazy-loaded Silero VAD (ONNX/JIT, CPU) shared by all /ws/asr workers.

    The model is stateful per stream: each worker calls reset() when an
    utterance opens so hidden state never bleeds across turns. Feed exactly
    512-sample float32 chunks @16 kHz; returns p(speech) in [0, 1].
    """

    def __init__(self):
        import warnings
        from silero_vad import load_silero_vad
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = load_silero_vad()

    def p(self, frame512: np.ndarray) -> float:
        import torch
        with torch.no_grad():
            return float(self.model(torch.from_numpy(frame512), 16000).item())

    def reset(self) -> None:
        """Clear the internal LSTM hidden state (JIT model keeps _h/_c).

        Called when an utterance opens so state from the previous turn can't
        bleed into the new one. Best-effort across silero-vad versions: if the
        attributes move, skipping the reset must never crash the worker.
        """
        import torch
        for attr in ("_h", "_c"):
            try:
                t = getattr(self.model, attr, None)
                if isinstance(t, torch.Tensor):
                    setattr(self.model, attr, torch.zeros_like(t))
            except Exception:  # noqa: BLE001 — hygiene only, never fatal
                pass


_silero: _Silero | None = None
_silero_lock = threading.Lock()


def _get_silero() -> _Silero:
    global _silero
    if _silero is None:
        with _silero_lock:
            if _silero is None:
                _silero = _Silero()
    return _silero


# ---------- Speaker-identity gate (whose voice is it?) ----------
#
# Neither a denoiser nor Silero can reject a SECOND HUMAN VOICE (a phone
# playing a video next to the mic): noise-cancelers remove stationary noise,
# and Silero classifies speech vs non-speech — a video's speech is speech.
# A single microphone also cannot measure distance, so "only the closest
# voice" is not physically available.
#
# The professional answer (the same trick as "Hey Siri" / "OK Google"):
# SPEAKER VERIFICATION. Enroll the primary speaker's voiceprint once (the
# voice-clone reference clip already on disk), and before transcribing an
# utterance verify it actually sounds like that speaker. Cosine similarity
# of d-vector embeddings: same speaker ~0.75-0.95, other voices/noise ~0.4-0.6
# -> a tunable threshold rejects the TV/phone/roommate no matter how loud.
#
# Tunables (.env):
#   VOICE_SPEAKER_GATE   auto (default) = on when resemblyzer + ref clip exist
#                        0/off = disable entirely
#   VOICE_SPEAKER_SIM_MIN  cosine threshold (default 0.62); raise for stricter
SPEAKER_GATE = (os.environ.get("VOICE_SPEAKER_GATE", "auto").strip().lower() or "auto")
# 0.45, not 0.62: the reference clip is a clean recording, but the LIVE voice
# arrives through Chrome's AEC/NS/AGC, and that channel difference alone drops
# the SAME speaker's cosine similarity to ~0.49-0.55 (observed on this setup).
# 0.62 rejected the user's own voice on every utterance; 0.45 still separates
# a clearly-different voice (phone/video/TV ~0.30-0.45) from the primary user.
SPEAKER_SIM_MIN = float(os.environ.get("VOICE_SPEAKER_SIM_MIN", "0.45"))
_speaker_emb: np.ndarray | None = None
_speaker_enc = None
_speaker_lock = threading.Lock()


def _get_speaker_ref() -> np.ndarray | None:
    """Lazily build the primary-speaker voiceprint from the reference clip."""
    global _speaker_emb, _speaker_enc
    if SPEAKER_GATE in ("0", "off", "false", "no"):
        return None
    if _speaker_emb is None:
        with _speaker_lock:
            if _speaker_emb is None:
                if not REF_AUDIO.exists():
                    log.warning("Speaker gate: reference clip %s missing — gate disabled", REF_AUDIO.name)
                    _speaker_emb = np.zeros(0, dtype=np.float32)  # sentinel: unavailable
                    return None
                try:
                    from resemblyzer import VoiceEncoder, preprocess_wav
                    if _speaker_enc is None:
                        _speaker_enc = VoiceEncoder()
                    wav = preprocess_wav(str(REF_AUDIO))
                    if len(wav) < int(1.0 * 16000):
                        log.warning("Speaker gate: reference clip too short — gate disabled")
                        _speaker_emb = np.zeros(0, dtype=np.float32)
                        return None
                    _speaker_emb = _speaker_enc.embed_utterance(wav)
                    log.info("Speaker gate enrolled from %s (%.1fs of voice)", REF_AUDIO.name, len(wav) / 16000)
                except ImportError:
                    log.warning("Speaker gate: resemblyzer not installed (uv pip install resemblyzer) — gate disabled")
                    _speaker_emb = np.zeros(0, dtype=np.float32)
                    return None
                except Exception as e:  # noqa: BLE001 — gate must never kill ASR
                    log.warning("Speaker gate enrollment failed (%s) — gate disabled", e)
                    _speaker_emb = np.zeros(0, dtype=np.float32)
                    return None
    if _speaker_emb is not None and _speaker_emb.size == 0:
        return None
    return _speaker_emb


def _speaker_similarity(samples: np.ndarray) -> float | None:
    """Cosine similarity of an utterance against the enrolled voiceprint.

    Uses only the loudest ~6 s of speech (embed_utterance is O(seconds) and
    utterances are short anyway). None = gate unavailable/disabled.
    """
    ref = _get_speaker_ref()
    if ref is None:
        return None
    try:
        from resemblyzer import preprocess_wav
        if samples.size > 6 * ASR_SR:
            samples = samples[-6 * ASR_SR:]  # most recent speech carries the ID
        emb = _speaker_enc.embed_utterance(preprocess_wav(samples, ASR_SR))
        return float(np.dot(ref, emb))
    except Exception as e:  # noqa: BLE001 — fail open: decode proceeds
        log.warning("Speaker gate check failed (%s) — allowing utterance", e)
        return None

# HF repo ids of the MLX-converted Whisper checkpoints per size. A full repo id
# (containing "/") in ASR_MODEL is passed through untouched.
# NOTE: the turbo checkpoint on HF is "whisper-large-v3-turbo" (no -mlx suffix) —
# guessing the suffix produced a 404 that silently killed ASR.
_MLX_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "large-turbo": "mlx-community/whisper-large-v3-turbo",  # alias
    "turbo": "mlx-community/whisper-turbo",
}


def _mlx_repo(model_name: str) -> str:
    if "/" in model_name:
        return model_name
    return _MLX_REPOS.get(model_name.lower(), f"mlx-community/whisper-{model_name}-mlx")


def _pick_asr_backend() -> str:
    """Which backend will run (no heavy imports): auto prefers mlx on the Mac."""
    if ASR_BACKEND == "mlx":
        return "mlx"
    if ASR_BACKEND in ("faster-whisper", "faster_whisper", "fw"):
        return "faster-whisper"
    if sys.platform == "darwin":  # auto: Apple Silicon + mlx installed -> mlx
        import importlib.util
        if importlib.util.find_spec("mlx_whisper") is not None:
            return "mlx"
    return "faster-whisper"


_asr_backend = None   # one loaded backend object, reused for every call
_asr_lock = threading.Lock()  # one Whisper call at a time (single device)
_asr_ready = False    # True once the ASR backend finished its warmup load


class _MlxAsr:
    """mlx-whisper (Apple Neural Engine). Greedy decode only; the library caches
    the loaded model in memory, so repeated calls stay fast."""

    name = "mlx"

    def __init__(self):
        try:
            import mlx_whisper  # noqa: F401 — Apple-only import
        except ImportError:
            raise RuntimeError(
                "ASR_BACKEND=mlx needs mlx-whisper (Apple Silicon only): run "
                "'uv pip install -p omnivoice-env/bin/python mlx-whisper'. "
                "On Kaggle/NVIDIA leave ASR_BACKEND=auto to use faster-whisper on CUDA."
            ) from None
        self._transcribe = mlx_whisper.transcribe
        self._repo = _mlx_repo(ASR_MODEL)
        # Fail fast + load at boot, not on the user's first utterance: a wrong
        # ASR_MODEL used to surface as an HF 404 INSIDE the /ws/asr worker and
        # silently kill transcription while /api/config still said asr_ready.
        from huggingface_hub import model_info
        model_info(self._repo)  # raises RepositoryNotFoundError for a bad repo id
        try:
            import numpy as _np
            self.transcribe(_np.zeros(ASR_SR, dtype=_np.float32))  # warm the weights
        except Exception as e:  # noqa: BLE001 — warm decode is best-effort
            log.warning("mlx ASR warm decode failed (first utterance may be slower): %s", e)

    def transcribe(self, samples: np.ndarray, beam: int = 1, vad: bool = False) -> str:
        # Keep mlx-whisper's temperature fallback ladder (we used to pin
        # temperature=0.0, which disabled Whisper's own repetition-loop
        # detection — the cause of "अगर अगर अगर…" transcripts).
        kwargs: dict = {
            "condition_on_previous_text": False,
            "verbose": None,
            "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        }
        if ASR_LANG:
            kwargs["language"] = ASR_LANG
        if ASR_INITIAL_PROMPT:
            kwargs["initial_prompt"] = ASR_INITIAL_PROMPT
        try:
            res = self._transcribe(samples, path_or_hf_repo=self._repo, **kwargs)
        except TypeError:
            # older mlx-whisper without the tuple-temperature API -> plain greedy
            kwargs = {k: v for k, v in kwargs.items() if k != "temperature"}
            res = self._transcribe(samples, path_or_hf_repo=self._repo, **kwargs)
        return (res.get("text") or "").strip()


class _FasterWhisperAsr:
    """faster-whisper (CTranslate2): CUDA fp16 or CPU int8."""

    name = "faster-whisper"

    def __init__(self):
        from faster_whisper import WhisperModel  # lazy: non-voice users never load it
        self._model = WhisperModel(ASR_MODEL, device=ASR_DEVICE, compute_type=ASR_COMPUTE)

    def transcribe(self, samples: np.ndarray, beam: int = 1, vad: bool = False) -> str:
        segs, _info = self._model.transcribe(
            samples, language=ASR_LANG, beam_size=beam,
            condition_on_previous_text=False, vad_filter=vad,
            initial_prompt=ASR_INITIAL_PROMPT,
            no_repeat_ngram_size=max(0, ASR_NO_REPEAT_NGRAM),  # 0 disables (CTranslate2 convention)
        )
        return "".join(s.text for s in segs).strip()


def _get_asr():
    """Load (once) the backend selected by ASR_BACKEND and reuse it for all calls."""
    global _asr_backend
    if _asr_backend is None:
        with _asr_lock:  # serialize construction — never load the model twice
            if _asr_backend is None:
                chosen = _pick_asr_backend()
                log.info("Loading ASR backend '%s' (model '%s')...", chosen, ASR_MODEL)
                _asr_backend = _MlxAsr() if chosen == "mlx" else _FasterWhisperAsr()
                global _asr_ready
                _asr_ready = True
                log.info("ASR backend '%s' ready (model '%s').", chosen, ASR_MODEL)
    return _asr_backend


def _warmup_asr():
    """Background warmup: preload the ASR model at startup so the FIRST
    utterance isn't delayed by the model download/load. Never crashes boot."""
    try:
        _get_asr()
        log.info("ASR warmup complete.")
    except Exception as e:  # noqa: BLE001 — warmup must never kill the server
        log.error("ASR warmup failed (will retry lazily on first use): %s", e)


def _collapse_repeats(text: str) -> str:
    """Fix Whisper repetition loops: "अगर अगर अगर अगर…" -> "अगर".

    Two passes:
      1. an identical word repeated more than ASR_MAX_DUP times IN A ROW is
         truncated (Hindi reduplication like "धीरे धीरे" still survives);
      2. if a single word makes up >= 40% of the whole utterance (>= 4 times),
         only its first two occurrences are kept — loops often RESTART mid-
         sentence instead of staying adjacent.
    Stray U+FFFD chars (truncated Devanagari tokens like "अ�") become spaces.
    """
    t = (text or "").strip()
    if not t:
        return t
    t = t.replace("\ufffd", " ")
    words = t.split()
    out: list[str] = []
    run_word, run_len = None, 0
    for w in words:
        if w == run_word:
            run_len += 1
            if run_len > max(1, ASR_MAX_DUP):
                continue  # collapse the (N+1)-th identical neighbour
        else:
            run_word, run_len = w, 1
        out.append(w)
    if len(out) >= 4:
        counts: dict[str, int] = {}
        for w in out:
            counts[w] = counts.get(w, 0) + 1
        hot, n = max(counts.items(), key=lambda kv: kv[1])
        if n >= 4 and n * 5 >= len(out) * 2:  # hot word is >= 40% of the utterance
            seen, kept = 0, []
            for w in out:
                if w == hot:
                    seen += 1
                    if seen > 2:
                        continue
                kept.append(w)
            out = kept
    # loops often leave an orphan single-char Devanagari fragment ("अ") at the
    # end — drop it (real standalone words are never 1 Devanagari char except न)
    while len(out) >= 2 and len(out[-1]) == 1 and out[-1] != "न" \
            and "\u0900" <= out[-1][0] <= "\u097F":
        out.pop()
    return " ".join(out)


def _whisper_text(samples: np.ndarray, beam: int = 1, vad: bool = False) -> str:
    """Transcribe one utterance buffer -> trimmed text (calls serialized).

    beam>1 / vad=True are applied only by the faster-whisper backend for the
    authoritative "final"; live partial captions stay greedy/raw everywhere.
    The mlx backend has no beam decoder and ignores both.
    """
    if samples.size == 0:
        return ""
    asr = _get_asr()
    with _asr_lock:
        text = asr.transcribe(np.ascontiguousarray(samples, dtype=np.float32), beam=beam, vad=vad)
    return _collapse_repeats(text)


@app.websocket("/ws/asr")
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
            nonlocal open_utt, early, held, total_at_early, buf, total, new_since
            open_utt = False
            samples = np.concatenate(buf) if total else np.zeros(0, dtype=np.float32)
            dur_s = total / ASR_SR
            reused = early and held is not None and 0 <= total - total_at_early <= int(0.05 * ASR_SR)
            held_text = held or ""
            early = False; held = None; total_at_early = 0
            buf = []; total = 0; new_since = 0.0
            if len(samples) < int(max(0.25, MIN_UTT_MS / 1000.0) * ASR_SR):
                log.info("ASR utterance discarded (%s): %.2fs audio too short", reason, dur_s)
                out_q.put(("rejected", "blip"))   # UI shows a dismiss chip
                out_q.put(("final", ""))
                return
            # Speaker-identity gate: is this the PRIMARY speaker (or someone
            # else / a phone video in the background)? Same speaker ~0.75-0.95,
            # others ~0.4-0.6 -> rejected utterances are silently dropped.
            sim = _speaker_similarity(samples)
            if sim is not None and sim < SPEAKER_SIM_MIN:
                log.info(
                    "Speaker gate: utterance REJECTED (%s): sim=%.2f < %.2f (not the primary speaker)",
                    reason, sim, SPEAKER_SIM_MIN,
                )
                out_q.put(("rejected", "speaker"))  # UI shows a dismiss chip
                out_q.put(("final", ""))
                return
            if ASR_SPECULATIVE:
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
                        if open_utt:
                            buf.append(frame)
                            total += len(frame)
                            new_since += len(frame) / ASR_SR
                        p_voice = vad.p(frame)
                        if p_voice >= SILERO_ON_THRESH:
                            speech_run += 32.0; silence_run = 0.0
                        elif p_voice < SILERO_HOLD_THRESH:
                            speech_run = 0.0
                            if open_utt:
                                silence_run += 32.0
                        else:  # hysteresis band: keep current state, no counters
                            if open_utt:
                                silence_run = 0.0
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
                                speech_run = 0.0; silence_run = 0.0
                                last_partial = time.monotonic()
                                vad.reset()
                                out_q.put(("vad_start", ""))
                                log.info("Server VAD: utterance OPENED")
                            else:
                                pre_roll.append(frame)
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
                continue
            if ctl == "cancel":
                buf.clear(); total = 0; new_since = 0.0; open_utt = False
                early = False; held = None; total_at_early = 0
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


# ---------- LLM chat (keeps the API key server-side) ----------
@app.post("/api/chat")
def chat(req: ChatRequest):
    key = _llm_api_key()
    if not key:
        raise HTTPException(
            status_code=503,
            detail="LLM API key not configured. Create Voice_Cloning/.env with GROQ_API_KEY=... (or legacy MISTRAL_API_KEY=...)",
        )

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
            r = httpx.post(
                MISTRAL_URL,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload,
                timeout=60.0,
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


# Serve the browser voice-chat panel at http://127.0.0.1:8000/
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


if __name__ == "__main__":
    host = os.environ.get("VOICE_HOST", "0.0.0.0")
    port = int(os.environ.get("VOICE_PORT", "8000"))
    uvicorn.run(app, host=host, port=port, log_level="info")
