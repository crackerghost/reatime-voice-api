"""Speech text cleanup and Hindi pronunciation normalization."""

import re

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
    "github": "गिटहब", "git": "गिट", "openai": "ओपन एआई",
    "chatgpt": "चैटजीपीटी", "youtube": "यूट्यूब", "android": "एंड्रॉइड",
    "npm": "एनपीएम", "aws": "एडब्ल्यूएस", "kaggle": "कैगल",
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

# Words the TTS (OmniVoice) misreads even in pure Devanagari — usually conjunct
# clusters with stacked matras (जिससे -> model merges ि+स+े and says "jisse"
# wrong / skips the double स). Fix by respelling at the SYLLABLE SEAM: a space
# spoken aloud is identical to the correct pronunciation, but the model
# segments each part cleanly. Order matters: longest keys first.
PRONUNCIATION_FIXES = {
    # double-स family (स+स conjunct with matras — the model merges them)
    "जिससे": "जिस से",
    "इससे": "इस से",
    "उससे": "उस से",
    "किससे": "किस से",
    "बससे": "बस से",
    # double-च conjunct (च्छ) — the most frequent word in spoken Hindi
    "अच्छा": "अच छा",
    "अच्छी": "अच छी",
    "अच्छे": "अच छे",
    "अच्छाः": "अच छाः",
    # conjunct + matra clusters the model reads as one garbled syllable
    "क्योंकि": "क्यों कि",
    "इसलिए": "इस लिए",
    # conversational Hinglish the LLM now uses — respelled at syllable seams
    # so OmniVoice says them like a speaker, not a textbook
    "एक्चुअली": "एक चुआ ली",
    "बेसिकली": "बे सिक ली",
    "समस्या": "सम स्या",
    "इस्तेमाल": "इस्ते माल",
    "क्षमा": "क्ष मा",
}
_PRONUNCIATION_RE = re.compile(
    "|".join(re.escape(k) for k in sorted(PRONUNCIATION_FIXES, key=len, reverse=True))
)


def _fix_pronunciation(text: str) -> str:
    """Respell problem words so the TTS says them correctly (sounds identical)."""
    if not PRONUNCIATION_FIXES or not text:
        return text
    return _PRONUNCIATION_RE.sub(
        lambda m: PRONUNCIATION_FIXES[m.group(0)], text
    )


HAS_LATIN = re.compile(r"[A-Za-z]")


def _devanagari_only(text: str) -> str:
    """Rewrite leftover Latin words as Devanagari so speech keeps the Hindi accent."""
    def repl(m):
        low = m.group(0).lower()
        if low in HINGLISH_TO_DEVANAGARI:
            return HINGLISH_TO_DEVANAGARI[low]
        return "".join(LATIN_TO_DEVANAGARI.get(ch, "") for ch in low)

    return re.sub(r"[A-Za-z]+", repl, text) if HAS_LATIN.search(text) else text

