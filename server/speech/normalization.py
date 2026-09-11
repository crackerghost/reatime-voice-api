"""Speech text cleanup and Hindi pronunciation normalization."""

import re

def _preserve_code_tokens(text: str) -> str:
    """Keep code/HTML tokens speakable BEFORE _speechify strips markup.

    <html> / </head> -> " html टैग " (later maps to एचटीएमएल टैग via
    HINGLISH_TO_DEVANAGARI). Without this, teaching HTML speaks as
    "टैग्स – आदि" with every tag name eaten by the <tag> stripper.
    """
    if not text or "<" not in text:
        return text
    return re.sub(r"</?\s*([A-Za-z][A-Za-z0-9]*)\s*/?>", r" \1 टैग ", text)


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
    # ---- daily courtesy / reactions (rose-roz wale) ----
    "hello": "हेलो", "hey": "हे", "hi": "हाय", "bye": "बाय",
    "thanks": "थैंक्स", "thank": "थैंक", "thankyou": "थैंक्यू",
    "sorry": "सॉरी", "please": "प्लीज़", "welcome": "वेलकम",
    "ok": "ओके", "okay": "ओके", "sure": "श्योर", "great": "ग्रेट",
    "nice": "नाइस", "awesome": "ऑसम", "perfect": "परफेक्ट",
    "done": "डन", "good": "गुड", "cool": "कूल",
    "congrats": "कॉन्ग्रैट्स", "allright": "ऑलराइट", "alright": "ऑलराइट",
    # ---- tutor classroom (tution wale) ----
    "learn": "लर्न", "teach": "टीच", "lesson": "लेसन",
    "topic": "टॉपिक", "topics": "टॉपिक्स", "chapter": "चैप्टर",
    "subject": "सब्जेक्ट", "exam": "एग्ज़ाम", "exams": "एग्ज़ाम्स",
    "test": "टेस्ट", "marks": "मार्क्स", "homework": "होमवर्क",
    "doubt": "डाउट", "doubts": "डाउट्स", "explain": "एक्सप्लेन",
    "concept": "कॉन्सेप्ट", "concepts": "कॉन्सेप्ट्स",
    "revision": "रिविज़न", "syllabus": "सिलेबस",
    "notes": "नोट्स", "note": "नोट", "home": "होम",
    "school": "स्कूल", "homeworkdone": "होमवर्क डन",
    # ---- daily verbs (bol-chaal wale) ----
    "help": "हेल्प", "try": "ट्राई", "check": "चेक",
    "start": "स्टार्ट", "stop": "स्टॉप", "wait": "वेट",
    "hold": "होल्ड", "see": "सी", "look": "लुक", "show": "शो",
    "tell": "टेल", "talk": "टॉक", "speak": "स्पीक",
    "listen": "लिसन", "understand": "अंडरस्टैंड",
    "clear": "क्लियर", "confirm": "कन्फर्म", "share": "शेयर",
    "open": "ओपन", "close": "क्लोज़", "save": "सेव",
    "delete": "डिलीट", "edit": "एडिट", "send": "सेंड",
    "receive": "रिसीव", "use": "यूज़", "using": "यूज़िंग",
    "work": "वर्क", "working": "वर्किंग", "donework": "डन वर्क",
    "thinkthink": "थिंक", "manage": "मैनेज",
    # ---- adjectives / fillers ----
    "hard": "हार्ड", "difficult": "डिफिकल्ट", "fast": "फास्ट",
    "slow": "स्लो", "right": "राइट", "wrong": "रॉन्ग",
    "actually": "एक्चुअली", "generally": "जनरली",
    "normally": "नॉर्मली", "usually": "यूज़ुअली",
    "seriously": "सीरियसली", "obviously": "ऑब्वियसली",
    "definitely": "डेफिनेटली", "problem": "प्रॉब्लम",
    "solution": "सॉल्यूशन", "idea": "आइडिया",
    "moment": "मोमिंट", "second": "सेकंड", "minute": "मिनट",
    # ---- extra tech (code-debug wale) ----
    "error": "एरर", "errors": "एरर्स", "bug": "बग", "bugs": "बग्स",
    "debug": "डिबग", "deploy": "डिप्लॉय", "frontend": "फ्रंटएंड",
    "backend": "बैकएंड", "database": "डेटाबेस", "client": "क्लाइंट",
    "login": "लॉगिन", "logout": "लॉगआउट", "password": "पासवर्ड",
    "account": "अकाउंट", "profile": "प्रोफाइल", "settings": "सेटिंग्स",
    "notification": "नोटिफिकेशन", "link": "लिंक",     "tag": "टैग",
    "tags": "टैग्स", "element": "एलिमेंट", "elements": "एलिमेंट्स",
    "attribute": "एट्रिब्यूट", "attributes": "एट्रिब्यूट्स",
    # HTML tag names as spoken (so <head> says हेड, not एच ई ए डी)
    "head": "हेड", "body": "बॉडी", "title": "टाइटल", "div": "डिव",
    "span": "स्पैन", "para": "पैरा", "image": "इमेज", "img": "इमेज",
    "script": "स्क्रिप्ट", "style": "स्टाइल", "href": "एचरेफ",
    # ---- connectors / helpers (missed = letter-spell, so keep explicit) ----
    "need": "नीड", "needs": "नीड्स", "and": "एंड", "or": "ऑर",
    "but": "बट", "because": "बिकॉज़", "with": "विद",
    "without": "विदाउट", "for": "फॉर", "from": "फ्रॉम",
    "you": "यू", "your": "योर", "we": "वी", "they": "दे",
    "this": "दिस", "that": "दैट", "what": "व्हाट", "when": "व्हेन",
    "how": "हाउ", "why": "व्हाई",
}

# Shuddh Hindi -> roz-marra bol-chaal (robotic feel ka root cause).
# LLM shuddh likh bhi de to TTS natural bolega. Longest-first apply.
SHUDDH_TO_BOLCHAAL = {
    "मैं आपकी क्या सहायता कर सकता हूँ": "बोलो, मेरी क्या हेल्प चाहिए",
    "मैं आपकी क्या सहायता कर सकती हूँ": "बोलो, मेरी क्या हेल्प चाहिए",
    "क्या सहायता कर सकता हूँ": "क्या हेल्प चाहिए",
    "सहायता": "हेल्प",
    "आवश्यकता": "ज़रूरत",
    "आवश्यक": "ज़रूरी",
    "जानकारी": "इंफो",
    "उदाहरण": "एग्ज़ाम्पल",
    "कृपया": "प्लीज़",
    "क्षण": "सेकंड",
    "निश्चित": "पक्का",
    "निश्चित रूप से": "पक्का",
    "शुभ": "अच्छा",
    "क्षमता": "पावर",
    "उपयोग": "यूज़",
    "प्रयोग": "ट्राई",
    "समस्या": "प्रॉब्लम",
    "समाधान": "सॉल्यूशन",
    "प्रश्न": "क्वेश्चन",
    "उत्तर": "आंसर",
    "शिक्षक": "टीचर",
    "विद्यार्थी": "स्टूडेंट",
    "पुस्तक": "बुक",
    "प्रतीक्षा": "वेट",
    "तुरंत": "जल्दी से",
    "शीघ्र": "जल्दी",
    "वार्तालाप": "बातचीत",
    "प्रतिक्रिया": "रिएक्शन",
    "अनुभव": "एक्सपीरियंस",
    "महत्वपूर्ण": "इम्पॉर्टेंट",
    "अत्यंत": "बहुत",
    "एवं": "और",
    "तथा": "और",
    "किंतु": "लेकिन",
    "परंतु": "लेकिन",
    "अतः": "इसलिए",
    "यथाशीघ्र": "जल्दी से",
}
_SHUDDH_RE = re.compile(
    "|".join(re.escape(k) for k in sorted(SHUDDH_TO_BOLCHAAL, key=len, reverse=True))
)


def _naturalize(text: str) -> str:
    """Shuddh Hindi -> daily bol-chaal before TTS (sounds identical in meaning)."""
    if not text:
        return text
    return _SHUDDH_RE.sub(lambda m: SHUDDH_TO_BOLCHAAL[m.group(0)], text)

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

