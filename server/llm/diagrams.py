"""Intent-based diagram generation for the visual tutor surface."""

import json
import re
import threading

DIAGRAM_MAX_ELEMENTS = 40
DIAGRAM_MAX_TEXT = 180
# Overload guard for default-allow gating: at most this many concurrent
# planner calls per process. Long turns spawn one thread per audio window;
# beyond this the step is skipped (best-effort — voice is unaffected).
_MAX_PARALLEL_PLANNERS = 6
_PLANNER_SEMAPHORE = threading.Semaphore(_MAX_PARALLEL_PLANNERS)
# Board text is English-only. Devanagari is voice-only.
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]+")
DIAGRAM_INTENT_RE = re.compile(
    r"(?:draw|diagram|flowchart|visuali[sz]e|mind\s*map|architecture|timeline|process|relationship|"
    r"चित्र|डायग्राम|फ्लोचार्ट|दिखाओ|समझाने के लिए|तुलना|टाइमलाइन|प्रोसेस)",
    re.IGNORECASE,
)

DIAGRAM_TOOL = {
    "type": "function",
    "function": {
        "name": "draw_flowchart_or_diagram",
        "description": "Render concise concept nodes, notes, and connecting arrows on the whiteboard.",
        "parameters": {
            "type": "object",
            "properties": {
                "elements": {
                    "type": "array",
                    "maxItems": DIAGRAM_MAX_ELEMENTS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "type": {"type": "string", "enum": ["rectangle", "ellipse", "diamond", "text", "arrow"]},
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "width": {"type": "number"},
                            "height": {"type": "number"},
                            "text": {"type": "string"},
                            "backgroundColor": {"type": "string"},
                            "startNodeId": {"type": "string"},
                            "endNodeId": {"type": "string"},
                            "arrowLabel": {"type": "string"},
                            "trigger": {
                                "type": "string",
                                "description": "1-3 exact English words from the STEP that name this element (e.g. 'computer', 'query selector'). The board draws it the moment the tutor speaks them.",
                            },
                        },
                        # x/y required only for SHAPES. Arrows are defined by
                        # startNodeId/endNodeId (server computes their geometry),
                        # and Groq 400-rejects the whole call when the LLM omits
                        # arrow coordinates — the top planner failure in prod logs.
                        "required": ["id", "type"],
                    },
                },
            },
            "required": ["elements"],
        },
    },
}


_GREETING_ONLY_RE = re.compile(
    r"^(नमस्ते|हेलो|हाय|हैलो|नमस्कार|कैसे\s+हो|क्या\s+हाल|hello|hi|hey|"
    r"good\s+(morning|afternoon|evening|night)|how\s+are\s+you|"
    r"हाँ|हां|अच्छा|ओके|ok|okay|thanks|थैंक्स?|शुक्रिया)\W*$",
    re.IGNORECASE,
)


def _board_text(value: object) -> str:
    """Strip Devanagari and collapse whitespace so the canvas stays English."""
    return DEVANAGARI_RE.sub(" ", str(value or "")).strip()


def should_generate(text: str, history: list[dict] | None, enabled: bool) -> bool:
    """Auto-decide: every teaching turn gets an LLM judge call.

    DIAGRAM_GATE=auto (default, paid tier): default-allow, judge decides.
    DIAGRAM_GATE=explicit (dev tier, 8k TPM): only explicit draw-intent
    ("draw/diagram/दिखाओ...") triggers the planner, saving ~10 Groq calls/turn.
    Voice never waits: the judge runs parallel to TTS, diagram pops when ready.
    Only pure greetings / tiny acks are skipped to save the extra LLM call.
    """
    if not enabled:
        return False
    clean_text = (text or "").strip()
    if not clean_text or len(clean_text) < 8:
        return False
    if _GREETING_ONLY_RE.search(clean_text):
        return False
    # Explicit ask always wins (no LLM judgement needed to trigger).
    if DIAGRAM_INTENT_RE.search(clean_text):
        return True
    import os as _os
    if _os.environ.get("DIAGRAM_GATE", "auto").strip().lower() == "explicit":
        return False
    # Everything else: let the judge decide (explain html, क्या है, कैसे...).
    return True


def _step_prompt(step_text: str, topic: str, tag: str = "w") -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are a visual teaching assistant drawing ONE step of an explanation "
                "on a shared whiteboard. DEFAULT TO DRAWING: any explanation with "
                "parts, steps, sequence, cause-effect, comparison, timeline, "
                "hierarchy, a system, a definition with 2+ components, or how "
                "something works deserves a board — processes, frontend/backend/"
                "data flow, architectures included. When in doubt, DRAW; a visual "
                "almost always helps grounding. Return no tool call ONLY for pure "
                "greetings, bare yes/no answers with no explanation, unstructured "
                "opinions, jokes, or meta talk. When drawing, return a draw_flowchart_or_diagram "
                "tool call with 3-6 nodes plus arrows for THIS step only. Labels may "
                "explain, not just name — up to ~15 words per node when the meaning "
                "needs it (e.g. 'h1-h6 tags: headings, h1 biggest'). Size width to "
                "fit the text (long text = width 300-500). Every rectangle/ellipse/"
                "diamond MUST carry non-empty "
                "text naming the concrete thing from the STEP (tag names, file names, "
                "exact terms — never blank labels). Language rule: the BOARD is "
                "always ENGLISH — short English labels (max 6 words), code/tag/ "
                "attribute/file names exactly as-is (HTML, h1, href, index.html). "
                "Never Devanagari on the board; Hindi is voice-only. "
                "Node ids: ASCII ONLY (a-z, 0-9, hyphen), prefixed with the given "
                "TAG (e.g. TAG-n1, TAG-n2). Never Devanagari or invented words in "
                "ids. Every element ALSO gets a trigger: the 1-3 exact English "
                "words from the STEP that name it (label 'Computer' -> trigger "
                "'computer'; 'query selector' -> 'query selector'). The board "
                "draws each element the instant the tutor speaks its trigger, "
                "so triggers must be words the STEP actually says. Shapes give x/y/width/height; ARROWS give ONLY id, type, "
                "startNodeId, endNodeId — never x/y on arrows. Layout is dynamic "
                "per step: top-to-bottom flow for sequences/processes "
                "(x ~80..400, y growing), side-by-side for comparisons "
                "(x spread 80..640). Shapes: rectangle = component/step, "
                "ellipse = start/end, diamond = decision, arrow = flow."
            ),
        },
        {"role": "user", "content": f"TAG: {tag}\nTOPIC: {topic[:200]}\nSTEP: {step_text[:600]}"},
    ]


# Production policy: DEFAULT-ALLOW. Any explaining counts as drawable until
# proven otherwise — the LLM judge is far better at "does this need a
# visual?" than any keyword list. This regex blocks ONLY windows that are
# certainly not explanations (greetings, praise, acks, goodbyes).
_NONVISUAL_RE = re.compile(
    r"^(नमस्ते|नमस्कार|हेलो|हाय|हैलो|hello|hi|hey|thanks|थैंक्स?|शुक्रिया|"
    r"धन्यवाद|ओके|ok(ay)?|अच्छा|हाँ|हां|yes|no|नहीं|bye|बाय|अलविदा|"
    r"good\s+(morning|afternoon|evening|night)|शुभ\s+(प्रभात|रात्रि)|"
    r"बिल्कुल सही|बहुत बढ़िया|शाबाश|congrats|welcome|वेलकम)"
    r"[!।.\s?]*$",
    re.IGNORECASE,
)


# Dynamic course layer: the course compiler fills these per course/module.
# The static gate above stays fully generic. Keywords are indexed in BOTH
# scripts because the gate reads raw LLM text (Devanagari) while authors
# write Latin: each Latin keyword also indexes its phonetic rendering, and
# every glossary key AND value is indexed (values are the exact Devanagari
# the tutor actually speaks, e.g. फोटोसिन्थेसिस).
_COURSE_KEYWORDS: frozenset = frozenset()


def set_course_keywords(words) -> None:
    """Install this course's domain keywords (feeds the visual gate)."""
    from server.speech.normalization import _phonetic
    forms: set[str] = set()
    for w in (words or []):
        w = str(w).strip()
        if not w:
            continue
        forms.add(w.lower())
        if re.search(r"[A-Za-z]", w):
            try:
                forms.add(_phonetic(w))
            except Exception:  # noqa: BLE001 — gate must never break
                pass
    global _COURSE_KEYWORDS
    _COURSE_KEYWORDS = frozenset(f for f in forms if len(f) > 2)


def set_course_context(glossary: dict | None = None, keywords=None) -> None:
    """One call the roadmap integration makes per module/topic.

    Glossary values (exact Devanagari terms) join the keyword index, so a
    window saying फोटोसिन्थेसिस matches even though the author wrote
    'photosynthesis'. Also installs the TTS glossary overlay.
    """
    from server.speech import normalization as _norm
    _norm.set_course_glossary(glossary)
    forms: set[str] = set()
    for w in (keywords or []):
        w = str(w).strip()
        if w:
            forms.add(w.lower())
    for k, v in (glossary or {}).items():
        if str(k).strip():
            forms.add(str(k).lower())
        if str(v).strip():
            forms.add(str(v))
    set_course_keywords(forms)


def is_visual_step(step_text: str) -> bool:
    """Default-allow: every substantive window earns a planner call.

    Only obvious non-explanations are filtered; the LLM judge makes the
    final draw/no-draw call per step. Course keywords now act as a recall
    BOOST (log boost below) rather than a requirement.
    """
    text = (step_text or "").strip()
    if len(text) < 24:
        return False
    if _NONVISUAL_RE.search(text):
        return False
    return True


def generate_for_step(
    key: str,
    step_text: str,
    topic: str,
    stop_evt: threading.Event,
    *,
    client,
    url: str,
    model: str,
    reasoning_effort: str = "",
    max_tokens: int = 700,
    id_prefix: str = "w",
) -> dict | None:
    """Watcher planner: one small board delta for ONE spoken window.

    Best-effort by design — None means 'nothing drawable this step', never an
    error. Voice never waits for this; the caller tags the result with window_n
    and the client merges it on arrival.

    Latency design: NO reasoning param is ever sent here (a 3-6 node board
    needs zero thinking; gpt-oss thinking is the biggest chunk of the old
    2-5 s per-window cost). reasoning_effort is accepted for backward
    compat but ignored. max_tokens is env-tunable (DIAGRAM_MAX_TOKENS).

    Overload guard: at most _MAX_PARALLEL_PLANNERS concurrent Groq calls;
    beyond that the step is skipped (best-effort — voice is unaffected).
    With default-allow gating every window calls the planner, so one long
    turn must not open unbounded parallel requests.
    """
    if stop_evt.is_set() or not (step_text or "").strip():
        return None
    if not is_visual_step(step_text):
        return None
    if not _PLANNER_SEMAPHORE.acquire(blocking=False):
        return None
    try:
        payload = {
            "model": model,
            "messages": _step_prompt(step_text, topic, id_prefix),
            "tools": [DIAGRAM_TOOL],
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        # Deliberately no reasoning_effort: see docstring.
        response = None
        try:
            response = client.post(
                url,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
        except Exception as e:
            # Log the body once — Groq 400s carry the real reason (bad tool
            # payload, token budget, model capability). Voice is unaffected.
            try:
                body = response.text[:300] if response is not None else ""
            except Exception:
                body = ""
            import logging as _logging
            _logging.getLogger("voice_api").warning("Diagram planner skipped (%s) %s", e, body)
            return None
        message = (response.json().get("choices") or [{}])[0].get("message") or {}
        arguments = None
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            if function.get("name") == "draw_flowchart_or_diagram":
                arguments = function.get("arguments")
                break
        if not arguments:
            return None
        try:
            diagram = normalize(json.loads(arguments))
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if not diagram or stop_evt.is_set():
            return None
        # Namespace ids per window so deltas merge without collisions.
        nodes = {}
        out = []
        for item in diagram["elements"]:
            old_id = item["id"]
            new_id = f"{id_prefix}-{old_id}"[:80]
            nodes[old_id] = new_id
            item["id"] = new_id
            out.append(item)
        for item in out:
            if item["type"] == "arrow":
                if item.get("startNodeId") in nodes:
                    item["startNodeId"] = nodes[item["startNodeId"]]
                if item.get("endNodeId") in nodes:
                    item["endNodeId"] = nodes[item["endNodeId"]]
        # Blank-label nodes render as empty boxes — worse than no board. Drop
        # shapes with no text, then re-drop arrows left dangling by that.
        out = [it for it in out
               if it["type"] == "arrow" or str(it.get("text", "")).strip()]
        keep = {it["id"] for it in out if it["type"] != "arrow"}
        out = [it for it in out
               if it["type"] != "arrow"
               or (it.get("startNodeId") in keep and it.get("endNodeId") in keep)]
        if not any(it["type"] != "arrow" for it in out):
            return None
        # Deterministic layout: model coordinates are NOT trusted (prod boards
        # showed boxes piled on top of each other). Each window gets its own
        # band; shapes stack top-to-bottom inside it, sized to their label, so
        # batches can never overlap and paced reveal always draws downward.
        # Wrap every 4 windows into a new column to bound viewport growth.
        try:
            win_n = int(id_prefix.lstrip("w") or 1)
        except ValueError:
            win_n = 1
        row = (win_n - 1) % 4
        col = (win_n - 1) // 4
        base_x = 80 + col * 560
        base_y = 80 + row * 950
        shapes = [it for it in out if it["type"] != "arrow"]
        try:
            shapes.sort(key=lambda it: (float(it.get("y", 80)), float(it.get("x", 80))))
        except (TypeError, ValueError):
            pass
        cy = float(base_y)
        for item in shapes:
            label = str(item.get("text", ""))
            item["width"] = max(180, min(460, 140 + 8 * len(label)))
            item["height"] = 96 if item["type"] == "diamond" else 84
            item["x"] = max(-2000, min(2000, float(base_x)))
            item["y"] = max(-2000, min(2000, cy))
            cy += float(item["height"]) + 66
        boxes = {
            it["id"]: (it["x"], it["y"], it["width"], it["height"]) for it in shapes
        }
        for item in out:
            if item["type"] == "arrow":
                _arrow_geometry(item, boxes)
        return {"elements": out}
    finally:
        _PLANNER_SEMAPHORE.release()



def _prompt(text: str, history: list[dict]) -> list[dict]:
    recent = "\n".join(
        f"{m.get('role', 'user')}: {str(m.get('content', ''))[:500]}"
        for m in history[-4:]
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a visual teaching assistant. Decide whether a diagram materially improves "
                "this explanation. If yes, call draw_flowchart_or_diagram. If no, return no tool call. "
                "Use 3-10 concise nodes, max 6 words per node, and connect related nodes with arrows. "
                "Use logical coordinates, top-to-bottom for sequences and side-by-side for comparisons. "
                "Board text is always English — short English labels, code/tag/file names exactly "
                "as-is. Never Devanagari on the board; Hindi is voice-only."
            ),
        },
        {"role": "user", "content": f"Recent context:\n{recent}\n\nCurrent question:\n{text}"},
    ]


def _arrow_geometry(item: dict, boxes: dict[str, tuple]) -> None:
    """Set an arrow's x/y/width/height/points from its endpoint boxes.

    Edge-to-edge line (bottom→top for downward flow, right→left otherwise)
    so arrows always touch both boxes. No-op when an endpoint is unknown.
    """
    s = boxes.get(item.get("startNodeId", ""))
    e = boxes.get(item.get("endNodeId", ""))
    if not s or not e:
        return
    scx, s_bot, s_right, scy = s[0] + s[2] / 2, s[1] + s[3], s[0] + s[2], s[1] + s[3] / 2
    ecx, e_top, e_left, ecy = e[0] + e[2] / 2, e[1], e[0], e[1] + e[3] / 2
    if e[1] >= s[1] + s[3] - 10:
        p1, p2 = (scx, s_bot), (ecx, e_top)      # flow downward
    elif e[0] >= s[0] + s[2] - 10:
        p1, p2 = (s_right, scy), (e_left, ecy)    # flow rightward
    else:
        p1, p2 = (scx, s_bot), (ecx, e_top)      # fallback: downward
    x, y = min(p1[0], p2[0]), min(p1[1], p2[1])
    item["x"], item["y"] = x, y
    item["width"] = max(1, abs(p2[0] - p1[0]))
    item["height"] = max(1, abs(p2[1] - p1[1]))
    item["points"] = [[p1[0] - x, p1[1] - y], [p2[0] - x, p2[1] - y]]


def normalize(raw: object) -> dict | None:
    if not isinstance(raw, dict) or not isinstance(raw.get("elements"), list):
        return None
    allowed = {"rectangle", "ellipse", "diamond", "text", "arrow"}
    elements = []
    seen = set()
    for item in raw["elements"][:DIAGRAM_MAX_ELEMENTS]:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id", "")).strip()[:80]
        item_type = str(item.get("type", "")).strip()
        if not item_id or item_id in seen or item_type not in allowed:
            continue
        seen.add(item_id)
        try:
            x = max(-2000, min(2000, float(item.get("x", 80))))
            y = max(-2000, min(2000, float(item.get("y", 80))))
        except (TypeError, ValueError):
            x, y = 80.0, 80.0
        normalized = {"id": item_id, "type": item_type, "x": x, "y": y}
        for key, default in (("width", 180), ("height", 84)):
            try:
                normalized[key] = max(40, min(600, float(item.get(key, default))))
            except (TypeError, ValueError):
                normalized[key] = default
        if item_type in {"rectangle", "ellipse", "diamond", "text"}:
            normalized["text"] = _board_text(item.get("text", ""))[:DIAGRAM_MAX_TEXT]
            trigger = re.sub(r"[^A-Za-z0-9 ]+", "", str(item.get("trigger", ""))).strip()[:60]
            if trigger:
                normalized["trigger"] = trigger
                # Bilingual sync: the client matches triggers against the SPOKEN
                # caption, which is Devanagari-only ("क्वेरी सेलेक्टर"), while
                # the LLM returns English ("query selector"). Precompute the
                # spoken form so App.jsx can match without transliterating.
                try:
                    from server.speech.normalization import _devanagari_only as _to_hi
                    hi = _to_hi(trigger).strip()[:60]
                    if hi and hi.lower() != trigger.lower():
                        normalized["trigger_hi"] = hi
                except Exception:
                    pass
        if item_type == "arrow":
            normalized["startNodeId"] = str(item.get("startNodeId", "")).strip()[:80]
            normalized["endNodeId"] = str(item.get("endNodeId", "")).strip()[:80]
        color = str(item.get("backgroundColor", "")).strip()
        if color == "transparent" or re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            normalized["backgroundColor"] = color
        elements.append(normalized)
    boxes = {
        item["id"]: (item["x"], item["y"], item["width"], item["height"])
        for item in elements
        if item["type"] != "arrow" and str(item.get("text", "")).strip()
    }
    for item in elements:
        # Arrow geometry is computed HERE from the endpoint boxes — never
        # trusted from the LLM (it omits arrow x/y, which used to pile every
        # arrow at 80,80 as floating lines detached from the boxes).
        if item["type"] == "arrow":
            _arrow_geometry(item, boxes)
    nodes = set(boxes)
    elements = [
        item for item in elements
        if (item["type"] != "arrow" and str(item.get("text", "")).strip())
        or (
            item["type"] == "arrow"
            and item.get("startNodeId") in nodes
            and item.get("endNodeId") in nodes
        )
    ]
    return {"elements": elements} if elements else None


def generate(
    key: str,
    text: str,
    history: list[dict],
    stop_evt: threading.Event,
    *,
    client,
    url: str,
    model: str,
    reasoning_effort: str,
) -> dict | None:
    if stop_evt.is_set():
        return None
    payload = {
        "model": model,
        "messages": _prompt(text, history),
        "tools": [DIAGRAM_TOOL],
        "tool_choice": "auto",
        "temperature": 0.2,
        "max_tokens": 900,
    }
    if reasoning_effort and "gpt-oss" in model:
        payload["reasoning_effort"] = reasoning_effort
    response = client.post(
        url,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
    )
    response.raise_for_status()
    message = (response.json().get("choices") or [{}])[0].get("message") or {}
    arguments = None
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        if function.get("name") == "draw_flowchart_or_diagram":
            arguments = function.get("arguments")
            break
    if not arguments:
        return None
    try:
        diagram = normalize(json.loads(arguments))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return diagram if diagram and not stop_evt.is_set() else None
