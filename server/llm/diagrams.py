"""Intent-based diagram generation for the visual tutor surface."""

import json
import re
import threading

DIAGRAM_MAX_ELEMENTS = 40
DIAGRAM_MAX_TEXT = 180
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
                        },
                        "required": ["id", "type", "x", "y"],
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


def should_generate(text: str, history: list[dict] | None, enabled: bool) -> bool:
    """Auto-decide: every teaching turn gets an LLM judge call.

    No keyword gate — the judge prompt itself decides whether a diagram
    materially improves THIS explanation (tool call) or not (no call).
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
    # Everything else: let the judge decide (explain html, क्या है, कैसे...).
    return True


def _step_prompt(step_text: str, topic: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are a visual teaching assistant drawing ONE step of an explanation "
                "on a shared whiteboard. DRAW only when the step has a visual structure: "
                "a process/flow, a system with 3+ parts, frontend/backend/data flow, "
                "a comparison, a timeline, or an architecture. DO NOT draw for greetings, "
                "yes/no answers, opinions, single facts, jokes, or meta talk — return no "
                "tool call for those. When drawing, return a draw_flowchart_or_diagram "
                "tool call with 3-6 concise nodes (max 6 words each) plus arrows for "
                "THIS step only. Every rectangle/ellipse/diamond MUST carry non-empty "
                "text naming the concrete thing from the STEP (tag names, file names, "
                "exact terms — never blank labels). Node ids must be unique — prefix "
                "every id with the given STEP tag. Layout is dynamic per step: top-to-bottom flow for "
                "sequences/processes (x ~80..400, y growing), side-by-side for "
                "comparisons (x spread 80..640). Shapes: rectangle = component/step, "
                "ellipse = start/end, diamond = decision, arrow = flow."
            ),
        },
        {"role": "user", "content": f"TOPIC: {topic[:200]}\nSTEP: {step_text[:600]}"},
    ]


def generate_for_step(
    key: str,
    step_text: str,
    topic: str,
    stop_evt: threading.Event,
    *,
    client,
    url: str,
    model: str,
    reasoning_effort: str,
    id_prefix: str = "w",
) -> dict | None:
    """Watcher planner: one small board delta for ONE spoken window.

    Best-effort by design — None means 'nothing drawable this step', never an
    error. Voice never waits for this; the caller tags the result with window_n
    and the client merges it on arrival.
    """
    if stop_evt.is_set() or not (step_text or "").strip():
        return None
    payload = {
        "model": model,
        "messages": _step_prompt(step_text, topic),
        "tools": [DIAGRAM_TOOL],
        "tool_choice": "auto",
        "temperature": 0.2,
        # gpt-oss spends tokens thinking before answering — 450 starves the
        # tool arguments and Groq answers 400. 800 matches the whole-turn judge.
        "max_tokens": 800,
    }
    if reasoning_effort and "gpt-oss" in model:
        payload["reasoning_effort"] = reasoning_effort
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
    # Y-offset per window so steps stack downward instead of overlapping.
    try:
        win_n = int(id_prefix.lstrip("w") or 1)
    except ValueError:
        win_n = 1
    y_off = max(0, (win_n - 1)) * 220
    for item in out:
        try:
            item["y"] = max(-2000, min(2000, float(item.get("y", 80)) + y_off))
        except (TypeError, ValueError):
            pass
    return {"elements": out}


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
                "Use logical coordinates, top-to-bottom for sequences and side-by-side for comparisons."
            ),
        },
        {"role": "user", "content": f"Recent context:\n{recent}\n\nCurrent question:\n{text}"},
    ]


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
            normalized["text"] = str(item.get("text", "")).strip()[:DIAGRAM_MAX_TEXT]
        if item_type == "arrow":
            normalized["startNodeId"] = str(item.get("startNodeId", "")).strip()[:80]
            normalized["endNodeId"] = str(item.get("endNodeId", "")).strip()[:80]
        color = str(item.get("backgroundColor", "")).strip()
        if color == "transparent" or re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            normalized["backgroundColor"] = color
        elements.append(normalized)
    nodes = {item["id"] for item in elements if item["type"] != "arrow"}
    elements = [
        item for item in elements
        if item["type"] != "arrow" or (item.get("startNodeId") in nodes and item.get("endNodeId") in nodes)
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
