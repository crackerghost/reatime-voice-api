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


def should_generate(text: str, history: list[dict] | None, enabled: bool) -> bool:
    if not enabled:
        return False
    clean_text = (text or "").strip()
    if not clean_text:
        return False
    if DIAGRAM_INTENT_RE.search(clean_text):
        return True
    if history:
        user_msgs = [m.get("content", "") for m in history if m.get("role") == "user"]
        return bool(user_msgs and len(clean_text) < 48 and DIAGRAM_INTENT_RE.search(user_msgs[-1]))
    return False


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
