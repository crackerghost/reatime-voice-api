import { useEffect, useMemo, useRef, useState } from "react";
import { Excalidraw } from "@excalidraw/excalidraw";
import "@excalidraw/excalidraw/index.css";

const DEFAULT_WIDTH = 180;
const DEFAULT_HEIGHT = 84;
const MAX_TEXT = 180;

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const safeText = (value) => String(value || "").trim().slice(0, MAX_TEXT);

/* Stable numeric seed per id — id.length collides across nodes and makes
   Excalidraw render duplicates with identical roughness. */
const seedOf = (id) => {
  let h = 2166136261;
  const s = String(id);
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return Math.abs(h % 2147483647);
};

function toExcalidrawElements(items) {
  const nodes = new Map();
  const elements = [];

  for (const item of items || []) {
    if (!item || !item.id || !item.type) continue;
    const id = String(item.id).slice(0, 80);
    const x = Number.isFinite(item.x) ? item.x : 80;
    const y = Number.isFinite(item.y) ? item.y : 80;
    const width = clamp(Number(item.width) || DEFAULT_WIDTH, 80, 560);
    const height = clamp(Number(item.height) || DEFAULT_HEIGHT, 44, 260);
    const text = safeText(item.text);

    if (item.type === "arrow") continue;
    if (!["rectangle", "ellipse", "diamond", "text"].includes(item.type)) continue;

    if (item.type === "text") {
      elements.push({
        id,
        type: "text",
        x,
        y,
        width,
        height,
        angle: 0,
        text: text || " ",
        fontSize: 18,
        fontFamily: 2,
        textAlign: "left",
        verticalAlign: "middle",
        strokeColor: "#12304a",
        backgroundColor: "transparent",
        fillStyle: "solid",
        strokeWidth: 1,
        roughness: 0,
        opacity: 100,
        seed: seedOf(id),
        version: 1,
        versionNonce: seedOf(`nonce-${id}`),
        isDeleted: false,
        groupIds: [],
        frameId: null,
        index: null,
        roundness: null,
        boundElements: null,
        updated: Date.now(),
        link: null,
        locked: false,
      });
      continue;
    }

    const node = {
      id,
      type: item.type,
      x,
      y,
      width,
      height,
      angle: 0,
      strokeColor: item.type === "diamond" ? "#ff5a5f" : "#34566b",
      backgroundColor: item.backgroundColor === "transparent" ? "transparent" : (item.backgroundColor || "#fff1f1"),
      fillStyle: "solid",
      strokeWidth: 2,
      roughness: 0,
      opacity: 100,
      seed: seedOf(id),
      version: 1,
      versionNonce: seedOf(`nonce-${id}`),
      isDeleted: false,
      groupIds: [],
      frameId: null,
      roundness: item.type === "rectangle" ? { type: 3 } : null,
      boundElements: [],
      updated: Date.now(),
      link: null,
      locked: false,
    };
    nodes.set(id, node);
    elements.push(node);

    if (text) {
      const labelId = `${id}-label`;
      elements.push({
        id: labelId,
        type: "text",
        x: x + 14,
        y: y + Math.max(8, height / 2 - 12),
        width: Math.max(40, width - 28),
        height: 28,
        angle: 0,
        text,
        fontSize: 16,
        fontFamily: 2,
        textAlign: "center",
        verticalAlign: "middle",
        strokeColor: "#12304a",
        backgroundColor: "transparent",
        fillStyle: "solid",
        strokeWidth: 1,
        roughness: 0,
        opacity: 100,
        seed: labelId.length * 97,
        version: 1,
        versionNonce: labelId.length * 193,
        isDeleted: false,
        groupIds: [],
        frameId: null,
        roundness: null,
        boundElements: null,
        updated: Date.now(),
        link: null,
        locked: false,
      });
      node.boundElements.push({ type: "text", id: labelId });
    }
  }

  for (const item of items || []) {
    if (!item || item.type !== "arrow" || !item.id) continue;
    const start = nodes.get(String(item.startNodeId || ""));
    const end = nodes.get(String(item.endNodeId || ""));
    if (!start || !end) continue;

    // Direction-aware: vertical flow (stacked steps) vs horizontal (comparisons).
    const vertical = Math.abs(end.y - start.y) >= Math.abs(end.x - start.x);
    const startX = vertical ? start.x + start.width / 2 : start.x + start.width;
    const startY = vertical ? start.y + start.height : start.y + start.height / 2;
    const endX = vertical ? end.x + end.width / 2 : end.x;
    const endY = vertical ? end.y : end.y + end.height / 2;
    const arrowId = String(item.id).slice(0, 80);
    const arrow = {
      id: arrowId,
      type: "arrow",
      x: startX,
      y: startY,
      width: endX - startX,
      height: endY - startY,
      angle: 0,
      points: [[0, 0], [endX - startX, endY - startY]],
      lastCommittedPoint: null,
      startBinding: { elementId: start.id, focus: 0, gap: 8 },
      endBinding: { elementId: end.id, focus: 0, gap: 8 },
      startArrowhead: null,
      endArrowhead: "arrow",
      strokeColor: "#ff5a5f",
      backgroundColor: "transparent",
      fillStyle: "solid",
      strokeWidth: 2,
      roughness: 0,
      opacity: 100,
      seed: seedOf(arrowId),
      version: 1,
      versionNonce: seedOf(`nonce-${arrowId}`),
      isDeleted: false,
      groupIds: [],
      frameId: null,
      roundness: { type: 2 },
      boundElements: null,
      updated: Date.now(),
      link: null,
      locked: false,
    };
    elements.push(arrow);
    start.boundElements.push({ type: "arrow", id: arrow.id });
    end.boundElements.push({ type: "arrow", id: arrow.id });
  }

  return elements;
}

export default function DiagramWhiteboard({ diagram }) {
  const [api, setApi] = useState(null);
  const fittedRef = useRef(false);
  const elements = useMemo(() => toExcalidrawElements(diagram?.elements), [diagram]);

  useEffect(() => {
    fittedRef.current = false;
  }, [diagram?.elements?.length === 0]);

  useEffect(() => {
    if (!api || !elements.length) return;
    try {
      // Remote deltas must not pollute the user's undo stack.
      api.updateScene({ elements, commitToHistory: false });
      // Auto-fit once per board — never steal the viewport on later deltas
      // while the user may be panning/zooming or voice is mid-step.
      if (!fittedRef.current) {
        fittedRef.current = true;
        api.scrollToContent(elements, { fitToContent: true, animate: true, duration: 450 });
      }
    } catch {
      // Excalidraw API torn down (unmount/strict-mode) — next update recovers.
    }
  }, [api, elements]);

  return (
    <div className="h-[calc(min(72vh,760px)-4.75rem)] min-h-[420px] max-[980px]:h-[min(68vh,620px)] max-[640px]:h-[52vh] max-[640px]:min-h-[320px] [&_.excalidraw]:[--color-primary:#0f766e] [&_.excalidraw]:[--color-primary-darker:#12304a] [&_.excalidraw]:[--color-primary-light:#ccfbf1]" aria-label="Interactive visual explanation canvas">
      <Excalidraw
        excalidrawAPI={setApi}
        initialData={{
          elements,
          appState: {
            viewBackgroundColor: "#ffffff",
            currentItemFontFamily: 1,
            zenModeEnabled: false,
          },
        }}
        UIOptions={{
          canvasActions: {
            changeViewBackgroundColor: false,
            clearCanvas: true,
            export: { saveFileToDisk: true },
            loadScene: false,
            saveToActiveFile: false,
            toggleTheme: false,
          },
        }}
      />
    </div>
  );
}
