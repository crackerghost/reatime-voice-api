import { useEffect, useMemo, useRef, useState } from "react";
import { Excalidraw, convertToExcalidrawElements } from "@excalidraw/excalidraw";
import "@excalidraw/excalidraw/index.css";

const DEFAULT_WIDTH = 180;
const DEFAULT_HEIGHT = 84;
const MAX_TEXT = 180;
// Devanagari is voice-only. Never draw Hindi on the board.
const DEVANAGARI_RE = /[\u0900-\u097F]+/g;

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const safeText = (value) =>
  String(value || "")
    .replace(DEVANAGARI_RE, " ")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, MAX_TEXT);

function toExcalidrawElements(items) {
  const nodes = new Map();
  const skeletons = [];

  for (const item of items || []) {
    if (!item || !item.id || !item.type) continue;
    const id = String(item.id).slice(0, 80);
    const x = Number.isFinite(item.x) ? item.x : 80;
    const y = Number.isFinite(item.y) ? item.y : 80;
    const width = clamp(Number(item.width) || DEFAULT_WIDTH, 80, 560);
    const height = clamp(Number(item.height) || DEFAULT_HEIGHT, 44, 260);
    const text = safeText(item.text);

    if (item.type === "text") {
      if (text) {
        skeletons.push({
          id,
          type: "text",
          x,
          y,
          text,
          fontSize: 18,
          fontFamily: 2,
          textAlign: "left",
          verticalAlign: "middle",
          strokeColor: "#12304a",
          backgroundColor: "transparent",
          strokeWidth: 1,
          roughness: 0,
        });
      }
      continue;
    }

    if (!["rectangle", "ellipse", "diamond"].includes(item.type)) continue;
    if (!text) continue;

    nodes.set(id, { id, type: item.type });
    skeletons.push({
      id,
      type: item.type,
      x,
      y,
      width,
      height,
      strokeColor: item.type === "diamond" ? "#ff5a5f" : "#34566b",
      backgroundColor: item.backgroundColor === "transparent" ? "transparent" : (item.backgroundColor || "#fff1f1"),
      fillStyle: "solid",
      strokeWidth: 2,
      roughness: 0,
      roundness: item.type === "rectangle" ? { type: 3 } : null,
      label: {
        text,
        fontSize: 16,
        fontFamily: 2,
        textAlign: "center",
        verticalAlign: "middle",
      },
    });
  }

  for (const item of items || []) {
    if (!item || item.type !== "arrow" || !item.id) continue;
    const start = nodes.get(String(item.startNodeId || ""));
    const end = nodes.get(String(item.endNodeId || ""));
    if (!start || !end) continue;

    skeletons.push({
      id: String(item.id).slice(0, 80),
      type: "arrow",
      start: { id: start.id, type: start.type },
      end: { id: end.id, type: end.type },
      endArrowhead: "arrow",
      strokeColor: "#ff5a5f",
      backgroundColor: "transparent",
      strokeWidth: 2,
      roughness: 0,
    });
  }

  return convertToExcalidrawElements(skeletons, { regenerateIds: false });
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
