import { useEffect, useMemo, useRef, useState } from "react";
import { Excalidraw, convertToExcalidrawElements } from "@excalidraw/excalidraw";
import "@excalidraw/excalidraw/index.css";
import { FaDiagramProject } from "react-icons/fa6";

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

  const nodePos = new Map();
  for (const s of skeletons) {
    if (s.type !== "text") nodePos.set(s.id, { x: s.x, y: s.y, w: s.width || DEFAULT_WIDTH, h: s.height || DEFAULT_HEIGHT });
  }
  for (const item of items || []) {
    if (!item || item.type !== "arrow" || !item.id) continue;
    const start = nodes.get(String(item.startNodeId || ""));
    const end = nodes.get(String(item.endNodeId || ""));
    if (!start || !end) continue;

    // Server-computed geometry (points) draws the line exactly from the
    // source box edge to the target box edge. Binding fallback only for
    // payloads that predate server geometry.
    const pts = Array.isArray(item.points) && item.points.length >= 2
      ? item.points
      : null;
    if (pts) {
      const x = Number.isFinite(item.x) ? item.x : 80;
      const y = Number.isFinite(item.y) ? item.y : 80;
      skeletons.push({
        id: String(item.id).slice(0, 80),
        type: "arrow",
        x,
        y,
        width: clamp(Number(item.width) || 10, 1, 2000),
        height: clamp(Number(item.height) || 10, 1, 2000),
        // Excalidraw points are [x, y] tuples, not {x, y} objects.
        points: pts.map((p) => [Number(p[0]) || 0, Number(p[1]) || 0]),
        endArrowhead: "arrow",
        strokeColor: "#ff5a5f",
        backgroundColor: "transparent",
        strokeWidth: 2,
        roughness: 0,
      });
      continue;
    }
    const sp = nodePos.get(start.id) || { x: 80, y: 80, w: DEFAULT_WIDTH, h: DEFAULT_HEIGHT };
    const ep = nodePos.get(end.id) || { x: 80, y: 80, w: DEFAULT_WIDTH, h: DEFAULT_HEIGHT };

    skeletons.push({
      id: String(item.id).slice(0, 80),
      type: "arrow",
      x: (sp.x + ep.x) / 2,
      y: (sp.y + ep.y) / 2,
      start: { id: start.id },
      end: { id: end.id },
      endArrowhead: "arrow",
      strokeColor: "#ff5a5f",
      backgroundColor: "transparent",
      strokeWidth: 2,
      roughness: 0,
    });
  }

  return convertToExcalidrawElements(skeletons, { regenerateIds: false }).filter(Boolean);
}

export default function DiagramWhiteboard({ diagram, focus }) {
  const [api, setApi] = useState(null);
  const fittedRef = useRef(false);
  const hadContentRef = useRef(false);
  const prevLenRef = useRef(0);
  const elements = useMemo(() => toExcalidrawElements(diagram?.elements), [diagram]);
  const byId = useMemo(() => new Map(elements.map((el) => [el.id, el])), [elements]);

  const fitView = () => {
    if (!api || !elements.length) return;
    try {
      api.scrollToContent(elements, { fitToContent: true, animate: true, duration: 350 });
    } catch {
      // Excalidraw torn down — the next board update recovers.
    }
  };

  // Audio-synced focus: when the spoken window changes, scroll to THAT step's
  // nodes (not the whole board) so diagram position matches the explanation.
  // Falls back to whole-board fit when the focused nodes aren't on screen yet.
  useEffect(() => {
    if (!api || !focus?.ids?.length || !elements.length) return;
    const targets = focus.ids.map((id) => byId.get(id)).filter(Boolean);
    if (!targets.length) return;
    try {
      api.scrollToContent(targets, { fitToContent: false, animate: true, duration: 400 });
    } catch {
      try {
        api.scrollToContent(elements, { fitToContent: true, animate: true, duration: 350 });
      } catch {
        // Excalidraw torn down — next update recovers.
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [api, focus?.tick]);

  // Re-fit when a fresh board goes from empty to non-empty. Growing boards
  // re-fit gently (no animation steal) so stacked windows never drift into
  // the blank "Scroll back to content" state seen when y offsets run away.
  useEffect(() => {
    const hasContent = elements.length > 0;
    if (hasContent && !hadContentRef.current) fittedRef.current = false;
    hadContentRef.current = hasContent;
  }, [elements.length]);

  useEffect(() => {
    if (!api || !elements.length) {
      prevLenRef.current = elements.length;
      // New turn / clear: wipe the scene so the old board never ghosts
      // behind the empty-state overlay, and arm re-fit for the next board.
      if (api && elements.length === 0 && fittedRef.current) {
        try {
          api.updateScene({ elements: [], commitToHistory: false });
        } catch {
          // Excalidraw torn down — next mount recovers.
        }
        fittedRef.current = false;
      }
      return;
    }
    const grew = elements.length > prevLenRef.current;
    prevLenRef.current = elements.length;
    void grew;
    try {
      api.updateScene({ elements, commitToHistory: false });
      if (!fittedRef.current) {
        // First board of the turn: fit everything once. Later windows are
        // positioned by the focus effect (spoken step), not whole-board fit,
        // so the viewport follows the explanation instead of shrinking.
        fittedRef.current = true;
        requestAnimationFrame(fitView);
      }
    } catch {
      // Excalidraw API torn down (unmount/strict-mode) — next update recovers.
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [api, elements]);

  const isEmpty = elements.length === 0;
  return (
    <div className="relative h-full min-h-[480px] w-full overflow-hidden [&_.excalidraw]:[--color-primary:#0f766e] [&_.excalidraw]:[--color-primary-darker:#12304a] [&_.excalidraw]:[--color-primary-light:#ccfbf1]" aria-label="Interactive visual explanation canvas">
      {/* Keep Excalidraw mounted always: unmount/remount on every turn caused
          toolbar flash + ~500ms blank frame ("frame not coming properly").
          Empty state is an overlay, the canvas frame stays stable underneath. */}
      <div className="absolute inset-0">
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
      {isEmpty && (
        <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-2 bg-white/85 text-center backdrop-blur-[1px]">
          <FaDiagramProject className="h-8 w-8 text-slate-300" />
          <p className="text-sm font-semibold text-slate-500">No visual yet</p>
          <p className="max-w-[220px] text-xs leading-5 text-slate-400">
            Ask a question and the tutor will draw here in realtime.
          </p>
        </div>
      )}
      {!isEmpty && (
        <button
          onClick={fitView}
          className="absolute top-3 right-3 z-10 rounded-full border border-slate-200 bg-white/95 px-3 py-1.5 text-xs font-semibold text-slate-600 shadow-sm backdrop-blur transition hover:bg-white hover:text-slate-900"
          title="Fit board to view"
        >
          Fit view
        </button>
      )}
    </div>
  );
}
