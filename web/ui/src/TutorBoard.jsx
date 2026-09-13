import { useEffect, useMemo, useRef } from "react";
import { FaChevronLeft, FaChevronRight, FaDiagramProject, FaSatelliteDish } from "react-icons/fa6";
import { BOARD_THEME as T } from "./boardTheme.js";

/* TutorBoard — the lesson feed. A persistent, never-erased step timeline:
   each step is a section (diagram SVG + code/note/image blocks) revealed
   progressively as the tutor speaks. Themed strictly from boardTheme.js.

   Props:
   - steps: [{ key, title, elements: [revealed...] }] in lesson order.
   - focus: { stepKey, ids:[...], tick } — latest drawn element; the pen
     glides there and the viewport follows (when followLive).
   - stepIndex, onStep(i): controlled step navigation (back/forward).
   - followLive, onJumpLive(): resume auto-follow at the newest step.
*/

const DRAW_MS = 900; // stroke-draw animation per shape

function StepDiagram({ elements }) {
  const shapes = useMemo(
    () => (elements || []).filter((e) => e && e.type !== "arrow" && (e.text || "").trim()),
    [elements],
  );
  const arrows = useMemo(
    () => (elements || []).filter((e) => e && e.type === "arrow" && Array.isArray(e.points)),
    [elements],
  );
  const box = useMemo(() => {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const e of [...shapes, ...arrows]) {
      const w = Number(e.width) || 180, h = Number(e.height) || 84;
      minX = Math.min(minX, Number(e.x) || 0);
      minY = Math.min(minY, Number(e.y) || 0);
      maxX = Math.max(maxX, (Number(e.x) || 0) + w);
      maxY = Math.max(maxY, (Number(e.y) || 0) + h);
    }
    if (!isFinite(minX)) return null;
    const pad = 24;
    return {
      x: minX - pad, y: minY - pad,
      w: maxX - minX + pad * 2, h: maxY - minY + pad * 2,
    };
  }, [shapes, arrows]);

  const byId = useMemo(() => new Map(shapes.map((s) => [s.id, s])), [shapes]);
  if (!box) return null;

  const drawStyle = {
    fill: T.primarySoft,
    stroke: T.ink,
    strokeWidth: 2.5,
    pathLength: 1,
    strokeDasharray: 1,
    strokeDashoffset: 1,
    animation: `tutor-draw ${DRAW_MS}ms ease-out forwards`,
  };

  return (
    <svg
      viewBox={`${box.x} ${box.y} ${box.w} ${box.h}`}
      className="h-auto w-full"
      role="img"
      aria-label="Lesson diagram"
    >
      <style>{`@keyframes tutor-draw { to { stroke-dashoffset: 0; } }`}</style>
      {shapes.map((s) => {
        const w = Number(s.width) || 180, h = Number(s.height) || 84;
        const label = String(s.text || "");
        const fs = Math.max(12, Math.min(17, w / Math.max(8, label.length * 0.62)));
        const labelEl = (
          <text
            x={s.x + w / 2} y={s.y + h / 2}
            textAnchor="middle" dominantBaseline="central"
            fontSize={fs} fontWeight={600} fill={T.ink}
            style={{ animation: `tutor-fadein 500ms ease ${DRAW_MS}ms both` }}
          >
            {label.length > 42 ? label.slice(0, 41) + "…" : label}
          </text>
        );
        if (s.type === "ellipse")
          return (
            <g key={s.id}>
              <ellipse cx={s.x + w / 2} cy={s.y + h / 2} rx={w / 2} ry={h / 2} style={drawStyle} />
              {labelEl}
            </g>
          );
        if (s.type === "diamond") {
          const pts = `${s.x + w / 2},${s.y} ${s.x + w},${s.y + h / 2} ${s.x + w / 2},${s.y + h} ${s.x},${s.y + h / 2}`;
          return (
            <g key={s.id}>
              <polygon points={pts} style={{ ...drawStyle, stroke: T.primary }} />
              {labelEl}
            </g>
          );
        }
        return (
          <g key={s.id}>
            <rect x={s.x} y={s.y} width={w} height={h} rx={T.radius} style={drawStyle} />
            {labelEl}
          </g>
        );
      })}
      {arrows.map((a) => {
        if (!byId.get(String(a.startNodeId || "")) || !byId.get(String(a.endNodeId || ""))) return null;
        const [[x1, y1], [x2, y2]] = a.points;
        const id = `ah-${String(a.id).replace(/[^a-zA-Z0-9_-]/g, "")}`;
        return (
          <g key={a.id}>
            <defs>
              <marker id={id} markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
                <path d="M0,0 L8,4.5 L0,9 Z" fill={T.primary} />
              </marker>
            </defs>
            <polyline
              points={`${a.x + x1},${a.y + y1} ${a.x + x2},${a.y + y2}`}
              fill="none" stroke={T.primary} strokeWidth={2.5}
              markerEnd={`url(#${id})`}
              pathLength={1} strokeDasharray={1} strokeDashoffset={1}
              style={{ animation: `tutor-draw ${DRAW_MS}ms ease-out forwards` }}
            />
          </g>
        );
      })}
      <style>{`@keyframes tutor-fadein { from { opacity: 0; } to { opacity: 1; } }`}</style>
    </svg>
  );
}

function CodeBlock({ block }) {
  const code = String(block.code || block.text || "");
  return (
    <div className="overflow-hidden rounded-xl shadow-sm" style={{ background: T.codeBg }}>
      <div className="flex items-center gap-2 px-3.5 py-2" style={{ borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
        <span className="h-2.5 w-2.5 rounded-full" style={{ background: T.primary }} aria-hidden="true" />
        <span className="text-[0.68rem] font-bold tracking-[0.1em] uppercase" style={{ color: T.faint }}>
          {block.language || "code"}
        </span>
      </div>
      <pre
        className="overflow-x-auto px-3.5 py-3 font-mono text-[0.8rem] leading-6 whitespace-pre"
        style={{ color: T.codeInk }}
      >
        {code}
      </pre>
    </div>
  );
}

function NoteBlock({ block }) {
  return (
    <div
      className="rounded-xl px-3.5 py-2.5 text-[0.83rem] leading-6 shadow-sm"
      style={{ background: T.tealSoft, color: T.ink, borderLeft: `4px solid ${T.teal}` }}
    >
      {String(block.text || "")}
    </div>
  );
}

function ImageBlock({ block }) {
  if (!block.src) {
    return (
      <div
        className="flex h-36 animate-pulse flex-col items-center justify-center gap-2 rounded-xl"
        style={{ background: "#f1f5f9" }}
        aria-label="Image loading"
      >
        <FaSatelliteDish className="h-5 w-5" style={{ color: T.faint }} />
        <p className="text-xs font-medium" style={{ color: T.muted }}>Finding a picture…</p>
      </div>
    );
  }
  return (
    <figure className="overflow-hidden rounded-xl bg-white shadow-sm" style={{ border: `1px solid ${T.line}` }}>
      <img src={block.src} alt={block.alt || "Lesson image"} className="max-h-72 w-full object-cover" loading="lazy" />
      {block.credit && (
        <figcaption className="px-3 py-1.5 text-[0.65rem]" style={{ color: T.faint }}>
          📷 {block.credit}
        </figcaption>
      )}
    </figure>
  );
}

export default function TutorBoard({
  steps, focus, stepIndex, onStep, followLive, onJumpLive,
}) {
  const scrollRef = useRef(null);
  const penRef = useRef(null);
  const elRefs = useRef(new Map());
  const total = steps?.length || 0;

  // Viewport follows the freshly drawn element (live mode only).
  useEffect(() => {
    if (!followLive || !focus?.ids?.length) return;
    const id = focus.ids[0];
    const node = elRefs.current.get(id);
    if (node && typeof node.scrollIntoView === "function") {
      try {
        node.scrollIntoView({ behavior: "smooth", block: "center" });
      } catch { /* noop */ }
    }
    // Glide the tutor pen to the drawn element.
    const pen = penRef.current, box = scrollRef.current;
    if (pen && box && node) {
      try {
        const r = node.getBoundingClientRect(), b = box.getBoundingClientRect();
        pen.style.opacity = "1";
        pen.style.transform = `translate(${r.left - b.left + r.width / 2}px, ${r.top - b.top - 6}px)`;
      } catch { /* noop */ }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focus?.tick]);

  const gotoStep = (i) => {
    const clamped = Math.max(0, Math.min(total - 1, i));
    onStep && onStep(clamped);
    requestAnimationFrame(() => {
      const node = scrollRef.current?.querySelector?.(`[data-step="${clamped}"]`);
      if (node && typeof node.scrollIntoView === "function") {
        try {
          node.scrollIntoView({ behavior: "smooth", block: "start" });
        } catch { /* noop */ }
      }
    });
  };

  const setElRef = (id) => (node) => {
    if (!id) return;
    if (node) elRefs.current.set(id, node);
    else elRefs.current.delete(id);
  };

  return (
    <div className="relative flex h-full min-h-[480px] w-full flex-col overflow-hidden bg-white" aria-label="Lesson board">
      {/* Step navigator: board never erases — walk back and forward. */}
      <div
        className="z-10 flex shrink-0 items-center gap-2 border-b px-3 py-2"
        style={{ borderColor: T.line, background: "rgba(255,255,255,0.92)", backdropFilter: "blur(6px)" }}
      >
        <button
          onClick={() => gotoStep((stepIndex || 0) - 1)}
          disabled={!total || (stepIndex || 0) <= 0}
          className="inline-flex h-7 w-7 items-center justify-center rounded-full text-slate-500 transition hover:bg-slate-100 disabled:opacity-30"
          aria-label="Previous step"
        >
          <FaChevronLeft className="h-3 w-3" />
        </button>
        <span className="min-w-0 flex-1 truncate text-center text-[0.72rem] font-bold tracking-wide text-slate-500">
          {total ? `STEP ${Math.min((stepIndex || 0) + 1, total)} / ${total}` : "BOARD"}
        </span>
        <button
          onClick={() => gotoStep((stepIndex || 0) + 1)}
          disabled={!total || (stepIndex || 0) >= total - 1}
          className="inline-flex h-7 w-7 items-center justify-center rounded-full text-slate-500 transition hover:bg-slate-100 disabled:opacity-30"
          aria-label="Next step"
        >
          <FaChevronRight className="h-3 w-3" />
        </button>
        {!followLive && total > 0 && (
          <button
            onClick={() => onJumpLive && onJumpLive()}
            className="rounded-full px-2.5 py-1 text-[0.68rem] font-bold text-white shadow-sm transition hover:brightness-95"
            style={{ background: T.primary }}
          >
            ● Live
          </button>
        )}
      </div>

      {/* Progress hairline */}
      <div className="h-0.5 w-full shrink-0 bg-slate-100" aria-hidden="true">
        <div
          className="h-full transition-all duration-500"
          style={{
            width: total ? `${(((stepIndex || 0) + 1) / total) * 100}%` : "0%",
            background: T.primary,
          }}
        />
      </div>

      <div ref={scrollRef} className="relative min-h-0 flex-1 overflow-y-auto px-3 py-3">
        {/* Tutor pen: glides to each freshly drawn element. */}
        <div
          ref={penRef}
          className="pointer-events-none absolute top-0 left-0 z-10 h-3 w-3 rounded-full opacity-0 transition-all duration-700 ease-out"
          style={{ background: T.primary, boxShadow: `0 0 0 4px ${T.primary}33, 0 0 12px ${T.primary}` }}
          aria-hidden="true"
        />
        {!total && (
          <div className="flex h-full min-h-[320px] flex-col items-center justify-center gap-2 text-center">
            <FaDiagramProject className="h-8 w-8 text-slate-300" />
            <p className="text-sm font-semibold text-slate-500">No visual yet</p>
            <p className="max-w-[220px] text-xs leading-5 text-slate-400">
              Ask a question and the tutor will draw here in realtime.
            </p>
          </div>
        )}
        {steps?.map((step, si) => {
          const shapes = (step.elements || []).filter((e) => e && e.type !== "arrow" && e.type !== "code" && e.type !== "note" && e.type !== "image");
          const extras = (step.elements || []).filter((e) => e && (e.type === "code" || e.type === "note" || e.type === "image"));
          const arrows = (step.elements || []).filter((e) => e && e.type === "arrow");
          return (
            <section key={step.key} data-step={si} className="mb-4 last:mb-1">
              <p className="mb-1.5 text-[0.62rem] font-bold tracking-[0.14em] uppercase" style={{ color: T.primary }}>
                Step {si + 1}{step.title ? ` · ${step.title}` : ""}
              </p>
              {(shapes.length > 0 || arrows.length > 0) && (
                <div
                  ref={shapes[0] ? setElRef(shapes[0].id) : undefined}
                  className="rounded-2xl bg-white p-1 shadow-sm"
                  style={{ border: `1px solid ${T.line}` }}
                >
                  <StepDiagram elements={[...shapes, ...arrows]} />
                </div>
              )}
              <div className="mt-2 flex flex-col gap-2">
                {extras.map((b) => (
                  <div key={b.id} ref={setElRef(b.id)}>
                    {b.type === "code" && <CodeBlock block={b} />}
                    {b.type === "note" && <NoteBlock block={b} />}
                    {b.type === "image" && <ImageBlock block={b} />}
                  </div>
                ))}
              </div>
            </section>
          );
        })}
      </div>
    </div>
  );
}
