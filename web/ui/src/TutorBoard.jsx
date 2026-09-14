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

const DRAW_MS = 650; // stroke-draw animation per shape

function StepDiagram({ elements, focusIds }) {
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
      const w = Number(e.width) || 180, h = Number(e.height) || 60;
      minX = Math.min(minX, Number(e.x) || 0);
      minY = Math.min(minY, Number(e.y) || 0);
      maxX = Math.max(maxX, (Number(e.x) || 0) + w);
      maxY = Math.max(maxY, (Number(e.y) || 0) + h);
    }
    if (!isFinite(minX)) return null;
    const pad = 20;
    return {
      x: minX - pad, y: minY - pad,
      w: maxX - minX + pad * 2, h: maxY - minY + pad * 2,
    };
  }, [shapes, arrows]);

  const byId = useMemo(() => new Map(shapes.map((s) => [s.id, s])), [shapes]);
  const hot = useMemo(() => new Set(focusIds || []), [focusIds]);
  // Compare steps (side=left/right): a VS divider between the columns.
  const divider = useMemo(() => {
    let lMax = -Infinity, rMin = Infinity;
    for (const s of shapes) {
      if (s.side === "left") lMax = Math.max(lMax, (Number(s.x) || 0) + (Number(s.width) || 0));
      if (s.side === "right") rMin = Math.min(rMin, Number(s.x) || 0);
    }
    return isFinite(lMax) && isFinite(rMin) && rMin > lMax ? (lMax + rMin) / 2 : null;
  }, [shapes]);
  if (!box) return null;

  const toneOf = (s) => T.tones?.[s?.tone] || T.tones.core;
  // NOTE: pathLength MUST be an attribute — in a style object React emits
  // `pathLength` (invalid CSS; the valid property is `path-length`), the
  // dash pattern then applies in user units and every box renders DOTTED.
  const drawStyleFor = (s) => {
    const tone = toneOf(s);
    return {
      fill: tone.fill,
      stroke: tone.stroke,
      strokeWidth: 2,
      strokeDasharray: 1,
      strokeDashoffset: 1,
      animation: `tutor-draw ${DRAW_MS}ms ease-out forwards`,
    };
  };
  const hotExtra = {
    stroke: T.primary,
    strokeWidth: 2.75,
    filter: `drop-shadow(0 0 7px ${T.primary}66)`,
  };

  return (
    <svg
      viewBox={`${box.x} ${box.y} ${box.w} ${box.h}`}
      className="h-auto w-full"
      role="img"
      aria-label="Lesson diagram"
    >
      <style>{`@keyframes tutor-draw { to { stroke-dashoffset: 0; } }
@keyframes tutor-fadein { from { opacity: 0; transform: translateY(3px); } to { opacity: 1; transform: none; } }
@keyframes tutor-flow { to { stroke-dashoffset: -0.105; } }
@keyframes tutor-pop { 0% { opacity: 0; transform: scale(0.92); } 60% { opacity: 1; transform: scale(1.015); } 100% { opacity: 1; transform: scale(1); } }`}</style>
      {shapes.map((s) => {
        const w = Number(s.width) || 180, h = Number(s.height) || 60;
        const label = String(s.text || "");
        const isHot = hot.has(s.id);
        // Free-floating annotation: no box, just elegant text (tone-tinted).
        if (s.type === "text")
          return (
            <g key={s.id} style={{ animation: `tutor-fadein 450ms ease both` }}>
              <text
                x={s.x} y={s.y + 20}
                fontSize={14.5} fontWeight={650}
                fill={s.tone && s.tone !== "core" ? toneOf(s).stroke : T.inkSoft}
                style={isHot ? { filter: `drop-shadow(0 0 6px ${T.primary}66)` } : undefined}
              >
                {label}
              </text>
            </g>
          );
        const fs = Math.max(11, Math.min(13.5, w / Math.max(8, label.length * 0.62)));
        const labelEl = (
          <text
            x={s.x + w / 2} y={s.y + h / 2}
            textAnchor="middle" dominantBaseline="central"
            fontSize={fs} fontWeight={600} fill={T.ink}
            style={{ animation: `tutor-fadein 400ms ease ${DRAW_MS}ms both` }}
          >
            {label.length > 40 ? label.slice(0, 39) + "…" : label}
          </text>
        );
        const gStyle = {
          ...drawStyleFor(s),
          ...(isHot ? hotExtra : null),
          transformBox: "fill-box",
          transformOrigin: "center",
          animation: `tutor-draw ${DRAW_MS}ms ease-out forwards, tutor-pop 350ms ease ${DRAW_MS}ms both`,
        };
        if (s.type === "ellipse")
          return (
            <g key={s.id}>
              <ellipse cx={s.x + w / 2} cy={s.y + h / 2} rx={w / 2} ry={h / 2} pathLength={1} style={gStyle} />
              {labelEl}
            </g>
          );
        if (s.type === "diamond") {
          const pts = `${s.x + w / 2},${s.y} ${s.x + w},${s.y + h / 2} ${s.x + w / 2},${s.y + h} ${s.x},${s.y + h / 2}`;
          return (
            <g key={s.id}>
              <polygon points={pts} pathLength={1} style={gStyle} />
              {labelEl}
            </g>
          );
        }
        return (
          <g key={s.id}>
            <rect x={s.x} y={s.y} width={w} height={h} rx={T.radius} pathLength={1} style={gStyle} />
            {labelEl}
          </g>
        );
      })}
      {divider != null && (
        <g aria-hidden="true">
          <line
            x1={divider} y1={box.y + 6} x2={divider} y2={box.y + box.h - 6}
            stroke={T.faint} strokeWidth={1.5} strokeDasharray="5 5" opacity={0.8}
          />
          <g transform={`translate(${divider}, ${box.y + box.h / 2})`}>
            <rect x={-19} y={-12} width={38} height={24} rx={12} fill={T.ink} />
            <text textAnchor="middle" dominantBaseline="central" fontSize={11} fontWeight={800} fill="#fff">
              VS
            </text>
          </g>
        </g>
      )}
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
              fill="none" stroke={T.primary} strokeWidth={2.25}
              markerEnd={`url(#${id})`}
              pathLength={1} strokeDasharray="0.06 0.045" strokeDashoffset={1}
              style={{ animation: `tutor-draw ${DRAW_MS}ms ease-out forwards, tutor-flow 1.15s linear ${DRAW_MS + 50}ms infinite` }}
            />
          </g>
        );
      })}
    </svg>
  );
}

function CodeBlock({ block, caption }) {
  const code = String(block.code || block.text || "");
  const lines = useMemo(() => code.split("\n"), [code]);
  // Karaoke: the line whose code-ish tokens best match the spoken caption
  // (what the tutor is explaining RIGHT NOW) glows. Recomputes as the
  // caption streams in — no protocol change needed.
  const hotLine = useMemo(() => {
    if (!caption) return -1;
    const cap = String(caption).toLowerCase();
    let best = -1, bestScore = 0;
    lines.forEach((ln, i) => {
      const raw = ln.match(/[A-Za-z_][A-Za-z0-9_#.]*|\d+/g) || [];
      const toks = [...new Set(raw.map((t) => t.toLowerCase()))]
        .filter((t, k) => /[0-9_#.]/.test(raw[k]) || raw[k].length >= 7 || /[A-Z]/.test(raw[k]));
      let s = 0;
      for (const t of toks) if (t.length >= 2 && cap.includes(t)) s += t.length >= 6 ? 2 : 1;
      if (s > bestScore) { bestScore = s; best = i; }
    });
    return bestScore > 0 ? best : -1;
  }, [lines, caption]);
  return (
    <div className="overflow-hidden rounded-xl shadow-sm" style={{ background: T.codeBg }}>
      <div className="flex items-center gap-2 px-3.5 py-2" style={{ borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
        <span className="h-2.5 w-2.5 rounded-full" style={{ background: T.primary }} aria-hidden="true" />
        <span className="text-[0.68rem] font-bold tracking-[0.1em] uppercase" style={{ color: T.faint }}>
          {block.language || "code"}
        </span>
        {hotLine >= 0 && (
          <span className="ml-auto text-[0.62rem] font-bold tracking-wider uppercase" style={{ color: T.primary }}>
            ◉ line {hotLine + 1}
          </span>
        )}
      </div>
      <pre
        className="overflow-x-auto py-2 font-mono text-[0.8rem] leading-6"
        style={{ color: T.codeInk }}
      >
        {lines.map((ln, i) => (
          <div
            key={i}
            className="flex whitespace-pre transition-colors duration-300"
            style={i === hotLine
              ? { background: `${T.primary}22`, boxShadow: `inset 3px 0 0 ${T.primary}` }
              : undefined}
          >
            <span className="w-9 shrink-0 pr-2 text-right select-none" style={{ color: "#475569" }}>{i + 1}</span>
            <span className="pr-3.5">{ln || " "}</span>
          </div>
        ))}
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
  steps, focus, stepIndex, onStep, followLive, onJumpLive, caption,
}) {
  const scrollRef = useRef(null);
  const penRef = useRef(null);
  const elRefs = useRef(new Map());
  const total = steps?.length || 0;

  // Viewport follows the freshly drawn element (live mode only).
  // Container-local scrolling ONLY — scrollIntoView() climbs into the
  // document and drags the whole page (notch/calendar get cut off).
  const scrollNodeIntoView = (node, align = "center") => {
    const box = scrollRef.current;
    if (!box || !node) return;
    try {
      let target;
      if (node.offsetTop !== undefined && node.offsetParent !== null) {
        target = align === "start"
          ? node.offsetTop - 8
          : node.offsetTop - box.clientHeight / 2 + node.clientHeight / 2;
      } else {
        // SVG nodes have no offsetTop — use rect math instead.
        const r = node.getBoundingClientRect(), b = box.getBoundingClientRect();
        target = box.scrollTop + (r.top - b.top)
          - (align === "start" ? 8 : box.clientHeight / 2 - r.height / 2);
      }
      if (Number.isFinite(target)) box.scrollTo({ top: Math.max(0, target), behavior: "smooth" });
    } catch { /* noop */ }
  };
  useEffect(() => {
    if (!followLive || !focus?.ids?.length) return;
    const id = focus.ids[0];
    const node = elRefs.current.get(id);
    if (node) scrollNodeIntoView(node, "center");
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
      if (node) scrollNodeIntoView(node, "start");
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

      <div ref={scrollRef} className="relative min-h-0 flex-1 overflow-y-auto overscroll-contain px-3 py-3">
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
          const isLive = si === (steps?.length || 0) - 1;
          return (
            <section key={step.key} data-step={si} className="mb-4 last:mb-1">
              <div className="mb-1.5 flex items-center gap-2">
                <span
                  className="inline-flex h-5 min-w-5 items-center justify-center rounded-full px-1.5 text-[0.62rem] font-extrabold tracking-wide text-white shadow-sm"
                  style={{ background: T.primary }}
                >
                  {si + 1}
                </span>
                <p className="text-[0.66rem] font-bold tracking-[0.14em] uppercase" style={{ color: T.muted }}>
                  Step {si + 1}{step.title ? ` · ${step.title}` : ""}
                </p>
                {isLive && (
                  <span className="ml-auto inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[0.6rem] font-bold tracking-wider uppercase" style={{ background: `${T.primary}14`, color: T.primary }}>
                    <span className="relative flex h-1.5 w-1.5">
                      <span className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-60" style={{ background: T.primary }} />
                      <span className="relative inline-flex h-1.5 w-1.5 rounded-full" style={{ background: T.primary }} />
                    </span>
                    Live
                  </span>
                )}
              </div>
              {(shapes.length > 0 || arrows.length > 0) && (
                <div
                  ref={shapes[0] ? setElRef(shapes[0].id) : undefined}
                  className="rounded-2xl bg-white p-2 shadow-[0_2px_14px_rgba(18,48,74,0.07)]"
                  style={{
                    border: `1px solid ${T.line}`,
                    backgroundImage: `radial-gradient(circle, ${T.line} 1px, transparent 1px)`,
                    backgroundSize: "22px 22px",
                  }}
                >
                  <div className="rounded-xl bg-white/85 px-1 py-1" style={{ border: `1px solid ${T.line}66` }}>
                    <StepDiagram elements={[...shapes, ...arrows]} focusIds={focus?.ids} />
                  </div>
                </div>
              )}
              {!!extras.length && (
                <div className="mt-2 flex flex-col gap-2">
                  {extras.map((b) => (
                    <div key={b.id} ref={setElRef(b.id)}>
                      {b.type === "code" && <CodeBlock block={b} caption={caption} />}
                      {b.type === "note" && <NoteBlock block={b} />}
                      {b.type === "image" && <ImageBlock block={b} />}
                    </div>
                  ))}
                </div>
              )}
            </section>
          );
        })}
      </div>
    </div>
  );
}
