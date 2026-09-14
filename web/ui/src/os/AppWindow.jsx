import { useRef, useState, useEffect } from "react";

/* Shared macOS window: traffic lights, drag-by-titlebar, 8-handle edge +
   corner resize, green zoom = fullscreen. Geometry is owned by the parent
   (per-app memory); null geom = centered default at 50% height, cascaded so
   stacked windows keep their title bars visible.
   Drag uses pointer capture on the title bar + a 4px threshold, so a press
   starting anywhere on the bar (even on a traffic button) drags on move and
   still clicks when released in place.
   Split-screen: dragging the title bar to a screen edge/corner previews a
   tile (halves on sides, quarters in corners, fullscreen on top) and snaps
   on release — like a real OS. Dragging a snapped tile floats it again. */
export default function AppWindow({
  title, geom, onGeom, maximized, cascade = 0, leaving, hideChrome,
  parked, parkIndex = 0,
  snap, onSnap, onUnsnap, onSnapPreview,
  onClose, onMin, onMax, onFocus, children,
}) {
  const winRef = useRef(null);
  const MIN_W = 480;
  const MIN_H = 340;
  // Tile boxes mirror SNAP_BOXES in App.jsx (same gap math) so the drag
  // preview lands exactly where the window will.
  const GAP = 8;
  const TILE_BOXES = {
    left: { left: GAP, top: GAP, width: `calc(50% - ${GAP * 1.5}px)`, height: `calc(100% - ${GAP * 2}px)` },
    right: { left: `calc(50% + ${GAP / 2}px)`, top: GAP, width: `calc(50% - ${GAP * 1.5}px)`, height: `calc(100% - ${GAP * 2}px)` },
    tl: { left: GAP, top: GAP, width: `calc(50% - ${GAP * 1.5}px)`, height: `calc(50% - ${GAP * 1.5}px)` },
    tr: { left: `calc(50% + ${GAP / 2}px)`, top: GAP, width: `calc(50% - ${GAP * 1.5}px)`, height: `calc(50% - ${GAP * 1.5}px)` },
    bl: { left: GAP, top: `calc(50% + ${GAP / 2}px)`, width: `calc(50% - ${GAP * 1.5}px)`, height: `calc(50% - ${GAP * 1.5}px)` },
    br: { left: `calc(50% + ${GAP / 2}px)`, top: `calc(50% + ${GAP / 2}px)`, width: `calc(50% - ${GAP * 1.5}px)`, height: `calc(50% - ${GAP * 1.5}px)` },
  };
  // Ease animated geometry changes: fullscreen zoom, tile snaps and
  // desktop-park spread (drag/resize must still track the pointer 1:1).
  const [gliding, setGliding] = useState(false);
  const firstMount = useRef(true);
  useEffect(() => {
    if (firstMount.current) {
      firstMount.current = false;
      return;
    }
    setGliding(true);
    const t = setTimeout(() => setGliding(false), 560);
    return () => clearTimeout(t);
  }, [maximized, parked, snap]);

  // Default (never dragged/resized): 70% x 70% window, centered, cascaded
  // so stacked windows keep their title bars visible.
  // Snapped (split-screen tile): fixed tile box, always above floating geom.
  // Parked (desktop spread): a clickable title-bar strip tucked under the
  // menu bar — always hittable, fanned per window.
  const box = maximized || !geom
    ? maximized
      ? { left: 0, top: 0, width: "100%", height: "100%" }
      : parked
        ? { left: `${4 + parkIndex * 8}%`, top: 2, width: "30%", height: 46 }
        : snap && TILE_BOXES[snap]
          ? { ...TILE_BOXES[snap] }
          : {
            left: `calc(15% + ${cascade * 36}px)`,
            top: `calc(15% + ${cascade * 44}px)`,
            width: "70%",
            height: "70%",
          }
    : snap && TILE_BOXES[snap] && !parked
      ? { ...TILE_BOXES[snap] }
      : parked
      ? { left: `${4 + parkIndex * 8}%`, top: 2, width: "30%", height: 46 }
      : { left: geom.x, top: geom.y, width: geom.w, height: geom.h };

  const startDrag = (e) => {
    if (maximized || parked) return;
    if (e.button !== undefined && e.button !== 0) return;
    // Traffic buttons are click targets, never drag handles — an unsteady
    // press on red/yellow/green must still close/minimize/zoom.
    if (e.target.closest && e.target.closest("button")) return;
    onFocus && onFocus();
    const header = e.currentTarget;
    const win = winRef.current;
    const parent = win.offsetParent.getBoundingClientRect();
    const r = win.getBoundingClientRect();
    const ox = e.clientX - r.left;
    const oy = e.clientY - r.top;
    const sx = e.clientX;
    const sy = e.clientY;
    let dragging = false;
    let floated = false; // snapped tile converted to floating geom this gesture
    let lastZone = null;
    // Real-OS snap zones: generous corners (quarters) win over thin edges
    // (halves), top edge means fullscreen.
    const EDGE = 24;
    const CORNER = 72;
    const zoneAt = (cx, cy) => {
      const px = cx - parent.left;
      const py = cy - parent.top;
      const lClose = px <= CORNER;
      const rClose = px >= parent.width - CORNER;
      const tClose = py <= CORNER;
      const bClose = py >= parent.height - CORNER;
      if (tClose && lClose) return "tl";
      if (tClose && rClose) return "tr";
      if (bClose && lClose) return "bl";
      if (bClose && rClose) return "br";
      if (px <= EDGE) return "left";
      if (px >= parent.width - EDGE) return "right";
      if (py <= EDGE) return "top";
      return null;
    };
    try {
      header.setPointerCapture && header.setPointerCapture(e.pointerId);
    } catch {
      /* noop — mouse fallback below still works */
    }
    const move = (ev) => {
      if (!dragging && Math.hypot(ev.clientX - sx, ev.clientY - sy) < 4) return;
      dragging = true;
      // First real move on a snapped tile: float it in place (tile pixel
      // rect becomes the floating geom) so there's no position jump, then
      // drag normally. A click without movement never unsnaps.
      if (snap && !floated) {
        floated = true;
        onUnsnap && onUnsnap({
          w: Math.round(r.width),
          h: Math.round(r.height),
          x: Math.round(Math.max(0, Math.min(r.left - parent.left, parent.width - 140))),
          y: Math.round(Math.max(0, Math.min(r.top - parent.top, parent.height - 48))),
        });
      }
      const x = Math.max(0, Math.min(ev.clientX - parent.left - ox, parent.width - 140));
      const y = Math.max(0, Math.min(ev.clientY - parent.top - oy, parent.height - 48));
      onGeom({ w: Math.round(r.width), h: Math.round(r.height), x: Math.round(x), y: Math.round(y) });
      const zone = zoneAt(ev.clientX, ev.clientY);
      if (zone !== lastZone) {
        lastZone = zone;
        onSnapPreview && onSnapPreview(zone);
      }
    };
    const up = () => {
      header.removeEventListener("pointermove", move);
      header.removeEventListener("pointerup", up);
      header.removeEventListener("pointercancel", up);
      // Release inside a snap zone tiles the window (top edge = fullscreen
      // zoom). A rejected 5th tile keeps the deny highlight briefly, then
      // stays floating where it was dropped.
      if (dragging && lastZone) {
        const zone = lastZone;
        lastZone = null;
        if (zone === "top") {
          onSnapPreview && onSnapPreview(null);
          onMax && onMax();
        } else {
          const ok = onSnap ? onSnap(zone) : true;
          if (ok === false) {
            setTimeout(() => onSnapPreview && onSnapPreview(null), 450);
          } else {
            onSnapPreview && onSnapPreview(null);
          }
        }
      } else if (!dragging) {
        onSnapPreview && onSnapPreview(null);
      }
    };
    header.addEventListener("pointermove", move);
    header.addEventListener("pointerup", up);
    header.addEventListener("pointercancel", up);
  };

  const startResize = (dir) => (e) => {
    if (maximized || parked || snap) return;
    if (e.button !== undefined && e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    onFocus && onFocus();
    const grip = e.currentTarget;
    const win = winRef.current;
    const parent = win.offsetParent.getBoundingClientRect();
    const r = win.getBoundingClientRect();
    const base = { x: r.left - parent.left, y: r.top - parent.top, w: r.width, h: r.height };
    const sx = e.clientX;
    const sy = e.clientY;
    try {
      grip.setPointerCapture && grip.setPointerCapture(e.pointerId);
    } catch {
      /* noop */
    }
    const move = (ev) => {
      const dx = ev.clientX - sx;
      const dy = ev.clientY - sy;
      let { x, y, w, h } = base;
      if (dir.includes("e")) w = base.w + dx;
      if (dir.includes("s")) h = base.h + dy;
      if (dir.includes("w")) {
        w = base.w - dx;
        x = base.x + dx;
      }
      if (dir.includes("n")) {
        h = base.h - dy;
        y = base.y + dy;
      }
      if (w < MIN_W) {
        if (dir.includes("w")) x -= MIN_W - w;
        w = MIN_W;
      }
      if (h < MIN_H) {
        if (dir.includes("n")) y -= MIN_H - h;
        h = MIN_H;
      }
      if (x < 0) {
        if (dir.includes("w")) w += x;
        x = 0;
      }
      if (y < 0) {
        if (dir.includes("n")) h += y;
        y = 0;
      }
      w = Math.min(w, parent.width);
      h = Math.min(h, parent.height);
      if (x + w > parent.width) {
        if (dir.includes("w")) x = parent.width - w;
        else w = parent.width - x;
      }
      if (y + h > parent.height) {
        if (dir.includes("n")) y = parent.height - h;
        else h = parent.height - y;
      }
      onGeom({ x: Math.round(x), y: Math.round(y), w: Math.round(w), h: Math.round(h) });
    };
    const up = () => {
      grip.removeEventListener("pointermove", move);
      grip.removeEventListener("pointerup", up);
      grip.removeEventListener("pointercancel", up);
    };
    grip.addEventListener("pointermove", move);
    grip.addEventListener("pointerup", up);
    grip.addEventListener("pointercancel", up);
  };

  const edge = "absolute z-30 touch-none";
  const corner = "absolute z-30 h-3.5 w-3.5 touch-none";

  return (
      <div
        ref={winRef}
        style={box}
        onPointerDownCapture={() => onFocus && onFocus()}
        className={`os-window absolute flex flex-col overflow-hidden ${maximized ? "rounded-none" : "rounded-[18px]"} ${
          leaving
            ? `pointer-events-none ${leaving === "min" ? "animate-window-min" : "animate-window-out"}`
            : "pointer-events-auto animate-window-in"
        } ${gliding && !leaving ? "transition-[left,top,width,height,border-radius] duration-500 ease-out" : ""}`}
      >
        <div
          onPointerDown={startDrag}
          onDoubleClick={onMax}
          title={snap ? "Drag to float • double-click to zoom" : "Drag to move • drop on screen edges/corners to split"}
          aria-hidden={hideChrome}
          className={`os-titlebar relative z-20 flex shrink-0 cursor-grab touch-none items-center px-4 transition-all duration-200 select-none active:cursor-grabbing ${hideChrome ? "h-0 overflow-hidden border-0 opacity-0" : "h-10 opacity-100"}`}
          aria-label={`${title} title bar, drag to move`}
        >
          <span className="traffic-group flex items-center gap-2">
            <button onClick={onClose} aria-label="Close window" className="traffic flex h-3.5 w-3.5 items-center justify-center rounded-full bg-[#ff5f57] transition hover:brightness-90">
              <svg viewBox="0 0 8 8" className="traffic-icon h-2 w-2" aria-hidden="true"><path d="M1.5 1.5l4.7 4.7M6.2 1.5L1.5 6.2" stroke="rgba(90,20,20,0.85)" strokeWidth="1.2" strokeLinecap="round" /></svg>
            </button>
            <button onClick={onMin} aria-label="Minimize window" className="traffic flex h-3.5 w-3.5 items-center justify-center rounded-full bg-[#febc2e] transition hover:brightness-90">
              <svg viewBox="0 0 8 8" className="traffic-icon h-2 w-2" aria-hidden="true"><path d="M1.5 4h5" stroke="rgba(120,70,0,0.85)" strokeWidth="1.2" strokeLinecap="round" /></svg>
            </button>
            <button onClick={onMax} aria-label="Zoom window" className="traffic flex h-3.5 w-3.5 items-center justify-center rounded-full bg-[#28c840] transition hover:brightness-90">
              <svg viewBox="0 0 8 8" className="traffic-icon h-2 w-2" aria-hidden="true"><path d="M1.8 4.2L4 1.8l2.2 2.4M1.8 5.8L4 6.6l2.2-.8" stroke="rgba(10,80,30,0.85)" strokeWidth="1.1" strokeLinecap="round" strokeLinejoin="round" fill="none" /></svg>
            </button>
          </span>
          <span className="pointer-events-none absolute inset-0 flex items-center justify-center text-[13px] font-bold text-slate-700">
            {title}
          </span>
        </div>
        <div className={`relative z-10 flex min-h-0 flex-col transition-all duration-300 ${parked ? "h-0 flex-none overflow-hidden opacity-0" : "flex-1 opacity-100"}`}>{children}</div>

        {/* Split-screen tiles size via their tile box, not free resize:
            drag the title bar to float the window before resizing. */}
        {!maximized && !parked && !snap && (
          <>
            <div onPointerDown={startResize("n")} className={`${edge} top-0 right-4 left-4 h-1.5 cursor-ns-resize`} aria-label="Resize top" />
            <div onPointerDown={startResize("s")} className={`${edge} right-4 bottom-0 left-4 h-1.5 cursor-ns-resize`} aria-label="Resize bottom" />
            <div onPointerDown={startResize("e")} className={`${edge} top-4 right-0 bottom-4 w-1.5 cursor-ew-resize`} aria-label="Resize right" />
            <div onPointerDown={startResize("w")} className={`${edge} top-4 bottom-4 left-0 w-1.5 cursor-ew-resize`} aria-label="Resize left" />
            <div onPointerDown={startResize("nw")} className={`${corner} top-0 left-0 cursor-nwse-resize rounded-tl-[18px]`} aria-label="Resize top left" />
            <div onPointerDown={startResize("ne")} className={`${corner} top-0 right-0 cursor-nesw-resize rounded-tr-[18px]`} aria-label="Resize top right" />
            <div onPointerDown={startResize("sw")} className={`${corner} bottom-0 left-0 cursor-nesw-resize rounded-bl-[18px]`} aria-label="Resize bottom left" />
            <div
              onPointerDown={startResize("se")}
              className="absolute right-1 bottom-1 z-30 h-5 w-5 touch-none cursor-nwse-resize rounded-br-[16px] opacity-70 transition hover:opacity-100"
              aria-label="Resize bottom right"
              style={{
                background:
                  "linear-gradient(135deg, transparent 50%, rgba(100,116,139,0.8) 50%, rgba(100,116,139,0.8) 60%, transparent 60%, transparent 70%, rgba(100,116,139,0.8) 70%, rgba(100,116,139,0.8) 80%, transparent 80%)",
              }}
            />
          </>
        )}
      </div>
  );
}
