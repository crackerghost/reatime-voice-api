import { useRef, useState, useEffect } from "react";

/* Shared macOS window: traffic lights, drag-by-titlebar, 8-handle edge +
   corner resize, green zoom = fullscreen. Geometry is owned by the parent
   (per-app memory); null geom = centered default at 50% height, cascaded so
   stacked windows keep their title bars visible.
   Drag uses pointer capture on the title bar + a 4px threshold, so a press
   starting anywhere on the bar (even on a traffic button) drags on move and
   still clicks when released in place. */
export default function AppWindow({
  title, geom, onGeom, maximized, cascade = 0, leaving, hideChrome,
  parked, parkIndex = 0,
  onClose, onMin, onMax, onFocus, children,
}) {
  const winRef = useRef(null);
  const MIN_W = 480;
  const MIN_H = 340;
  // Ease animated geometry changes: fullscreen zoom and desktop-park spread
  // (drag/resize must still track the pointer 1:1).
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
  }, [maximized, parked]);

  // Default (never dragged/resized): 70% x 70% window, centered, cascaded
  // so stacked windows keep their title bars visible.
  // Parked (desktop spread): a clickable title-bar strip tucked under the
  // menu bar — always hittable, fanned per window.
  const box = maximized || !geom
    ? maximized
      ? { left: 0, top: 0, width: "100%", height: "100%" }
      : parked
        ? { left: `${4 + parkIndex * 8}%`, top: 2, width: "30%", height: 46 }
        : {
            left: `calc(15% + ${cascade * 36}px)`,
            top: `calc(15% + ${cascade * 44}px)`,
            width: "70%",
            height: "70%",
          }
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
    try {
      header.setPointerCapture && header.setPointerCapture(e.pointerId);
    } catch {
      /* noop — mouse fallback below still works */
    }
    const move = (ev) => {
      if (!dragging && Math.hypot(ev.clientX - sx, ev.clientY - sy) < 4) return;
      dragging = true;
      const x = Math.max(0, Math.min(ev.clientX - parent.left - ox, parent.width - 140));
      const y = Math.max(0, Math.min(ev.clientY - parent.top - oy, parent.height - 48));
      onGeom({ w: Math.round(r.width), h: Math.round(r.height), x: Math.round(x), y: Math.round(y) });
    };
    const up = () => {
      header.removeEventListener("pointermove", move);
      header.removeEventListener("pointerup", up);
      header.removeEventListener("pointercancel", up);
    };
    header.addEventListener("pointermove", move);
    header.addEventListener("pointerup", up);
    header.addEventListener("pointercancel", up);
  };

  const startResize = (dir) => (e) => {
    if (maximized || parked) return;
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
          title="Drag to move"
          aria-hidden={hideChrome}
          className={`os-titlebar relative z-20 flex shrink-0 cursor-grab touch-none items-center px-4 transition-all duration-200 select-none active:cursor-grabbing ${hideChrome ? "h-0 overflow-hidden border-0 opacity-0" : "h-10 opacity-100"}`}
          aria-label={`${title} title bar, drag to move`}
        >
          <span className="flex items-center gap-2">
            <button onClick={onClose} aria-label="Close window" className="traffic h-3.5 w-3.5 rounded-full bg-[#ff5f57] transition hover:brightness-90" />
            <button onClick={onMin} aria-label="Minimize window" className="traffic h-3.5 w-3.5 rounded-full bg-[#febc2e] transition hover:brightness-90" />
            <button onClick={onMax} aria-label="Zoom window" className="traffic h-3.5 w-3.5 rounded-full bg-[#28c840] transition hover:brightness-90" />
          </span>
          <span className="pointer-events-none absolute inset-0 flex items-center justify-center text-[13px] font-bold text-slate-700">
            {title}
          </span>
        </div>
        <div className={`relative z-10 flex min-h-0 flex-col transition-all duration-300 ${parked ? "h-0 flex-none overflow-hidden opacity-0" : "flex-1 opacity-100"}`}>{children}</div>

        {!maximized && !parked && (
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
