import { useEffect, useRef } from "react";
import { FaDesktop, FaMicrophone, FaPaperPlane, FaStop } from "react-icons/fa6";
import { engine } from "./audioEngine.js";

/* Aurora dock (like the reference): near-black panel with fluid color blobs
   — deep blue, violet, warm peach, mint — that dance with voice pitch/volume.
   Red #ff5a5f mic/send controls float on top. No circular globe. */

export default function BottomBar({
  input,
  setInput,
  sendText,
  connected,
  listening,
  micBusy,
  speaking,
  userTalking,
  sharing,
  visionEnabled,
  onToggleMic,
  onShareDown,
  onShareUp,
}) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    let raf = 0;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);

    // Soft aurora wash: huge radii, flat core, slow drift — no ball glows.
    const blobs = [
      { c: [59, 70, 255], x: 0.22, y: 1.05, r: 1.0, s: 0.14, p: 0.0 }, // deep blue
      { c: [124, 93, 250], x: 0.42, y: 1.12, r: 0.95, s: 0.16, p: 2.1 }, // violet
      { c: [255, 179, 122], x: 0.58, y: 1.08, r: 0.9, s: 0.14, p: 4.2 }, // warm peach
      { c: [92, 255, 157], x: 0.78, y: 1.02, r: 0.88, s: 0.14, p: 1.2 }, // mint
      { c: [255, 90, 95], x: 0.5, y: 1.2, r: 0.95, s: 0.11, p: 3.0 }, // brand red undertow
    ];
    let drift = blobs.map(() => Math.random() * 100);

    const size = () => {
      const r = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, r.width * dpr);
      canvas.height = Math.max(1, r.height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    size();
    const ro = new ResizeObserver(size);
    ro.observe(canvas);

    const draw = () => {
      const r = canvas.getBoundingClientRect();
      const W = r.width;
      const H = r.height;
      const mic = engine.readMic();
      const spk = engine.readSpeak();
      const active = speaking || (listening && userTalking);
      const level = Math.min(
        1,
        Math.max(
          speaking ? spk.rms * 9 : 0,
          listening && userTalking ? mic.rms * 11 : 0,
          listening ? mic.rms * 6 + 0.05 : 0.04,
        ),
      );
      const pitch = speaking ? spk.pitch : mic.pitch || 0;
      const t = performance.now() / 1000;
      const agitation = active ? 0.5 : 0.12;

      // near-black base like the reference
      ctx.globalCompositeOperation = "source-over";
      ctx.fillStyle = "#060609";
      ctx.fillRect(0, 0, W, H);

      // slow blended wash — voice gently breathes the field, no hotspots
      ctx.globalCompositeOperation = "lighter";
      blobs.forEach((b, i) => {
        if (active) drift[i] += 0.004 + Math.random() * 0.008 * (pitch > 0 ? 1 : 0.3);
        const px = (pitch > 0 ? (pitch / 480) * 0.1 : 0) * (i % 2 === 0 ? 1 : -1);
        const cx = (b.x + px) * W + Math.sin(t * b.s * agitation * 2 + b.p + drift[i] * 0.02) * W * 0.04 * (0.4 + level);
        const cy = b.y * H + Math.cos(t * b.s * agitation * 1.6 + b.p * 1.7) * H * 0.12 * (0.4 + level);
        const rad = Math.max(20, b.r * Math.min(W, H * 3.2) * (0.9 + level * 0.35));
        const alpha = active ? 0.2 + level * 0.14 : 0.15;
        const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, rad);
        const [cr, cg, cb] = b.c;
        g.addColorStop(0, `rgba(${cr},${cg},${cb},${alpha.toFixed(3)})`);
        g.addColorStop(0.7, `rgba(${cr},${cg},${cb},${(alpha * 0.55).toFixed(3)})`);
        g.addColorStop(1, "rgba(6,6,9,0)");
        ctx.fillStyle = g;
        ctx.fillRect(0, 0, W, H);
      });
      ctx.globalCompositeOperation = "source-over";

      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, [listening, speaking, userTalking]);

  return (
    <div className="sticky bottom-0 z-20 px-4 pt-2 pb-4 sm:px-6">
      <div className="relative overflow-hidden rounded-[28px] bg-[#060609] shadow-[0_18px_60px_rgba(5,5,10,0.5)]">
        <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" aria-hidden="true" />
        <form
          onSubmit={sendText}
          className="relative z-10 flex items-center gap-1.5 px-3 py-3"
        >
          <button
            type="button"
            onPointerDown={(e) => {
              e.preventDefault();
              onShareDown && onShareDown();
            }}
            onPointerUp={() => onShareUp && onShareUp()}
            onPointerLeave={() => onShareUp && onShareUp()}
            onContextMenu={(e) => e.preventDefault()}
            disabled={!visionEnabled}
            title="Hold to share your screen"
            className={`inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-full backdrop-blur transition active:scale-95 disabled:cursor-not-allowed disabled:opacity-35 ${
              sharing ? "bg-[#ff5a5f] text-white" : "bg-white/10 text-white/70 hover:bg-white/20 hover:text-white"
            }`}
          >
            <FaDesktop className="h-4 w-4" />
            <span className="sr-only">Share screen</span>
          </button>
          <button
            type="button"
            onClick={onToggleMic}
            disabled={micBusy}
            title={listening ? "Stop listening" : "Start listening"}
            className={`inline-flex h-11 w-11 shrink-0 items-center justify-center rounded-full text-white transition active:scale-95 disabled:cursor-not-allowed disabled:opacity-40 ${
              listening ? "bg-white text-slate-900" : "bg-[#ff5a5f]"
            }`}
          >
            {micBusy ? (
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
            ) : listening ? (
              <FaStop className="h-4 w-4" />
            ) : (
              <FaMicrophone className="h-4 w-4" />
            )}
            <span className="sr-only">{listening ? "Stop listening" : "Start listening"}</span>
          </button>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={speaking ? "Speaking… type to interrupt" : "Ask your tutor…"}
            aria-label="Message the tutor"
            className="min-h-11 flex-1 rounded-full bg-white/10 px-4 text-sm text-white outline-none backdrop-blur placeholder:text-white/50 focus:bg-white/15"
          />
          <button
            type="submit"
            disabled={!input.trim() || !connected}
            aria-label="Send message"
            className="inline-flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-[#ff5a5f] text-white shadow-[0_10px_24px_rgba(255,90,95,0.45)] transition hover:brightness-110 active:scale-95 disabled:cursor-not-allowed disabled:opacity-40"
          >
            <FaPaperPlane className="h-4 w-4" />
          </button>
        </form>
      </div>
    </div>
  );
}
