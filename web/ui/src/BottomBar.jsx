import { useEffect, useRef } from "react";
import { FaDesktop, FaMicrophone, FaPaperPlane, FaStop } from "react-icons/fa6";
import { engine } from "./audioEngine.js";

/* Gemini-style bottom input bar: white pill inside a live gradient bound.
   The bound + mini visualizer react to voice pitch/volume — hue drifts
   randomly as pitch moves (no circular globe). */

const RED = "#ff5a5f";

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
  const wrapRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    let raf = 0;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    let hue = 355;
    let hueTarget = 355;

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
      ctx.clearRect(0, 0, W, H);
      const mic = engine.readMic();
      const spk = engine.readSpeak();
      const active = speaking || userTalking;
      const level = Math.min(
        1,
        Math.max(speaking ? spk.rms * 9 : 0, listening && userTalking ? mic.rms * 11 : 0, 0.04),
      );
      const pitch = speaking ? spk.pitch : mic.pitch;
      if (active && pitch > 0) {
        // random gradient drift on pitch move
        hueTarget = 345 + ((pitch / 480) * 40 + Math.random() * 14);
      } else if (!active) {
        hueTarget = 355;
      }
      hue += (hueTarget - hue) * 0.08;

      const N = 56;
      const bw = W / N;
      const t = performance.now() / 1000;
      for (let i = 0; i < N; i++) {
        const wave =
          0.5 + 0.5 * Math.sin(i * 0.55 + t * (active ? 6 : 1.6)) * Math.sin(i * 0.21 - t * 2.2);
        const h = Math.max(2, (0.12 + level * 0.88) * H * (0.25 + 0.75 * wave));
        const x = i * bw + bw * 0.22;
        const g = ctx.createLinearGradient(0, H - h, 0, H);
        g.addColorStop(0, `hsla(${hue.toFixed(0)}, 100%, 64%, 0.95)`);
        g.addColorStop(1, `hsla(${((hue + 38) % 360).toFixed(0)}, 95%, 60%, 0.35)`);
        ctx.fillStyle = g;
        const y = H - h;
        if (ctx.roundRect) {
          ctx.beginPath();
          ctx.roundRect(x, y, bw * 0.56, h, 3);
          ctx.fill();
        } else {
          ctx.fillRect(x, y, bw * 0.56, h);
        }
      }
      // live bounding gradient follows the same hue
      if (wrapRef.current) {
        wrapRef.current.style.background = `linear-gradient(135deg, hsl(${hue.toFixed(
          0
        )}, 100%, 64%), hsl(${((hue + 40) % 360).toFixed(0)}, 95%, 62%), hsl(340, 85%, 55%))`;
      }
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
      <div
        ref={wrapRef}
        className="rounded-[28px] p-[2px] shadow-[0_18px_50px_rgba(255,90,95,0.22)]"
        style={{ background: `linear-gradient(135deg, ${RED}, #ff8a5c, #c81e5b)` }}
      >
        <div className="rounded-[26px] bg-white px-2 pt-1.5 pb-2">
          <canvas ref={canvasRef} className="h-9 w-full" aria-hidden="true" />
          <form
            onSubmit={sendText}
            className="flex items-center gap-1.5 px-1 pt-1"
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
              className={`inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-full transition active:scale-95 disabled:cursor-not-allowed disabled:opacity-35 ${
                sharing
                  ? "bg-[#ff5a5f] text-white"
                  : "text-slate-400 hover:bg-slate-100 hover:text-[#ff5a5f]"
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
                listening ? "bg-slate-900" : "bg-[#ff5a5f]"
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
              className="min-h-11 flex-1 bg-transparent px-2 text-sm text-slate-800 outline-none placeholder:text-slate-400"
            />
            <button
              type="submit"
              disabled={!input.trim() || !connected}
              aria-label="Send message"
              className="inline-flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-[#ff5a5f] text-white shadow-[0_10px_24px_rgba(255,90,95,0.35)] transition hover:brightness-95 active:scale-95 disabled:cursor-not-allowed disabled:opacity-40"
            >
              <FaPaperPlane className="h-4 w-4" />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
