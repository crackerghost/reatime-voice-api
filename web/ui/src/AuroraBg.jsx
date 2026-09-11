import { useEffect, useRef } from "react";
import { engine } from "./audioEngine.js";

/* Full-bleed aurora wash: fixed bottom 50% of the screen, colors fully
   blended into each other (no separate balls), dancing with voice
   pitch/volume. Top edge fades into the page via CSS mask. */

const BLOBS = [
  { c: [48, 64, 220], x: 0.14, y: 0.62, r: 1.1, s: 0.03, p: 0.0 }, // deep blue
  { c: [110, 80, 235], x: 0.34, y: 0.78, r: 1.05, s: 0.035, p: 2.1 }, // violet
  { c: [255, 165, 110], x: 0.56, y: 0.74, r: 0.95, s: 0.03, p: 4.2 }, // warm peach
  { c: [90, 230, 150], x: 0.8, y: 0.6, r: 0.92, s: 0.03, p: 1.2 }, // mint
  { c: [235, 120, 160], x: 0.45, y: 0.9, r: 1.0, s: 0.025, p: 3.0 }, // rose undertow
];

export default function AuroraBg({ listening, speaking, userTalking }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    let raf = 0;
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    let drift = BLOBS.map(() => Math.random() * 100);

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
      if (W < 2 || H < 2) {
        raf = requestAnimationFrame(draw);
        return;
      }
      const mic = engine.readMic();
      const spk = engine.readSpeak();
      const active = speaking || (listening && userTalking);
      const level = Math.min(
        1,
        Math.max(
          speaking ? spk.rms * 9 : 0,
          listening && userTalking ? mic.rms * 11 : 0,
          listening ? mic.rms * 6 + 0.06 : 0.05,
        ),
      );
      const pitch = speaking ? spk.pitch : mic.pitch || 0;
      const t = performance.now() / 1000;
      const agitation = active ? 0.22 : 0.06;

      ctx.globalCompositeOperation = "source-over";
      ctx.clearRect(0, 0, W, H);

      // Ultra-soft wash: huge radii, flat soft core (no glowing hotspot),
      // slow drift only. Voice gently breathes the whole field.
      ctx.globalCompositeOperation = "lighter";
      BLOBS.forEach((b, i) => {
        if (active) drift[i] += 0.001 + Math.random() * 0.002 * (pitch > 0 ? 1 : 0.3);
        const px = (pitch > 0 ? (pitch / 480) * 0.12 : 0) * (i % 2 === 0 ? 1 : -1);
        const cx =
          (b.x + px) * W +
          Math.sin(t * b.s * agitation * 2 + b.p + drift[i] * 0.02) * W * 0.05 * (0.4 + level);
        const cy =
          b.y * H +
          Math.cos(t * b.s * agitation * 1.5 + b.p * 1.7) * H * 0.1 * (0.4 + level);
        const rad = Math.max(30, b.r * Math.max(W * 0.7, H * 2.2) * (0.9 + level * 0.4));
        const alpha = active ? 0.13 + level * 0.1 : 0.1;
        const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, rad);
        const [cr, cg, cb] = b.c;
        g.addColorStop(0, `rgba(${cr},${cg},${cb},${alpha.toFixed(3)})`);
        g.addColorStop(0.7, `rgba(${cr},${cg},${cb},${(alpha * 0.55).toFixed(3)})`);
        g.addColorStop(1, "rgba(255,255,255,0)");
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
    <div
      className="pointer-events-none fixed inset-x-0 bottom-0 z-0 h-[50vh] [mask-image:linear-gradient(to_bottom,transparent,black_38%)]"
      aria-hidden="true"
    >
      <canvas ref={canvasRef} className="h-full w-full" />
    </div>
  );
}
