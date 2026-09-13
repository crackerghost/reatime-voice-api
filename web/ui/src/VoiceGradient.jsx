import { useEffect, useRef } from "react";
import { engine } from "./audioEngine.js";

/* Bottom 50% background glow. The layer reads live mic/speaker pitch + RMS and
   dances with voice, brightening while the user or tutor is speaking. */

export default function VoiceGradient({ listening, speaking, userTalking }) {
  const layerRef = useRef(null);
  const gradientRef = useRef(null);
  const heightRef = useRef(50);

  useEffect(() => {
    let raf = 0;

    const draw = () => {
      const layer = layerRef.current;
      const gradient = gradientRef.current;
      if (layer && gradient) {
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

        // Slow, smooth height breathing while anyone is speaking; calm at 50% otherwise.
        const slowWave = Math.sin(t * 0.5);
        const targetHeight = active ? 50 + slowWave * 10 * (0.45 + level) : 50;
        heightRef.current += (targetHeight - heightRef.current) * 0.035;
        layer.style.height = `${heightRef.current}%`;

        const driftX = (pitch > 0 ? (pitch / 480) * 22 : 0) + Math.sin(t * 0.5) * 6 * (0.4 + level);
        const driftY = Math.cos(t * 0.4) * 4 * (0.4 + level);
        const breatheX = 0.88 + level * 0.22;
        const breatheY = 0.78 + level * 0.42;
        const glowScale = 0.96 + level * 0.12;

        layer.style.opacity = active ? String(Math.min(1, 0.55 + level * 0.45)) : "0.45";
        layer.style.transform = `translate(${driftX}px, ${driftY}px) scale(${glowScale})`;
        gradient.style.transform = `scaleX(${breatheX}) scaleY(${breatheY}) rotate(${(pitch / 480) * 3}deg)`;
      }
      raf = requestAnimationFrame(draw);
    };

    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [listening, speaking, userTalking]);

  return (
    <div
      ref={layerRef}
      className="voice-gradient-layer absolute inset-x-0 bottom-0 h-1/2"
      aria-hidden="true"
    >
      <div ref={gradientRef} className="gemini-gradient" />
    </div>
  );
}
