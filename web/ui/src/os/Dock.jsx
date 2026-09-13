import { useRef, useState } from "react";
import {
  FaGraduationCap, FaGlobe, FaNoteSticky, FaChalkboardUser, FaCode,
} from "react-icons/fa6";

const APPS = [
  { id: "tutor", name: "Tutor", icon: <FaGraduationCap />, bg: "linear-gradient(165deg,#ffb199 0%,#ff5a5f 55%,#d63a4e 100%)", fg: "#fff" },
  { id: "whiteboard", name: "Whiteboard", icon: <FaChalkboardUser />, bg: "linear-gradient(165deg,#fde68a 0%,#f59e0b 60%,#b45309 100%)", fg: "#fff" },
  { id: "browser", name: "Browser", icon: <FaGlobe />, bg: "linear-gradient(165deg,#7dd3fc 0%,#2563eb 60%,#1e3a8a 100%)", fg: "#fff" },
  { id: "code", name: "Code", icon: <FaCode />, bg: "linear-gradient(165deg,#c4b5fd 0%,#7c3aed 60%,#4c1d95 100%)", fg: "#fff" },
  { id: "notes", name: "Notes", icon: <FaNoteSticky />, bg: "linear-gradient(165deg,#fefce8 0%,#fde047 55%,#eab308 100%)", fg: "#92400e" },
];

/* Bug OS dock: the 4 apps, mac-style cursor magnification, running dots. */
export default function Dock({ activeApp, onOpen, visible, running, noteCount }) {
  const trackRef = useRef(null);
  const [mag, setMag] = useState([]);
  const dot = (id) => (running || []).includes(id) || id === activeApp;

  const onMove = (e) => {
    const track = trackRef.current;
    if (!track) return;
    const scales = [];
    for (const kid of track.children) {
      const r = kid.getBoundingClientRect();
      const d = Math.abs(e.clientX - (r.left + r.width / 2));
      const f = Math.max(0, 1 - d / 110);
      scales.push(1 + f * f * 0.55);
    }
    setMag(scales);
  };

  return (
    <div
      className={`absolute inset-x-0 bottom-0 z-40 flex justify-center transition-all duration-300 ${
        visible ? "translate-y-0 opacity-100" : "translate-y-28 opacity-0"
      }`}
      aria-label="Dock"
    >
      <div
        ref={trackRef}
        onMouseMove={onMove}
        onMouseLeave={() => setMag([])}
        className="os-dock mb-2 flex items-end gap-2 rounded-[26px] px-3 pt-2.5 pb-2"
      >
        {APPS.map((a, i) => {
          const s = mag[i] || 1;
          return (
            <button
              key={a.id}
              onClick={() => onOpen(a.id)}
              title={a.name}
              aria-label={`Open ${a.name}`}
              className="rounded-2xl"
              style={{
                transform: `scale(${s}) translateY(${(s - 1) * -16}px)`,
                transformOrigin: "bottom center",
                transition: "transform 120ms ease-out",
              }}
            >
              <span className="group relative flex flex-col items-center">
                <span
                  aria-hidden="true"
                  className="dock-icon flex h-12 w-12 items-center justify-center text-xl"
                  style={{ background: a.bg, color: a.fg }}
                >
                  {a.icon}
                </span>
                <span className="pointer-events-none absolute -top-8 hidden rounded-lg bg-black/70 px-2 py-1 text-[11px] font-medium whitespace-nowrap text-white group-hover:block">
                  {a.id === "notes" && noteCount > 0 ? `${a.name} (${Math.min(noteCount, 9)})` : a.name}
                </span>
              </span>
              <span className="mt-1 flex h-1.5 items-start justify-center" aria-hidden="true">
                {dot(a.id) ? <span className="block h-1 w-1 rounded-full bg-white shadow-[0_0_4px_rgba(255,255,255,0.9)]" /> : null}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
