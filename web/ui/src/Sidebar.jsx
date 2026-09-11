import { useEffect, useRef } from "react";
import gsap from "gsap";
import {
  FaClock,
  FaComments,
  FaDesktop,
  FaMicrophone,
  FaTrash,
  FaWandMagicSparkles,
  FaXmark,
} from "react-icons/fa6";

export default function Sidebar({
  open,
  onClose,
  connected,
  listening,
  asrReady,
  messageCount,
  hasDiagram,
  onClear,
  onToggleMic,
  onShareScreen,
  sharing,
  visionEnabled,
}) {
  const ref = useRef(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    gsap.to(el, {
      x: open ? 0 : -320,
      autoAlpha: open ? 1 : 0,
      duration: 0.45,
      ease: "power3.out",
      overwrite: true,
    });
  }, [open ]);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    gsap.from(el, { x: -40, autoAlpha: 0, duration: 0.6, ease: "power3.out" });
  }, []);

  const status = connected
    ? listening
      ? asrReady
        ? "Listening"
        : "Warming up"
      : "Ready"
    : "Connecting";

  return (
    <aside
      ref={ref}
      className={`z-30 flex h-full w-72 shrink-0 flex-col border-r border-slate-200/80 bg-white max-md:fixed max-md:inset-y-0 max-md:left-0 max-md:shadow-2xl ${open ? "" : "pointer-events-none"}`}
      aria-label="Tutor sidebar"
      aria-hidden={!open}
    >
      <div className="flex items-center justify-between px-5 pt-6 pb-5">
        <div className="flex items-center gap-3">
          <span className="flex h-10 w-10 items-center justify-center rounded-2xl bg-[#ff5a5f] text-white shadow-[0_10px_26px_rgba(255,90,95,0.35)]">
            <FaWandMagicSparkles className="h-4 w-4" />
          </span>
          <div>
            <h1 className="text-sm font-bold tracking-[0.12em] text-slate-900 uppercase">Saathi</h1>
            <p className="mt-0.5 text-[11px] text-slate-500">Hindi voice tutor</p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="inline-flex h-8 w-8 items-center justify-center rounded-full text-slate-400 transition hover:bg-slate-100 hover:text-slate-700"
          aria-label="Close sidebar"
        >
          <FaXmark className="h-4 w-4" />
        </button>
      </div>

      <div className="mx-5 flex items-center gap-2 rounded-2xl bg-slate-50 px-3.5 py-2.5 text-xs font-medium text-slate-600">
        <span
          className={`h-2 w-2 shrink-0 rounded-full ${connected ? "bg-emerald-500" : "bg-amber-500"}`}
        />
        {status}
        {hasDiagram && (
          <span className="ml-auto rounded-full bg-[#ff5a5f]/10 px-2 py-0.5 text-[10px] font-bold text-[#ff5a5f]">
            BOARD LIVE
          </span>
        )}
      </div>

      <nav className="mt-4 flex flex-col gap-1 px-3" aria-label="Tutor actions">
        <button
          onClick={onToggleMic}
          className="group flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium text-slate-700 transition hover:bg-[#ff5a5f]/5 hover:text-[#ff5a5f]"
        >
          <FaMicrophone className="h-4 w-4 text-[#ff5a5f]" />
          {listening ? "Stop listening" : "Start listening"}
        </button>
        <button
          onClick={onShareScreen}
          disabled={!visionEnabled}
          className="group flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium text-slate-700 transition hover:bg-[#ff5a5f]/5 hover:text-[#ff5a5f] disabled:cursor-not-allowed disabled:opacity-40"
        >
          <FaDesktop className="h-4 w-4 text-[#ff5a5f]" />
          {sharing ? "Stop screen share" : "Share screen"}
          <span className="ml-auto rounded-md border border-dashed border-slate-300 px-1.5 py-0.5 text-[10px] font-bold text-slate-400">
            X
          </span>
        </button>
        <button
          onClick={onClear}
          className="group flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium text-slate-700 transition hover:bg-[#ff5a5f]/5 hover:text-[#ff5a5f]"
        >
          <FaTrash className="h-4 w-4 text-[#ff5a5f]" />
          Clear chat
        </button>
      </nav>

      <div className="mt-auto border-t border-slate-200/70 px-5 py-4">
        <div className="flex items-center gap-2 text-[11px] text-slate-500">
          <FaComments className="h-3.5 w-3.5 text-[#ff5a5f]" />
          {messageCount} messages this session
        </div>
        <div className="mt-2 flex items-center gap-2 text-[11px] text-slate-400">
          <FaClock className="h-3.5 w-3.5" />
          Hold X to show your screen, release to ask
        </div>
      </div>
    </aside>
  );
}
