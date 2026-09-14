import { useMemo, useState } from "react";
import {
  FaBookOpen, FaCheck, FaChevronDown, FaChevronRight, FaCirclePlay,
  FaClock, FaGraduationCap, FaListUl, FaMagnifyingGlass, FaArrowRight, FaArrowLeft, FaCircleCheck,
} from "react-icons/fa6";
import { getProgressPercent } from "../data/courseData.js";

/* Tutor — professional course browser.
 * Left: searchable curriculum (course → module → lesson).
 * Right: lesson overview card + sticky start action.
 * Same props/API as before so App.jsx wiring is untouched. */

const KIND_STYLE = {
  coding: "bg-violet-50 text-violet-700 ring-violet-200",
  project: "bg-amber-50 text-amber-800 ring-amber-200",
  theory: "bg-sky-50 text-sky-700 ring-sky-200",
  video: "bg-rose-50 text-rose-700 ring-rose-200",
  quiz: "bg-emerald-50 text-emerald-700 ring-emerald-200",
};
const KIND_LABEL = { theory: "Theory", coding: "Hands-on", video: "Video", quiz: "Quiz", project: "Project" };

const COVER = [
  "from-[#ff5a5f] via-[#f43f5e] to-[#be123c]",
  "from-[#fb7185] via-[#ff5a5f] to-[#9f1239]",
];

const RED = "#ff5a5f";

function initials(title = "") {
  return title.split(/[\s—–-]+/).filter(Boolean).slice(0, 2).map((w) => w[0]).join("").toUpperCase();
}

function flatLessons(course) {
  const out = [];
  for (const m of course?.modules || []) for (const l of m.lessons || []) out.push({ courseId: course.id, moduleId: m.id, moduleTitle: m.title, ...l });
  return out;
}

export default function TutorApp({
  courses = [],
  progress,
  activeCourseId,
  activeLessonId,
  selected,
  onSelect,
  onStartLesson,
  onCompleteLesson,
}) {
  const [q, setQ] = useState("");
  const [openCourses, setOpenCourses] = useState(() => ({ [activeCourseId]: true }));
  const [openModules, setOpenModules] = useState(() => {
    const m = {};
    for (const c of courses || []) for (const mod of c.modules || []) m[`${c.id}:${mod.id}`] = selected?.courseId === c.id && selected?.moduleId === mod.id;
    return m;
  });

  const needle = q.trim().toLowerCase();
  const filtered = useMemo(() => {
    if (!needle) return courses;
    const words = needle.split(/\s+/);
    return (courses || [])
      .map((c) => ({
        ...c,
        modules: (c.modules || [])
          .map((m) => ({
            ...m,
            lessons: (m.lessons || []).filter((l) => {
              const hay = [c.title, m.title, l.title, (l.keyTerms || []).join(" ")].join(" ").toLowerCase();
              return words.every((w) => hay.includes(w));
            }),
          }))
          .filter((m) => m.lessons.length || m.title.toLowerCase().includes(needle)),
      }))
      .filter((c) => c.modules.length || c.title.toLowerCase().includes(needle));
  }, [courses, needle]);

  const selCourse = (courses || []).find((c) => c.id === selected?.courseId) || (courses || [])[0] || null;
  const selModule = selCourse?.modules?.find((m) => m.id === selected?.moduleId) || selCourse?.modules?.[0] || null;
  const selLesson = selModule?.lessons?.find((l) => l.id === selected?.lessonId) || selModule?.lessons?.[0] || null;

  const all = useMemo(() => (selCourse ? flatLessons(selCourse) : []), [selCourse]);
  const idx = all.findIndex((l) => l.lessonId === undefined ? false : l.id === selLesson?.id);
  const prev = idx > 0 ? all[idx - 1] : null;
  const next = idx >= 0 && idx < all.length - 1 ? all[idx + 1] : null;
  const totalMins = all.reduce((s, l) => s + (l.durationMin || 0), 0);
  const pct = selCourse ? getProgressPercent(selCourse, progress) : 0;
  const doneIds = new Set(progress?.completedLessonIds || []);
  const statusOf = (id) => (id === activeLessonId ? "current" : doneIds.has(id) ? "done" : "todo");

  return (
    <div className="flex h-full min-h-0 overflow-hidden rounded-2xl border border-slate-200/80 bg-white text-slate-900">
      {/* ── Sidebar ─────────────────────────────── */}
      <aside className="flex w-72 shrink-0 flex-col border-r border-slate-200/80 bg-slate-50/60">
        <div className="border-b border-slate-200/70 px-4 pt-4 pb-3">
          <div className="flex items-center gap-2">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#ff5a5f] text-white shadow-[0_6px_16px_rgba(255,90,95,0.4)]">
              <FaGraduationCap className="h-4 w-4" />
            </span>
            <div className="min-w-0">
              <p className="text-[13px] font-bold tracking-tight">Catalog</p>
              <p className="text-[11px] text-slate-500">{courses.length} course{courses.length === 1 ? "" : "s"} · {all.length} lessons</p>
            </div>
          </div>
          <div className="mt-3 flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 shadow-sm transition focus-within:border-[#ff5a5f] focus-within:ring-2 focus-within:ring-[#ff5a5f]/15">
            <FaMagnifyingGlass className="h-3.5 w-3.5 shrink-0 text-slate-400" />
            <input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              aria-label="Search courses and topics"
              placeholder="Search lessons, topics…"
              className="min-w-0 flex-1 bg-transparent text-[13px] outline-none placeholder:text-slate-400"
            />
            {q && (
              <button onClick={() => setQ("")} aria-label="Clear search" className="text-[11px] font-semibold text-slate-400 hover:text-slate-700">Clear</button>
            )}
          </div>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-2 py-2">
          {filtered.length === 0 && (
            <div className="rounded-xl border border-dashed border-slate-200 bg-white px-3 py-6 text-center">
              <p className="text-[13px] font-semibold">No matches</p>
              <p className="mt-1 text-xs text-slate-500">Try “react”, “flexbox”, or “mongo”.</p>
            </div>
          )}
          {filtered.map((c, ci) => {
            const open = needle ? true : !!openCourses[c.id];
            const cpct = getProgressPercent(c, progress);
            return (
              <div key={c.id} className="mb-1.5 overflow-hidden rounded-xl border border-slate-200/80 bg-white shadow-[0_1px_2px_rgba(15,23,42,0.05)]">
                <button
                  onClick={() => setOpenCourses((p) => ({ ...p, [c.id]: !p[c.id] }))}
                  aria-expanded={open}
                  className="flex w-full items-center gap-2.5 px-3 py-2.5 text-left transition hover:bg-slate-50"
                >
                  <span className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br text-[11px] font-black text-white ${COVER[ci % COVER.length]}`}>
                    {initials(c.title)}
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-[13px] font-bold tracking-tight">{c.title}</span>
                    <span className="mt-0.5 flex items-center gap-2">
                      <span className="h-1 w-20 overflow-hidden rounded-full bg-slate-100">
                        <span className="block h-full rounded-full bg-[#ff5a5f]" style={{ width: `${cpct}%` }} />
                      </span>
                      <span className="text-[11px] font-semibold text-slate-500 tabular-nums">{cpct}%</span>
                    </span>
                  </span>
                  {open ? <FaChevronDown className="h-3 w-3 text-slate-400" /> : <FaChevronRight className="h-3 w-3 text-slate-400" />}
                </button>

                {open && (
                  <div className="border-t border-slate-100 px-1.5 py-1.5">
                    {(c.modules || []).map((m) => {
                      const key = `${c.id}:${m.id}`;
                      const mOpen = needle ? true : !!openModules[key];
                      const doneCount = (m.lessons || []).filter((l) => doneIds.has(l.id)).length;
                      return (
                        <div key={key} className="mb-0.5">
                          <button
                            onClick={() => setOpenModules((p) => ({ ...p, [key]: !p[key] }))}
                            aria-expanded={mOpen}
                            className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left transition hover:bg-slate-50"
                          >
                            {mOpen ? <FaChevronDown className="h-2.5 w-2.5 text-slate-400" /> : <FaChevronRight className="h-2.5 w-2.5 text-slate-400" />}
                            <span className="min-w-0 flex-1 truncate text-xs font-bold text-slate-700">{m.title}</span>
                            <span className="rounded-md bg-slate-100 px-1.5 py-0.5 text-[10px] font-bold text-slate-500 tabular-nums">{doneCount}/{m.lessons.length}</span>
                          </button>
                          {mOpen && (
                            <ul className="mt-0.5 mb-1 ml-3.5 space-y-0.5 border-l border-slate-200 pl-2">
                              {(m.lessons || []).map((l) => {
                                const st = statusOf(l.id);
                                const sel = selected?.lessonId === l.id && selected?.courseId === c.id;
                                return (
                                  <li key={l.id}>
                                    <button
                                      onClick={() => onSelect && onSelect(c.id, m.id, l.id)}
                                      title={l.title}
                                      aria-current={sel}
                                      className={`group flex w-full items-center gap-2 rounded-lg px-2 py-[7px] text-left transition ${
                                        sel ? "bg-[#ff5a5f] text-white shadow-[0_6px_16px_rgba(255,90,95,0.35)]" : "hover:bg-red-50"
                                      }`}
                                    >
                                      <span className={`flex h-4.5 w-4.5 h-[18px] w-[18px] shrink-0 items-center justify-center rounded-full border text-[10px] ${
                                        st === "done"
                                          ? sel ? "border-emerald-300 bg-emerald-400/20 text-emerald-200" : "border-emerald-500 bg-emerald-500 text-white"
                                          : st === "current"
                                            ? sel ? "border-white/60 text-white" : "border-[#ff5a5f] text-[#ff5a5f]"
                                            : sel ? "border-white/30 text-white/70" : "border-slate-300 text-transparent"
                                      }`}>
                                        {st === "done" ? <FaCheck className="h-2.5 w-2.5" /> : st === "current" ? <FaCirclePlay className="h-2.5 w-2.5" /> : <span className="h-1 w-1 rounded-full bg-current" />}
                                      </span>
                                      <span className={`min-w-0 flex-1 truncate text-[13px] ${sel ? "font-bold" : st === "done" ? "text-slate-500 line-through decoration-slate-300" : "font-medium text-slate-700"}`}>
                                        {l.title}
                                      </span>
                                      <span className={`shrink-0 text-[10px] font-semibold tabular-nums ${sel ? "text-white/60" : "text-slate-400"}`}>{l.durationMin}m</span>
                                    </button>
                                  </li>
                                );
                              })}
                            </ul>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </div>
        <div className="border-t border-slate-200/70 px-4 py-2.5 text-[11px] text-slate-500">
          Voice + whiteboard learning · interrupt anytime
        </div>
      </aside>

      {/* ── Detail ──────────────────────────────── */}
      <section className="flex min-w-0 flex-1 flex-col bg-white">
        {!selLesson ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-2 p-8 text-center">
            <span className="flex h-12 w-12 items-center justify-center rounded-2xl bg-slate-100"><FaBookOpen className="h-5 w-5 text-slate-400" /></span>
            <p className="text-sm font-bold">Select a lesson</p>
            <p className="max-w-60 text-xs leading-5 text-slate-500">Choose a course, module, and lesson on the left to see the overview.</p>
          </div>
        ) : (
          <>
            {/* hero — light, red accent, no black */}
            <div className="shrink-0 border-b border-red-100 bg-gradient-to-br from-red-50 via-white to-white px-6 pt-5 pb-4">
              <p className="text-[11px] font-bold tracking-[0.14em] text-[#ff5a5f] uppercase">
                {selCourse?.title} <span className="mx-1 text-slate-300">/</span> <span className="text-slate-500">{selModule?.title}</span>
              </p>
              <h1 className="mt-1 text-xl font-bold tracking-tight text-slate-900">{selLesson.title}</h1>
              <div className="mt-2.5 flex flex-wrap items-center gap-2 text-[11px]">
                <span className={`rounded-full px-2.5 py-1 font-bold ring-1 ${KIND_STYLE[selLesson.kind] || KIND_STYLE.theory}`}>{KIND_LABEL[selLesson.kind] || selLesson.kind}</span>
                {selLesson.durationMin && (
                  <span className="flex items-center gap-1.5 rounded-full border border-slate-200 bg-white px-2.5 py-1 font-semibold text-slate-600">
                    <FaClock className="h-3 w-3 text-[#ff5a5f]" /> {selLesson.durationMin} min
                  </span>
                )}
                <span className="flex items-center gap-1.5 rounded-full border border-slate-200 bg-white px-2.5 py-1 font-semibold text-slate-600">
                  <FaListUl className="h-3 w-3 text-[#ff5a5f]" /> Lesson {(idx >= 0 ? idx + 1 : 1)} of {all.length} · {totalMins} min total
                </span>
                {selLesson.id === activeLessonId && (
                  <span className="rounded-full bg-red-50 px-2.5 py-1 font-bold text-[#ff5a5f] ring-1 ring-red-200">● In progress</span>
                )}
              </div>
              <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-red-100">
                <div className="h-full rounded-full bg-[#ff5a5f] transition-all" style={{ width: `${pct}%` }} />
              </div>
              <p className="mt-1.5 text-[11px] font-semibold text-slate-500 tabular-nums">{pct}% of course complete</p>
            </div>

            {/* body */}
            <div className="min-h-0 flex-1 overflow-y-auto px-6 py-4">
              <div className="grid gap-3">
                <div className="rounded-xl border border-slate-200/80 p-4 shadow-[0_1px_2px_rgba(15,23,42,0.05)]">
                  <p className="text-[11px] font-bold tracking-[0.12em] text-slate-400 uppercase">Objective</p>
                  <p className="mt-1 text-[13px] leading-6 text-slate-700">{selLesson.objective}</p>
                </div>
                <div className="rounded-xl border border-slate-200/80 p-4 shadow-[0_1px_2px_rgba(15,23,42,0.05)]">
                  <p className="text-[11px] font-bold tracking-[0.12em] text-slate-400 uppercase">What you will learn</p>
                  <p className="mt-1 text-[13px] leading-6 text-slate-700">{selLesson.summary}</p>
                  {(selLesson.keyTerms?.length > 0) && (
                    <div className="mt-2.5 flex flex-wrap gap-1.5 border-t border-slate-100 pt-2.5">
                      {selLesson.keyTerms.map((k) => (
                        <span key={k} className="rounded-md border border-slate-200 bg-slate-50 px-2 py-1 font-mono text-[11px] font-semibold text-slate-600">{k}</span>
                      ))}
                    </div>
                  )}
                </div>
                {(selLesson.boardOutline?.length > 0) && (
                  <div className="rounded-xl border border-slate-200/80 p-4 shadow-[0_1px_2px_rgba(15,23,42,0.05)]">
                    <p className="text-[11px] font-bold tracking-[0.12em] text-slate-400 uppercase">Whiteboard plan</p>
                    <ol className="mt-2 space-y-1.5">
                      {selLesson.boardOutline.map((s, i) => (
                        <li key={i} className="flex items-start gap-2.5 text-[13px] text-slate-700">
                          <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-[#ff5a5f] text-[10px] font-black text-white tabular-nums">{i + 1}</span>
                          <span className="leading-5">{s}</span>
                        </li>
                      ))}
                    </ol>
                  </div>
                )}
                {selLesson.quiz && (
                  <div className="rounded-xl border border-amber-200 bg-amber-50/60 p-4">
                    <p className="text-[11px] font-bold tracking-[0.12em] text-amber-700 uppercase">Checkpoint quiz</p>
                    <p className="mt-1 text-[13px] font-bold text-slate-900">{selLesson.quiz.q}</p>
                    <p className="mt-1 text-xs text-slate-500">{(selLesson.quiz.options || []).length} options · answered on the board after the lesson</p>
                  </div>
                )}
              </div>
            </div>

            {/* footer */}
            <div className="shrink-0 border-t border-slate-200/80 bg-slate-50/70 px-6 py-3">
              <div className="flex items-center gap-2">
                <button
                  onClick={() => prev && onSelect && onSelect(prev.courseId, prev.moduleId, prev.id)}
                  disabled={!prev}
                  className="flex h-10 w-10 items-center justify-center rounded-xl border border-slate-200 bg-white text-slate-600 transition hover:border-[#ff5a5f] hover:text-[#ff5a5f] disabled:opacity-30"
                  aria-label="Previous lesson"
                  title={prev ? `Previous: ${prev.title}` : "No previous lesson"}
                >
                  <FaArrowLeft className="h-3.5 w-3.5" />
                </button>
                <button
                  onClick={() => onStartLesson && onStartLesson(selCourse.id, selModule.id, selLesson.id)}
                  title={selLesson.id === activeLessonId ? "Resume this lesson with voice + whiteboard" : `Start learning: ${selLesson.title}`}
                  className="flex h-10 flex-1 items-center justify-center gap-2 rounded-xl bg-[#ff5a5f] px-4 text-[13px] font-bold text-white shadow-[0_8px_20px_rgba(255,90,95,0.35)] transition hover:brightness-95 active:scale-[0.99]"
                >
                  <FaCirclePlay className="h-4 w-4" />
                  {selLesson.id === activeLessonId ? "Continue learning" : "Start learning"}
                </button>
                <button
                  onClick={() => next && onSelect && onSelect(next.courseId, next.moduleId, next.id)}
                  disabled={!next}
                  className="flex h-10 w-10 items-center justify-center rounded-xl border border-slate-200 bg-white text-slate-600 transition hover:border-[#ff5a5f] hover:text-[#ff5a5f] disabled:opacity-30"
                  aria-label="Next lesson"
                  title={next ? `Next: ${next.title}` : "No next lesson"}
                >
                  <FaArrowRight className="h-3.5 w-3.5" />
                </button>
              </div>
              <div className="mt-2 flex items-center justify-center">
                {doneIds.has(selLesson.id) ? (
                  <span className="flex items-center gap-1.5 text-[11px] font-bold text-emerald-600">
                    <FaCircleCheck className="h-3.5 w-3.5" /> Completed
                    {next && onCompleteLesson && (
                      <button
                        onClick={() => onCompleteLesson && onCompleteLesson(selCourse.id, selModule.id, selLesson.id, true)}
                        className="font-semibold text-slate-500 underline underline-offset-2 hover:text-[#ff5a5f]"
                      >
                        Continue to next
                      </button>
                    )}
                  </span>
                ) : (
                  <button
                    onClick={() => onCompleteLesson && onCompleteLesson(selCourse.id, selModule.id, selLesson.id, false)}
                    className="text-[11px] font-semibold text-slate-500 underline underline-offset-2 transition hover:text-[#ff5a5f]"
                  >
                    Mark complete & continue
                  </button>
                )}
              </div>
            </div>
          </>
        )}
      </section>
    </div>
  );
}
