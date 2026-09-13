import { useState } from "react";
import {
  FaArrowLeft, FaArrowRight, FaRotateRight, FaHouse, FaLock,
  FaMagnifyingGlass, FaPlus, FaXmark, FaArrowUpRightFromSquare,
} from "react-icons/fa6";

const GOOGLE_HOME = "https://www.google.com/webhp?igu=1";
const NEWTAB = "bugos:newtab";

const QUICK = [
  { name: "Google", domain: "google.com", url: "https://www.google.com/webhp?igu=1" },
  { name: "YouTube", domain: "youtube.com", url: "https://www.youtube.com" },
  { name: "Wikipedia", domain: "wikipedia.org", url: "https://en.wikipedia.org" },
  { name: "GitHub", domain: "github.com", url: "https://github.com" },
  { name: "MDN", domain: "developer.mozilla.org", url: "https://developer.mozilla.org" },
  { name: "Stack Overflow", domain: "stackoverflow.com", url: "https://stackoverflow.com" },
];

const favicon = (domain) => `https://www.google.com/s2/favicons?sz=64&domain=${domain}`;

function toUrl(raw) {
  const t = (raw || "").trim();
  if (!t) return GOOGLE_HOME;
  if (/^https?:\/\//i.test(t)) return t;
  if (/^[\w-]+(\.[\w-]+)+(\/\S*)?$/.test(t) && !t.includes(" ")) return `https://${t}`;
  return `https://www.google.com/search?q=${encodeURIComponent(t)}&igu=1`;
}

function hostOf(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return "New tab";
  }
}

let tabSeq = 1;
const makeTab = (url) => ({ id: `t${tabSeq++}`, history: [url || GOOGLE_HOME], idx: 0, reload: 0 });

/* Professional tabbed browser: toolbar, omnibox, tabs, new-tab page.
   Active-tab URL is shared upward so the tutor sees what you browse. */
export default function BrowserApp({ url, onNavigate }) {
  const [tabs, setTabs] = useState(() => [makeTab(url || GOOGLE_HOME)]);
  const [activeId, setActiveId] = useState(() => "t1");
  const [draft, setDraft] = useState(url || GOOGLE_HOME);
  const [hero, setHero] = useState("");

  const active = tabs.find((t) => t.id === activeId) || tabs[0];
  const current = active.history[active.idx];
  const canBack = active.idx > 0;
  const canFwd = active.idx < active.history.length - 1;

  const commit = (next) => {
    const cur = next.find((t) => t.id === activeId) || next[0];
    onNavigate(cur.history[cur.idx]);
  };

  const navigate = (raw) => {
    const target = toUrl(raw);
    setTabs((prev) => {
      const next = prev.map((t) =>
        t.id === activeId
          ? { ...t, history: [...t.history.slice(0, t.idx + 1), target], idx: t.idx + 1 }
          : t,
      );
      commit(next);
      return next;
    });
    setDraft(target);
  };

  const goBack = () => {
    if (!canBack) return;
    setTabs((prev) => {
      const next = prev.map((t) => (t.id === activeId ? { ...t, idx: t.idx - 1 } : t));
      commit(next);
      return next;
    });
    setDraft(active.history[active.idx - 1]);
  };

  const goFwd = () => {
    if (!canFwd) return;
    setTabs((prev) => {
      const next = prev.map((t) => (t.id === activeId ? { ...t, idx: t.idx + 1 } : t));
      commit(next);
      return next;
    });
    setDraft(active.history[active.idx + 1]);
  };

  const reload = () =>
    setTabs((prev) => prev.map((t) => (t.id === activeId ? { ...t, reload: t.reload + 1 } : t)));

  const goHome = () => navigate(GOOGLE_HOME);

  const addTab = () => {
    const t = makeTab(NEWTAB);
    setTabs((prev) => [...prev, t]);
    setActiveId(t.id);
    setDraft("");
    onNavigate(NEWTAB);
  };

  const closeTab = (e, id) => {
    e.stopPropagation();
    setTabs((prev) => {
      if (prev.length === 1) {
        const fresh = [makeTab(NEWTAB)];
        setActiveId(fresh[0].id);
        setDraft("");
        onNavigate(NEWTAB);
        return fresh;
      }
      const next = prev.filter((t) => t.id !== id);
      if (id === activeId) {
        const cur = next[next.length - 1];
        setActiveId(cur.id);
        setDraft(cur.history[cur.idx] === NEWTAB ? "" : cur.history[cur.idx]);
        onNavigate(cur.history[cur.idx]);
      }
      return next;
    });
  };

  const switchTab = (id) => {
    setActiveId(id);
    const t = tabs.find((x) => x.id === id);
    const u = t.history[t.idx];
    setDraft(u === NEWTAB ? "" : u);
    onNavigate(u);
  };

  const submit = (e) => {
    e?.preventDefault();
    if (current === NEWTAB && !draft.trim() && hero.trim()) navigate(hero);
    else if (draft.trim()) navigate(draft);
  };

  const iconBtn =
    "inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-slate-600 transition hover:bg-slate-900/5 disabled:opacity-30";

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* tab strip */}
      <div className="flex shrink-0 items-end gap-1 px-1 pt-1">
        <div className="flex min-w-0 flex-1 items-end gap-1 overflow-x-auto">
          {tabs.map((t) => {
            const u = t.history[t.idx];
            const isNew = u === NEWTAB;
            const on = t.id === activeId;
            return (
              <button
                key={t.id}
                onClick={() => switchTab(t.id)}
                title={isNew ? "New tab" : u}
                className={`flex max-w-44 min-w-0 items-center gap-1.5 rounded-t-xl px-3 py-1.5 text-xs font-medium transition ${
                  on ? "bg-white text-slate-900 shadow-sm" : "text-slate-500 hover:bg-white/50"
                }`}
              >
                {isNew ? (
                  <FaPlus className="h-3 w-3 shrink-0" />
                ) : (
                  <img
                    src={favicon(hostOf(u))}
                    alt=""
                    className="h-3.5 w-3.5 shrink-0 rounded-sm"
                    onError={(e) => {
                      e.currentTarget.style.display = "none";
                    }}
                  />
                )}
                <span className="truncate">{isNew ? "New tab" : hostOf(u)}</span>
                <span
                  role="button"
                  aria-label="Close tab"
                  onClick={(e) => closeTab(e, t.id)}
                  className="rounded-full p-0.5 transition hover:bg-slate-900/10"
                >
                  <FaXmark className="h-3 w-3" />
                </span>
              </button>
            );
          })}
        </div>
        <button onClick={addTab} aria-label="New tab" title="New tab" className={`${iconBtn} mb-0.5`}>
          <FaPlus className="h-3.5 w-3.5" />
        </button>
      </div>

      {/* toolbar */}
      <div className="os-glass flex shrink-0 items-center gap-1 rounded-2xl px-2 py-1.5">
        <button onClick={goBack} disabled={!canBack} aria-label="Back" title="Back" className={iconBtn}>
          <FaArrowLeft className="h-3.5 w-3.5" />
        </button>
        <button onClick={goFwd} disabled={!canFwd} aria-label="Forward" title="Forward" className={iconBtn}>
          <FaArrowRight className="h-3.5 w-3.5" />
        </button>
        <button onClick={reload} aria-label="Reload" title="Reload" className={iconBtn}>
          <FaRotateRight className="h-3.5 w-3.5" />
        </button>
        <button onClick={goHome} aria-label="Home" title="Home (Google)" className={iconBtn}>
          <FaHouse className="h-3.5 w-3.5" />
        </button>
        <form
          onSubmit={submit}
          className="flex min-w-0 flex-1 items-center gap-2 rounded-full bg-white/70 px-3 py-1.5 shadow-inner"
        >
          <FaLock className="h-3 w-3 shrink-0 text-emerald-600" />
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onFocus={(e) => e.target.select()}
            aria-label="Address bar"
            placeholder="Search Google or type a URL"
            className="min-w-0 flex-1 bg-transparent text-sm text-slate-800 outline-none placeholder:text-slate-400"
          />
          {draft.trim() && (
            <button
              type="submit"
              aria-label="Go"
              className="rounded-full bg-slate-900 p-1.5 text-white transition hover:bg-slate-700"
            >
              <FaMagnifyingGlass className="h-3 w-3" />
            </button>
          )}
        </form>
        {current !== NEWTAB && (
          <a
            href={current}
            target="_blank"
            rel="noreferrer"
            aria-label="Open in new tab"
            title="Open in new tab"
            className={iconBtn}
          >
            <FaArrowUpRightFromSquare className="h-3.5 w-3.5" />
          </a>
        )}
      </div>

      {/* page */}
      <div className="relative mt-2 min-h-0 flex-1 overflow-hidden rounded-2xl border border-white/50 bg-white shadow-sm">
        {current === NEWTAB ? (
          <div className="flex h-full flex-col items-center justify-center gap-5 overflow-y-auto bg-gradient-to-b from-white to-slate-100 p-6">
            <p className="text-4xl font-bold tracking-tight">
              <span className="text-[#4285F4]">B</span>
              <span className="text-[#EA4335]">u</span>
              <span className="text-[#FBBC05]">g</span>
              <span className="ml-2 text-slate-700">Search</span>
            </p>
            <form onSubmit={submit} className="flex w-full max-w-md items-center gap-2 rounded-full border border-slate-200 bg-white px-4 py-2.5 shadow-md">
              <FaMagnifyingGlass className="h-4 w-4 shrink-0 text-slate-400" />
              <input
                value={hero}
                onChange={(e) => setHero(e.target.value)}
                aria-label="Search the web"
                placeholder="Search Google…"
                className="min-w-0 flex-1 bg-transparent text-sm text-slate-800 outline-none"
              />
            </form>
            <div className="grid w-full max-w-md grid-cols-3 gap-2">
              {QUICK.map((q) => (
                <button
                  key={q.domain}
                  onClick={() => navigate(q.url)}
                  className="flex flex-col items-center gap-1.5 rounded-2xl border border-slate-100 bg-white px-2 py-3 shadow-sm transition hover:shadow-md"
                >
                  <img
                    src={favicon(q.domain)}
                    alt=""
                    className="h-6 w-6 rounded-md"
                    onError={(e) => {
                      e.currentTarget.style.display = "none";
                    }}
                  />
                  <span className="text-xs font-medium text-slate-600">{q.name}</span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <iframe
            key={`${active.id}-${active.idx}-${active.reload}`}
            src={current}
            title="Browser"
            sandbox="allow-scripts allow-same-origin allow-forms allow-popups"
            className="h-full w-full border-0 bg-white"
          />
        )}
      </div>
      <p className="mt-1 truncate px-1 text-[11px] text-slate-500">
        {current === NEWTAB
          ? "New tab — search above or pick a site."
          : (
            <>
              Tutor sees: <span className="font-medium text-slate-700">{current}</span>
              <span className="text-slate-400"> · blank page? the site blocks embedding — use ↗</span>
            </>
          )}
      </p>
    </div>
  );
}
