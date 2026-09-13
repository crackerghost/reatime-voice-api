import { useEffect, useRef, useState } from "react";
import {
  FaPlay, FaPlus, FaTrash, FaFileCode, FaEye, FaTerminal,
  FaEraser, FaFolderOpen, FaFolder, FaChevronRight, FaChevronDown,
} from "react-icons/fa6";

const MONACO_VERSION = "0.52.2";
const STORE_KEY = "bugos-code";

const STARTER = [
  {
    path: "index.html",
    content: `<link rel="stylesheet" href="css/styles.css">\n<h1>Hello from Bug OS \u{1F41E}</h1>\n<p>Edit any file — folders work, preview updates live.</p>\n<button id="btn">Click me</button>\n<script src="js/app.js"><\/script>\n`,
  },
  {
    path: "css/styles.css",
    content: `body { font-family: system-ui, sans-serif; padding: 32px; background: #f8fafc; color: #0f172a; }\nh1 { color: #e11d48; }\nbutton { background: #0f172a; color: #fff; border: 0; border-radius: 10px; padding: 10px 18px; cursor: pointer; }\n`,
  },
  {
    path: "js/app.js",
    content: `console.log("preview booted");\ndocument.getElementById("btn").addEventListener("click", () => {\n  console.log("button clicked at", new Date().toLocaleTimeString());\n});\n`,
  },
];

const langOf = (path) => {
  if (path.endsWith(".html")) return "html";
  if (path.endsWith(".css")) return "css";
  if (path.endsWith(".json")) return "json";
  if (path.endsWith(".md")) return "markdown";
  if (path.endsWith(".ts")) return "typescript";
  if (path.endsWith(".py")) return "python";
  return "javascript";
};

const isKeep = (f) => f.path.split("/").pop() === ".keep";
const visibleFiles = (files) => files.filter((f) => !isKeep(f));

function loadMonaco() {
  if (window.monaco) return Promise.resolve(window.monaco);
  if (window.__bugosMonacoLoading) return window.__bugosMonacoLoading;
  window.__bugosMonacoLoading = new Promise((resolve, reject) => {
    const loader = document.createElement("script");
    loader.src = `https://cdn.jsdelivr.net/npm/monaco-editor@${MONACO_VERSION}/min/vs/loader.js`;
    loader.onload = () => {
      window.require.config({
        paths: { vs: `https://cdn.jsdelivr.net/npm/monaco-editor@${MONACO_VERSION}/min/vs` },
      });
      window.require(["vs/editor/editor.main"], () => resolve(window.monaco));
    };
    loader.onerror = () => reject(new Error("monaco CDN failed"));
    document.head.appendChild(loader);
  });
  return window.__bugosMonacoLoading;
}

// Intercepts console.* + uncaught errors inside the preview iframe.
const CONSOLE_HOOK = `<script>(function(){function fmt(a){try{return typeof a==="object"?JSON.stringify(a):String(a)}catch(e){return String(a)}}function send(m,args){parent.postMessage({__bugos:1,method:m,text:args.map(fmt).join(" ")},"*")}["log","info","warn","error"].forEach(function(m){var o=console[m].bind(console);console[m]=function(){try{send(m,[].slice.call(arguments))}catch(e){}return o.apply(null,arguments)}});window.addEventListener("error",function(e){send("error",[e.message])});send("info",["preview ready"])})();<\/script>`;

function buildSrcDoc(files) {
  const get = (n) => files.find((f) => f.path === n || f.path.endsWith(`/${n}`))?.content || "";
  const css = files
    .filter((f) => f.path.endsWith(".css"))
    .map((f) => `<style>${f.content}</style>`)
    .join("\n");
  const js = files
    .filter((f) => f.path.endsWith(".js"))
    .map((f) => `<script>${f.content}<\/script>`)
    .join("\n");
  const html = get("index.html") || "<body></body>";
  const body = `${CONSOLE_HOOK}\n${css}\n${js}\n`;
  return html.includes("</body>") ? html.replace("</body>", `${body}</body>`) : `${html}\n${body}`;
}

function buildTree(files) {
  const root = { folders: [], files: [] };
  const map = new Map();
  const nodeFor = (parts) => {
    const key = parts.join("/");
    if (!map.has(key)) {
      const node = { name: parts[parts.length - 1], path: key, folders: [], files: [] };
      map.set(key, node);
      if (parts.length === 1) root.folders.push(node);
      else nodeFor(parts.slice(0, -1)).folders.push(node);
    }
    return map.get(key);
  };
  for (const f of visibleFiles(files)) {
    const parts = f.path.split("/");
    if (parts.length === 1) root.files.push(f);
    else nodeFor(parts.slice(0, -1)).files.push(f);
  }
  const sortAll = (n) => {
    n.folders.sort((a, b) => a.name.localeCompare(b.name));
    n.files.sort((a, b) => a.path.localeCompare(b.path));
    n.folders.forEach(sortAll);
  };
  sortAll(root);
  return root;
}

/* Code editor app: VS Code black + white, folder tree, Monaco, live preview.
   Persists locally. Reports the active file upward for tutor context. */
export default function CodeApp({ onContext }) {
  const [files, setFiles] = useState(() => {
    try {
      const v = JSON.parse(localStorage.getItem(STORE_KEY) || "null");
      if (Array.isArray(v) && visibleFiles(v).length) return v;
    } catch {
      /* fresh */
    }
    return STARTER;
  });
  const [activePath, setActivePath] = useState("index.html");
  const [expanded, setExpanded] = useState({ "": true });
  const [rightTab, setRightTab] = useState("preview");
  const [logs, setLogs] = useState([]);
  const [srcDoc, setSrcDoc] = useState("");
  const [ready, setReady] = useState(false);
  const [failed, setFailed] = useState(false);
  const mountRef = useRef(null);
  const monacoRef = useRef(null);
  const editorRef = useRef(null);
  const modelsRef = useRef({});
  const filesRef = useRef(files);
  const iframeRef = useRef(null);
  filesRef.current = files;

  const tree = buildTree(files);
  const allFolders = [];
  const collect = (n) => {
    n.folders.forEach((f) => {
      allFolders.push(f.path);
      collect(f);
    });
  };
  collect(tree);
  const isOpen = (p) => expanded[p] !== false;

  // Persist + report context.
  useEffect(() => {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify(files));
    } catch {
      /* private mode */
    }
    const active = files.find((f) => f.path === activePath);
    onContext &&
      onContext(
        active ? { file: active.path, lang: langOf(active.path), chars: active.content.length } : null,
      );
  }, [files, activePath, onContext]);

  // Boot Monaco once (VS Code dark).
  useEffect(() => {
    let dead = false;
    loadMonaco()
      .then((monaco) => {
        if (dead) return;
        monacoRef.current = monaco;
        const first = filesRef.current.find((f) => f.path === "index.html") || visibleFiles(filesRef.current)[0];
        const model = monaco.editor.createModel(first.content, langOf(first.path));
        modelsRef.current[first.path] = model;
        editorRef.current = monaco.editor.create(mountRef.current, {
          model,
          theme: "vs-dark",
          fontSize: 14,
          minimap: { enabled: false },
          automaticLayout: true,
          scrollBeyondLastLine: false,
          padding: { top: 12 },
        });
        editorRef.current.onDidChangeModelContent(() => {
          const path = editorRef.current.__bugosPath;
          const value = editorRef.current.getValue();
          setFiles((prev) => prev.map((f) => (f.path === path ? { ...f, content: value } : f)));
        });
        editorRef.current.__bugosPath = first.path;
        setReady(true);
      })
      .catch(() => {
        if (!dead) setFailed(true);
      });
    return () => {
      dead = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Console messages from the preview iframe.
  useEffect(() => {
    const onMsg = (e) => {
      if (!e.data || e.data.__bugos !== 1) return;
      if (iframeRef.current && e.source !== iframeRef.current.contentWindow) return;
      setLogs((prev) => [...prev.slice(-199), { id: Date.now() + Math.random(), method: e.data.method, text: e.data.text }]);
    };
    window.addEventListener("message", onMsg);
    return () => window.removeEventListener("message", onMsg);
  }, []);

  const openFile = (path) => {
    const monaco = monacoRef.current;
    const ed = editorRef.current;
    if (monaco && ed) {
      const file = filesRef.current.find((f) => f.path === path);
      if (!file) return;
      if (!modelsRef.current[path]) {
        modelsRef.current[path] = monaco.editor.createModel(file.content, langOf(path));
      } else {
        modelsRef.current[path].setValue(file.content);
      }
      ed.setModel(modelsRef.current[path]);
      ed.__bugosPath = path;
    }
    setActivePath(path);
  };

  const addFile = (base) => {
    const name = prompt("New file (folders with /, e.g. js/util.js):", base ? `${base}/untitled.js` : "untitled.js");
    if (!name) return;
    const path = name.trim();
    if (!path || path.endsWith("/") || filesRef.current.some((f) => f.path === path)) return;
    setFiles((prev) => [...prev, { path, content: "" }]);
    const folder = path.split("/").slice(0, -1).join("/");
    if (folder) setExpanded((p) => ({ ...p, [folder]: true }));
    openFile(path);
  };

  const addFolder = (base) => {
    const name = prompt("New folder:", base || "assets");
    if (!name) return;
    const path = name.trim().replace(/\/+$/, "");
    if (!path) return;
    setFiles((prev) => [...prev, { path: `${path}/.keep`, content: "" }]);
    setExpanded((p) => ({ ...p, [path]: true }));
  };

  const deleteFile = (path) => {
    if (visibleFiles(filesRef.current).length <= 1) return;
    if (!confirm(`Delete ${path}?`)) return;
    setFiles((prev) => {
      const next = prev.filter((f) => f.path !== path);
      if (path === activePath) {
        const rest = visibleFiles(next);
        if (rest.length) openFile(rest[0].path);
      }
      return next;
    });
    const m = modelsRef.current[path];
    if (m) {
      m.dispose();
      delete modelsRef.current[path];
    }
  };

  const run = () => {
    setLogs([]);
    setSrcDoc(buildSrcDoc(filesRef.current));
  };

  // Live preview: rebuild shortly after edits stop.
  useEffect(() => {
    const t = setTimeout(() => setSrcDoc(buildSrcDoc(files)), 900);
    return () => clearTimeout(t);
  }, [files]);

  const logColor = (m) =>
    m === "error" ? "text-rose-400" : m === "warn" ? "text-amber-400" : m === "info" ? "text-sky-400" : "text-gray-300";

  const fileRow = (f) => (
    <div
      key={f.path}
      onClick={() => openFile(f.path)}
      className={`group flex cursor-pointer items-center gap-1.5 rounded-md px-2 py-1 text-[13px] transition ${
        f.path === activePath ? "bg-[#37373d] text-white" : "text-[#cccccc] hover:bg-[#2a2d2e]"
      }`}
    >
      {f.path === activePath && <span className="absolute left-0 h-5 w-[3px] rounded-r bg-[#007acc]" />}
      <FaFileCode className="h-3 w-3 shrink-0 text-[#519aba]" />
      <span className="min-w-0 flex-1 truncate">{f.path.split("/").pop()}</span>
      <button
        onClick={(e) => {
          e.stopPropagation();
          deleteFile(f.path);
        }}
        aria-label={`Delete ${f.path}`}
        className="hidden rounded p-0.5 text-gray-500 transition group-hover:block hover:bg-rose-900/50 hover:text-rose-300"
      >
        <FaTrash className="h-3 w-3" />
      </button>
    </div>
  );

  const folderNode = (node, depth) => (
    <div key={node.path}>
      <div
        onClick={() => setExpanded((p) => ({ ...p, [node.path]: !isOpen(node.path) }))}
        onContextMenu={(e) => {
          e.preventDefault();
          addFile(node.path);
        }}
        title="Click to fold · right-click for new file here"
        className="flex cursor-pointer items-center gap-1 rounded-md px-2 py-1 text-[13px] font-semibold text-[#bbbbbb] transition hover:bg-[#2a2d2e]"
        style={{ paddingLeft: 8 + depth * 12 }}
      >
        {isOpen(node.path) ? <FaChevronDown className="h-2.5 w-2.5" /> : <FaChevronRight className="h-2.5 w-2.5" />}
        {isOpen(node.path) ? <FaFolderOpen className="h-3.5 w-3.5 text-[#c09553]" /> : <FaFolder className="h-3.5 w-3.5 text-[#c09553]" />}
        <span className="truncate">{node.name}</span>
      </div>
      {isOpen(node.path) && (
        <div>
          {node.folders.map((c) => folderNode(c, depth + 1))}
          <div style={{ paddingLeft: 12 + depth * 12 }}>{node.files.map(fileRow)}</div>
        </div>
      )}
    </div>
  );

  return (
    <div className="flex h-full min-h-0 gap-0 overflow-hidden bg-[#1e1e1e] text-[#cccccc]">
      {/* explorer */}
      <div className="flex w-48 shrink-0 flex-col border-r border-[#2d2d2d] bg-[#252526]">
        <div className="flex items-center justify-between px-3 pt-3 pb-1 text-[11px] font-bold tracking-wider text-[#bbbbbb] uppercase">
          <span>Explorer</span>
          <span className="flex gap-1">
            <button onClick={() => addFile()} title="New file" className="rounded p-1 transition hover:bg-[#37373d] hover:text-white">
              <FaPlus className="h-3 w-3" />
            </button>
            <button onClick={() => addFolder()} title="New folder" className="rounded p-1 transition hover:bg-[#37373d] hover:text-white">
              <FaFolder className="h-3 w-3" />
            </button>
          </span>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto px-1.5 pb-2">
          {tree.folders.map((n) => folderNode(n, 0))}
          <div>{tree.files.map(fileRow)}</div>
        </div>
      </div>

      {/* editor */}
      <div className="flex min-w-0 flex-1 flex-col bg-[#1e1e1e]">
        <div className="flex shrink-0 items-center gap-1 bg-[#252526] px-2 py-1">
          <span className="truncate px-1 font-mono text-xs text-[#bbbbbb]">{activePath}</span>
          <button
            onClick={run}
            className="ml-auto flex items-center gap-1 rounded-md bg-[#0e639c] px-2.5 py-1 text-xs font-bold text-white transition hover:bg-[#1177bb]"
          >
            <FaPlay className="h-3 w-3" /> Run
          </button>
        </div>
        <div ref={mountRef} className="min-h-0 flex-1">
          {!ready && !failed && <p className="p-4 text-xs text-gray-500">Loading editor…</p>}
          {failed && <p className="p-4 text-xs text-rose-400">Editor CDN unreachable — check network.</p>}
        </div>
      </div>

      {/* preview + console */}
      <div className="flex w-[38%] shrink-0 flex-col border-l border-[#2d2d2d] bg-[#1e1e1e]">
        <div className="flex shrink-0 items-center gap-1 bg-[#252526] px-2 py-1">
          {[
            { id: "preview", label: "Preview", icon: <FaEye /> },
            { id: "console", label: `Console${logs.length ? ` (${logs.length})` : ""}`, icon: <FaTerminal /> },
          ].map((t) => (
            <button
              key={t.id}
              onClick={() => setRightTab(t.id)}
              className={`flex items-center gap-1 border-t-2 px-2.5 py-1 text-xs transition ${
                rightTab === t.id
                  ? "border-[#007acc] bg-[#1e1e1e] text-white"
                  : "border-transparent text-gray-500 hover:text-gray-300"
              }`}
            >
              {t.icon} {t.label}
            </button>
          ))}
          {rightTab === "console" && logs.length > 0 && (
            <button
              onClick={() => setLogs([])}
              aria-label="Clear console"
              className="ml-auto rounded p-1.5 text-gray-500 transition hover:bg-[#37373d] hover:text-white"
            >
              <FaEraser className="h-3 w-3" />
            </button>
          )}
        </div>
        {rightTab === "preview" ? (
          <iframe
            ref={iframeRef}
            title="Preview"
            sandbox="allow-scripts"
            srcDoc={srcDoc}
            className="min-h-0 flex-1 border-0 bg-white"
          />
        ) : (
          <div className="min-h-0 flex-1 overflow-y-auto bg-[#1e1e1e] p-2 font-mono text-[11px] leading-5">
            {logs.length === 0 && <p className="text-gray-600">console output lands here…</p>}
            {logs.map((l) => (
              <p key={l.id} className={logColor(l.method)}>
                <span className="opacity-60">[{l.method}]</span> {l.text}
              </p>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
