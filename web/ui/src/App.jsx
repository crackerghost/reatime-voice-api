import { lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";
import { FaBars, FaClock, FaDesktop, FaMicrophone, FaPaperPlane, FaStop, FaTrash, FaVolumeHigh, FaWandMagicSparkles, FaXmark } from "react-icons/fa6";
import Sidebar from "./Sidebar.jsx";
import BottomBar from "./BottomBar.jsx";
import AuroraBg from "./AuroraBg.jsx";
const DiagramWhiteboard = lazy(() => import("./DiagramWhiteboard.jsx"));
import { engine } from "./audioEngine.js";
import { chatMessage, pingMessage, stopMessage } from "./services/ttsProtocol.js";

const GREETING = "नमस्ते! मैं आपका हिंदी ट्यूटर हूँ। माइक दबाकर बोलिए, लिखकर पूछिए — और स्क्रीन दिखाने के लिए X दबाकर रखें, बोलते रहिए, छोड़ते ही मैं स्क्रीन देखकर जवाब दूँगा। जब मैं बोलूँ तो बीच में कुछ भी बोलिए, मैं तुरंत रुक जाऊँगा।";
const WS_URL = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/tts`;
const ASR_URL = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/asr`;
const CONFIG_URL = `${location.protocol}//${location.host}/api/config`;
const VISION_URL = `${location.protocol}//${location.host}/api/vision`;

/* Live tunables, fetched once from the backend /api/config (which reads them
   from .env). Everything here mirrors an env var server-side, so tuning
   happens in .env — never by editing this file. Defaults below match the
   current behaviour and apply until the fetch resolves. */
const CFG = {
  chatStep: 8, // nfe_step the UI sends for chat replies
  jitterFrames: 1, // start after one TTS frame; raise via VOICE_JITTER_FRAMES if slower hardware underruns
  maxHistory: 12,
  wsReconnectMs: 1500,
  recRestartMs: 400,
  vadTickMs: 50,
  autoSendMs: 750, // end-of-speech silence tail; <500ms chops speech into fragments that each fire a reply
  specChat: true, // server sends a fast "speculative" transcript; reply starts on it, final reconciles
  sendMinChars: 2, // min non-space chars to treat as real words
  vadNoiseFloor: 0.005,
  vadThresholdMin: 0.014,
  vadGateMult: 3.4, // voice gate = noise * this (or threshold_min, whichever is higher)
  vadSustainMs: 250, // energy this long = real voice (barge / arm send)
  vadTextMs: 150, // …or recognizer words + this much energy confirm
  vadFailsafeMs: 650, // 650ms sustained energy barge-in trigger (prevents echo/noise cutting the assistant)
  speakTailMs: 700, // ignore recognition this long after OUR speaker audio stops (echo)
  vadRecActiveMs: 400, // recognizer counts as active within this window
  asrVadMode: "auto", // "server" = Silero VAD on the server owns turn-taking (noise-proof)
  bargeIdleMs: 900, // recognizer-idle safety-net send delay
  visionEnabled: false, // server has a vision engine ready (screen understanding)
  pushMode: true, // push-to-see: hold the button, auto-send on release (SCREEN_PUSH_MODE)
  pushMaxMs: 5000, // hold longer than this -> auto-send anyway
  pushTickMs: 400, // capture cadence DURING the hold (each changed frame warms the VLM)
};
const num = (v, d) => (v === undefined || v === null || Number.isNaN(Number(v)) ? d : Number(v));
const mergeCfg = (c) => {
  if (!c) return;
  CFG.chatStep = num(c.chat_step, CFG.chatStep);
  CFG.jitterFrames = Math.max(0, Math.round(num(c.jitter_frames, CFG.jitterFrames)));
  CFG.maxHistory = num(c.max_history, CFG.maxHistory);
  CFG.wsReconnectMs = num(c.ws_reconnect_ms, CFG.wsReconnectMs);
  CFG.recRestartMs = num(c.rec_restart_ms, CFG.recRestartMs);
  CFG.vadTickMs = num(c.vad_tick_ms, CFG.vadTickMs);
  CFG.autoSendMs = num(c.auto_send_ms, CFG.autoSendMs);
  CFG.sendMinChars = num(c.send_min_chars, CFG.sendMinChars);
  CFG.vadNoiseFloor = num(c.vad_noise_floor, CFG.vadNoiseFloor);
  CFG.vadThresholdMin = num(c.vad_threshold_min, CFG.vadThresholdMin);
  CFG.vadGateMult = num(c.vad_gate_mult, CFG.vadGateMult);
  CFG.vadSustainMs = num(c.vad_sustain_ms, CFG.vadSustainMs);
  CFG.vadTextMs = num(c.vad_text_ms, CFG.vadTextMs);
  CFG.vadFailsafeMs = num(c.vad_failsafe_ms, CFG.vadFailsafeMs);
  CFG.speakTailMs = num(c.speak_tail_ms, CFG.speakTailMs);
  CFG.vadRecActiveMs = num(c.vad_rec_active_ms, CFG.vadRecActiveMs);
  CFG.asrVadMode = c.asr_vad_mode || CFG.asrVadMode;
  CFG.bargeIdleMs = num(c.barge_idle_ms, CFG.bargeIdleMs);
  CFG.specChat = c.spec_chat !== undefined ? !!c.spec_chat : CFG.specChat; // live tunable
  CFG.visionEnabled = c.vision_enabled !== undefined ? !!c.vision_enabled : CFG.visionEnabled;
  CFG.pushMode = c.screen_push_mode !== undefined ? !!c.screen_push_mode : CFG.pushMode;
  CFG.pushMaxMs = num(c.screen_push_max_ms, CFG.pushMaxMs);
  CFG.pushTickMs = num(c.screen_push_tick_ms, CFG.pushTickMs);
};
if (typeof fetch === "function") {
  fetch(CONFIG_URL)
    .then((r) => (r.ok ? r.json() : null))
    .then(mergeCfg)
    .catch(() => {});
}

let msgId = 0;
const nextId = () => ++msgId;

const fmtClock = (ts) =>
  ts ? new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "";
const fmtElapsed = (s) => {
  if (s == null) return "";
  return s >= 60 ? `${Math.floor(s / 60)}m ${Math.round(s % 60)}s` : `${s.toFixed(1)}s`;
};

export default function App() {
  const [messages, setMessages] = useState([{ id: nextId(), role: "assistant", text: GREETING }]);
  const [connected, setConnected] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [listening, setListening] = useState(false);
  const [userTalking, setUserTalking] = useState(false);
  const [interim, setInterim] = useState("");
  const [input, setInput] = useState("");
  const [typing, setTyping] = useState(false);
  const [micBusy, setMicBusy] = useState(false);
  const [sharing, setSharing] = useState(false); // push-to-see capture active (button held)
  const [diagram, setDiagram] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // ---- mutable runtime state (safe across renders) ----
  const wsRef = useRef(null);
  const chatEl = useRef(null);
  const apiRef = useRef({});
  const historyRef = useRef([]);
  const pendingRef = useRef([]);
  const currentSourceRef = useRef(null);
  const speakingRef = useRef(false);
  const dropRef = useRef(false);
  const drainRestartRef = useRef(false); // queue underran mid-reply — next frame starts playback at once
  const drainStartedRef = useRef(false); // drain() already kicked for the current turn (start-event opens speaking early)
  const activeRef = useRef(false); // a reply is being streamed from the server
  const framesRef = useRef(0);
  const turnStartRef = useRef(0); // browser-side: when the current turn was submitted (first-audio stopwatch)
  const activeTurnIdRef = useRef(0);
  const diagramTurnRef = useRef(null);
  const openAssistantId = useRef(null);
  const assistantTextRef = useRef("");
  const asrWsRef = useRef(null); // /ws/asr connection (streaming faster-whisper)
  const asrOpenRef = useRef(false);
  const asrBusyRef = useRef(false); // a "final" transcript is on its way
  const streamingRef = useRef(false); // mic PCM currently being sent to the ASR
  // Push-to-see state: hidden <video> + hold timer + freshest captured frame.
  const sharingRef = useRef(false); // a capture session is active (mirror of `sharing`)
  const screenStreamRef = useRef(null);
  const screenVideoRef = useRef(null);
  const screenTickRef = useRef(0); // capture interval DURING the hold
  const pushTimerRef = useRef(0); // pushMaxMs auto-send cap
  const pushActiveRef = useRef(false); // button currently held down
  const pushFrameRef = useRef(null); // { b64, hash, ts } — newest captured frame
  const pushStartRef = useRef(0); // performance.now() when the hold began
  const pushWarmedHashRef = useRef(""); // hash already sent to /api/vision this hold (dedupe)
  const pushRetryRef = useRef(0); // first-capture retries until video is ready
  const pushReleasePendingRef = useRef(false); // released before the first frame landed -> flush on first frame
  const pushTextRef = useRef(""); // mic question heard DURING the hold (sent with the frame on release)
  const screenCanvasRef = useRef(null); // full-size JPEG encoder canvas
  const screenDiffCanvasRef = useRef(null); // 32x24 signature canvas
  const screenFetchBusyRef = useRef(false); // a warm-up fetch is in flight (never queue)
  const lastSentHashRef = useRef(""); // hash of the frame attached to the last turn
  // VAD state — thresholds/initial values come from CFG (env-driven).
  const vadRef = useRef({
    noise: CFG.vadNoiseFloor, // adaptive ambient floor (updated when idle)
    threshold: Math.max(CFG.vadNoiseFloor * CFG.vadGateMult, CFG.vadThresholdMin),
    hotTicks: 0, // consecutive ticks above open threshold (debounce noise blips)
    strongTicks: 0, // consecutive ticks WELL above the gate (only this may cut the assistant)
    textHeard: false, // recognizer delivered real words during this burst
    anyVoice: false, // analyser saw real voice at least once this burst
    bargeLatched: false, // already cut the assistant during this burst
    voiceSilentSince: 0, // ms timestamp when the burst went silent
    uttEndedAt: 0, // ms timestamp when the utterance was closed (speculative-release timer)
    lastUtt: 0, // recognizer last heard anything (ms)
    lastSpeakAt: 0, // when OUR speaker output last had real audio (echo guard)
    voiceEnergy: null, // leaky-integrator short-term RMS for hysteresis
    streamingRefCur: false, // mirror of streamingRef.current for tick-closed access
    recordingSince: 0, // when the current utterance was opened (diagnostic)
  });
  const listeningRef = useRef(false);
  const userTalkingRef = useRef(false);
  const micToggleRef = useRef(null); // set once by the mic effect; survives re-renders

  // ---------- helpers ----------
  const scrollToBottom = () => {
    const el = chatEl.current;
    if (el) el.scrollTop = el.scrollHeight;
  };
  useEffect(scrollToBottom, [messages, typing]);

  const pushHistory = (role, content) => {
    historyRef.current.push({ role, content });
    if (historyRef.current.length > CFG.maxHistory) historyRef.current.shift();
  };

  const showError = (text) =>
    setMessages((m) => [...m, { id: nextId(), role: "error", text }]);

  /* ---- audio playback with an analyser feeding the globe ---- */
  const playBlob = useCallback(async (blob) => {
    const ctx = engine.unlock();
    const buf = await ctx.decodeAudioData(await blob.arrayBuffer());
    let src;
    try {
      src = ctx.createBufferSource();
      src.buffer = buf;
      engine.connectSpeak(src);
      if (typeof src.start === "function") src.start(); // legacy WebKit fallback
      else src.noteOn(0);
      currentSourceRef.current = src;
      return new Promise((resolve) => {
        src.onended = () => {
          currentSourceRef.current = null;
          try { src.disconnect(); } catch { /* noop */ } // free the audio graph — every frame adds nodes
          resolve();
        };
      });
    } catch {
      try { src && src.disconnect(); } catch { /* noop */ }
      throw new Error("audio play failure");
    }
  }, []);

  /* ---- play an already-decoded buffer (drain prefetches decodes) ---- */
  const playBuf = useCallback((buf) => {
    const ctx = engine.unlock();
    return new Promise((resolve, reject) => {
      const src = ctx.createBufferSource();
      src.buffer = buf;
      engine.connectSpeak(src);
      currentSourceRef.current = src;
      src.onended = () => {
        currentSourceRef.current = null;
        try { src.disconnect(); } catch { /* noop */ } // free the audio graph
        resolve();
      };
      src.onerror = () => {
        currentSourceRef.current = null;
        try { src.disconnect(); } catch { /* noop */ }
        reject(new Error("audio playback error"));
      };
      try {
        src.start();
      } catch (e) {
        currentSourceRef.current = null;
        try { src.disconnect(); } catch { /* noop */ }
        reject(e);
      }
    });
  }, []);

  const drain = useCallback(async () => {
    const ctx = engine.unlock();
    let nxt = null; // AudioBuffer prefetched for the next frame (decoded while current plays)
    const startedAt = Date.now();
    while (pendingRef.current.length > 0) {
      if (dropRef.current) {
        pendingRef.current.length = 0;
        speakingRef.current = false;
        setSpeaking(false);
        return;
      }
      const blob = pendingRef.current.shift();
      // decode the NEXT queued frame in parallel with playing this one, so
      // the handoff is gapless instead of "play -> stop -> decode -> play"
      const pre = (pendingRef.current.length > 0)
        ? ctx.decodeAudioData(await pendingRef.current[0].arrayBuffer()).catch(() => null)
        : Promise.resolve(null);
      let buf = nxt;
      nxt = null;
      if (!buf) {
        try {
          buf = await ctx.decodeAudioData(await blob.arrayBuffer());
        } catch {
          // bad frame — drop it but keep the queue and speaking state healthy
          console.warn("[voice] dropped a corrupt TTS audio frame");
          await pre.catch(() => null); // keep the prefetch warm
          continue;
        }
      }
      try {
        await playBuf(buf);
      } catch {
        // playback failed — stop claiming we're speaking so the UI recovers
        currentSourceRef.current = null;
        speakingRef.current = false;
        setSpeaking(false);
        pendingRef.current = [];
        console.warn("[voice] TTS playback failed; cleared pending frames");
        return;
      }
      nxt = await pre; // cache for the next iteration
    }
    // Queue drained while the reply is still streaming — remember it so the
    // NEXT arriving frame restarts playback immediately instead of waiting
    // for the jitter buffer to refill (which would add dead air).
    if (activeRef.current) drainRestartRef.current = true;
    speakingRef.current = false;
    setSpeaking(false);
  }, [playBuf]);

  /* ---- hard stop: instant audio cut + server cancel (barge-in) ---- */
  const hardStop = useCallback(() => {
    dropRef.current = true; // drop stale frames/text until the next "start"
    drainRestartRef.current = false;
    drainStartedRef.current = false;
    if (currentSourceRef.current) {
      try {
        currentSourceRef.current.stop();
      } catch { /* noop */ }
      currentSourceRef.current = null;
    }
    pendingRef.current.length = 0;
    if (wsRef.current && wsRef.current.readyState === 1) {
      wsRef.current.send(stopMessage());
    }
    speakingRef.current = false;
    setSpeaking(false);
  }, []);

  /* ---- push-to-see: hold X (or the screen button) → capture → describe →
     auto-send on release ----
     Latency design (replaces the continuous 1.2 s screen-share loop):
     1. Press  → the capture stream is already granted (see below), so frames
        start immediately (400 ms cadence, NO change-detection — the user IS
        the trigger). First press of a session shows Chrome's picker ONCE;
        the stream then stays alive silently between presses (a browser can
        never capture without the one-time permission picker).
     2. Every captured frame goes to /api/vision in the BACKGROUND with a
        unique hash, so the Qwen2.5-VL describe runs WHILE the user is still
        holding/talking. The last warm-up's result stays in the server cache.
     3. Release (or 5 s cap) → the FRESHEST frame is attached to the turn as
        {screen: {image, hash, wait_ms}}. A question spoken into the mic
        during the hold is absorbed and sent as the turn text; with no text
        the turn IS the screen ("इस स्क्रीन के बारे में बताओ").
     4. No GPU/network vision cost while X is NOT held — only the idle local
        capture stream (no frames fetched, no uploads). Revoke anytime via
        Chrome's own "Stop sharing" bar. */
  const stopScreenCapture = useCallback(() => {
    pushActiveRef.current = false;
    pushReleasePendingRef.current = false;
    if (pushTimerRef.current) { clearTimeout(pushTimerRef.current); pushTimerRef.current = 0; }
    if (screenTickRef.current) { clearInterval(screenTickRef.current); screenTickRef.current = 0; }
    const stream = screenStreamRef.current;
    if (stream) stream.getTracks().forEach((t) => t.stop());
    screenStreamRef.current = null;
    const vid = screenVideoRef.current;
    if (vid) { try { vid.srcObject = null; } catch { /* noop */ } }
    sharingRef.current = false;
    pushFrameRef.current = null;
    setSharing(false);
  }, []);

  /* Grant the capture stream ONCE and keep it alive between holds. Chrome
     always shows its permission picker on the first getDisplayMedia (no API
     skips it — security rule), but a live stream can be reused silently, so
     every X-hold after the first starts at ZERO picker latency. */
  const ensureCaptureStream = useCallback(async () => {
    const existing = screenStreamRef.current;
    if (existing && existing.getVideoTracks()[0]?.readyState === "live") return existing;
    const stream = await navigator.mediaDevices.getDisplayMedia({
      video: { frameRate: 5 }, audio: false,
    });
    const vid = document.createElement("video");
    vid.srcObject = stream;
    vid.muted = true;
    vid.playsInline = true;
    await vid.play().catch(() => {});
    screenStreamRef.current = stream;
    screenVideoRef.current = vid;
    stream.getVideoTracks()[0]?.addEventListener("ended", () => {
      // user hit Chrome's "Stop sharing" — revoke fully
      stopScreenCapture();
    });
    return stream;
  }, [stopScreenCapture]);

  /* One capture tick while the button is held: grab the frame NOW (no diff
     gate — during a hold every tick is a deliberate capture) and warm the
     vision cache in the background. Never queues: one in-flight fetch max. */
  const pushCaptureTick = () => {
    const vid = screenVideoRef.current;
    if (!vid || !pushActiveRef.current) return;
    if (!vid.videoWidth) {
      // The video element needs a moment after .play() — retry quickly instead
      // of silently dropping the whole hold's warm-up window.
      if (pushRetryRef.current < 10) {
        pushRetryRef.current += 1;
        setTimeout(pushCaptureTick, 100);
      }
      return;
    }
    if (!screenCanvasRef.current) {
      screenCanvasRef.current = document.createElement("canvas");
      screenDiffCanvasRef.current = document.createElement("canvas");
      screenDiffCanvasRef.current.width = 32;
      screenDiffCanvasRef.current.height = 24;
    }
    // encode at 960px/q0.65 — OCR-grade quality, ~2x smaller upload than 1280
    const scale = Math.min(1, 960 / vid.videoWidth);
    const c = screenCanvasRef.current;
    c.width = Math.round(vid.videoWidth * scale);
    c.height = Math.round(vid.videoHeight * scale);
    c.getContext("2d").drawImage(vid, 0, 0, c.width, c.height);
    const b64 = c.toDataURL("image/jpeg", 0.65).split(",")[1];
    // per-tick signature — each distinct screen state gets its own cache key,
    // so the LAST tick's describe is always the freshest one in the cache
    const dctx = screenDiffCanvasRef.current.getContext("2d", { willReadFrequently: true });
    dctx.drawImage(vid, 0, 0, 32, 24);
    const px = dctx.getImageData(0, 0, 32, 24).data;
    let sig = 0;
    for (let i = 0, j = 0; i < 32 * 24; i++, j += 4) {
      sig = (sig * 31 + ((px[j] * 299 + px[j + 1] * 587 + px[j + 2] * 114) / 1000)) | 0;
    }
    const hash = `p${sig >>> 0}`;
    pushFrameRef.current = { b64, hash, ts: Date.now() };
    // Dedupe: an unchanged screen during a hold has the SAME hash — re-POSTing
    // it would only burn GPU re-describing (or OCR-ing) identical pixels. Warm
    // each DISTINCT screen state once per hold.
    if (hash === pushWarmedHashRef.current || screenFetchBusyRef.current) return;
    pushWarmedHashRef.current = hash;
    screenFetchBusyRef.current = true;
    const t0 = performance.now();
    fetch(VISION_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: b64, hash }),
    })
      .then((r) => (r.ok ? r.json() : null))
      .then((j) => j && console.info(
        `[vision] hold warm-up ${j.cached ? "HIT" : "FILLED"} (${Math.round(performance.now() - t0)}ms)`,
      ))
      .catch(() => {})
      .finally(() => { screenFetchBusyRef.current = false; });
  };

  const startPushCapture = useCallback(async () => {
    if (sharingRef.current || pushActiveRef.current) return;
    if (!CFG.visionEnabled) {
      showError("सर्वर पर vision चालू नहीं है — .env में VISION_BACKEND जाँचें।");
      return;
    }
    if (!navigator.mediaDevices?.getDisplayMedia) {
      showError("इस ब्राउज़र में स्क्रीन कैप्चर उपलब्ध नहीं है — Chrome/Edge आज़माएँ।");
      return;
    }
    try {
      // First press of a session: Chrome's picker shows here (unavoidable).
      // Every later press: this resolves instantly from the live stream.
      await ensureCaptureStream();
      sharingRef.current = true;
      pushActiveRef.current = true;
      pushStartRef.current = performance.now();
      pushWarmedHashRef.current = "";
      pushRetryRef.current = 0;
      pushReleasePendingRef.current = false;
      pushTextRef.current = "";
      setSharing(true);
      // first frame IMMEDIATELY, then every pushTickMs until release/cap
      pushCaptureTick();
      screenTickRef.current = setInterval(pushCaptureTick, CFG.pushTickMs);
      // hard cap: a held key auto-sends anyway ("not more than 5 sec")
      pushTimerRef.current = setTimeout(() => {
        if (pushActiveRef.current) {
          console.info("[push] hold exceeded max — auto-sending");
          apiRef.current.releasePush();
        }
      }, CFG.pushMaxMs);
    } catch (e) {
      if (e && e.name === "NotAllowedError") {
        showError("स्क्रीन कैप्चर की अनुमति नहीं मिली — दोबारा कोशिश करें और 'Share' दबाएँ।");
      } else {
        showError("स्क्रीन कैप्चर शुरू नहीं हो पाया: " + (e.message || e.name));
      }
      console.warn("[push] getDisplayMedia failed:", e);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ensureCaptureStream]);  /* Release the held X/button: stop the capture LOOP (the granted stream
     stays alive for the next hold — no picker again), attach the freshest
     frame to a turn, and send. A question spoken into the mic while holding
     becomes the turn text; with no text the turn IS the screen. */
  const releasePush = useCallback(() => {
    if (!pushActiveRef.current) return;
    pushActiveRef.current = false;
    if (pushTimerRef.current) { clearTimeout(pushTimerRef.current); pushTimerRef.current = 0; }
    if (screenTickRef.current) { clearInterval(screenTickRef.current); screenTickRef.current = 0; }
    const heldMs = Math.round(performance.now() - pushStartRef.current);
    const frame = pushFrameRef.current;
    pushFrameRef.current = null;
    sharingRef.current = false;
    setSharing(false);
    if (!frame) {
      // First capture hasn't landed yet (very fast tap / picker just closed):
      // arm a short flush so the first frame STILL becomes the turn instead
      // of being dropped.
      if (pushRetryRef.current > 0 || screenVideoRef.current) {
        pushReleasePendingRef.current = true;
        setTimeout(() => {
          if (pushReleasePendingRef.current && pushFrameRef.current) {
            pushReleasePendingRef.current = false;
            apiRef.current.releasePush();
          }
        }, 300);
      }
      console.warn("[push] released after " + heldMs + "ms but no frame was captured yet — flushing on first frame");
      return;
    }
    console.info(`[push] hold ${heldMs}ms -> turn with frame (${Math.round(frame.b64.length * 3 / 4 / 1024)} KB, hash ${frame.hash})`);
    lastSentHashRef.current = frame.hash;
    // If the release-instant frame was NEVER warmed (identical-frame dedupe,
    // dropped fetch, or a very short tap), describe it now. NOT forced: a
    // cache-first call piggybacks on any in-flight describe of the same hash
    // (which the turn's wait_ms then waits on) and costs 0 ms if a previous
    // hold already described these exact pixels.
    if (frame.hash !== pushWarmedHashRef.current) {
      fetch(VISION_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: frame.b64, hash: frame.hash }),
      }).catch(() => {});
    }
    // A question spoken during the hold wins; typed text second; screen-only
    // (no words) makes the turn "describe what you see".
    const text = (pushTextRef.current || "").trim() || (input || "").trim()
      || "इस स्क्रीन के बारे में बताओ — क्या दिख रहा है?";
    pushTextRef.current = "";
    setInput("");
    submitPushChat(text, frame, heldMs);

    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [input]);

  /* One capture tick (legacy continuous mode): downsample -> change detect ->
     encode JPEG on change -> background warm-up call. */
  const screenTick = () => {
    const vid = screenVideoRef.current;
    if (!vid || !sharingRef.current || !vid.videoWidth) return;
    if (!screenCanvasRef.current) {
      screenCanvasRef.current = document.createElement("canvas");
      screenDiffCanvasRef.current = document.createElement("canvas");
      screenDiffCanvasRef.current.width = 32;
      screenDiffCanvasRef.current.height = 24;
    }
    // 1) downsample to 32x24 gray and build a cheap signature
    const dctx = screenDiffCanvasRef.current.getContext("2d", { willReadFrequently: true });
    dctx.drawImage(vid, 0, 0, 32, 24);
    const px = dctx.getImageData(0, 0, 32, 24).data;
    let prevSig = 0;
    for (let i = 0, j = 0; i < 32 * 24; i++, j += 4) {
      prevSig = (prevSig * 31 + ((px[j] * 299 + px[j + 1] * 587 + px[j + 2] * 114) / 1000)) | 0;
    }
    // 2) skip if the screen looks identical to the last warm-up (cursor blinks
    //    etc. still sneak through — the hash makes the server cache absorb them)
    const hash = `s${prevSig >>> 0}`;
    if (hash === lastSentHashRef.current && !screenFetchBusyRef.current) return;
    // 3) encode the real frame (max 1280px wide, q0.7 ≈ 100-250 KB)
    const scale = Math.min(1, 1280 / vid.videoWidth);
    const c = screenCanvasRef.current;
    c.width = Math.round(vid.videoWidth * scale);
    c.height = Math.round(vid.videoHeight * scale);
    c.getContext("2d").drawImage(vid, 0, 0, c.width, c.height);
    const b64 = c.toDataURL("image/jpeg", 0.7).split(",")[1];
    // 4) background warm-up: describe the NEW screen now so the reply later
    //    hits the server cache and pays ZERO vision latency.
    //    NEVER queue: a local VLM takes 7-15 s per screen. One in-flight max.
    if (screenFetchBusyRef.current) return;
    screenFetchBusyRef.current = true;
    const t0 = performance.now();
    fetch(VISION_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: b64, hash }),
    })
      .then((r) => (r.ok ? r.json() : null))
      .then((j) => j && console.info(`[vision] warm cache ${j.cached ? "HIT" : "FILLED"} (${(j.description || "").length} chars, ${Math.round(performance.now() - t0)}ms)`))
      .catch(() => {})
      .finally(() => { screenFetchBusyRef.current = false; });
  };

  const startScreenShare = useCallback(async () => {
    if (sharingRef.current) return;
    if (!CFG.visionEnabled) {
      showError("सर्वर पर vision चालू नहीं है — .env में VISION_BACKEND जाँचें।");
      return;
    }
    if (!navigator.mediaDevices?.getDisplayMedia) {
      showError("इस ब्राउज़र में स्क्रीन शेयर उपलब्ध नहीं है — Chrome/Edge आज़माएँ।");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: { frameRate: 5 }, audio: false,
      });
      const vid = document.createElement("video");
      vid.srcObject = stream;
      vid.muted = true;
      vid.playsInline = true;
      await vid.play().catch(() => {});
      screenStreamRef.current = stream;
      screenVideoRef.current = vid;
      lastSentHashRef.current = "";
      sharingRef.current = true;
      setSharing(true);
      console.info("[screen] legacy continuous capture loop started (tick " + CFG.screenTickMs + "ms)");
      stream.getVideoTracks()[0]?.addEventListener("ended", () => {
        // user hit "Stop sharing" — tear the loop down
        const s = screenStreamRef.current;
        if (s) s.getTracks().forEach((t) => t.stop());
        screenStreamRef.current = null;
        const v = screenVideoRef.current;
        if (v) { try { v.srcObject = null; } catch { /* noop */ } }
        if (screenTickRef.current) { clearInterval(screenTickRef.current); screenTickRef.current = 0; }
        sharingRef.current = false;
        setSharing(false);
      }); // user hit "Stop sharing"
      screenTickRef.current = setInterval(screenTick, CFG.screenTickMs);
    } catch (e) {
      // VISIBLE failures: a swallowed NotAllowedError made users believe the
      // screen was being shared when the browser had actually blocked it.
      if (e && e.name === "NotAllowedError") {
        showError("स्क्रीन शेयर की अनुमति नहीं मिली — दोबारा कोशिश करें और 'Share' दबाएँ।");
      } else {
        showError("स्क्रीन शेयर शुरू नहीं हो पाया: " + (e.message || e.name));
      }
      console.warn("[screen] getDisplayMedia failed:", e);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* Capture the screen RIGHT NOW (used at question time so the tutor always
     sees the screen as it is when you ask — never a stale cached frame).
     960px/q0.65: OCR reads text fine at this size and the WS upload (and the
     inline OCR) shrink ~2x vs 1280px/q0.7 — pure latency win on the reply. */
  const grabScreen = () => {
    const vid = screenVideoRef.current;
    if (!vid || !sharingRef.current || !vid.videoWidth) return null;
    if (!screenCanvasRef.current) screenCanvasRef.current = document.createElement("canvas");
    const scale = Math.min(1, 960 / vid.videoWidth);
    const c = screenCanvasRef.current;
    c.width = Math.round(vid.videoWidth * scale);
    c.height = Math.round(vid.videoHeight * scale);
    c.getContext("2d").drawImage(vid, 0, 0, c.width, c.height);
    const b64 = c.toDataURL("image/jpeg", 0.65).split(",")[1];
    // signature from the same instant — same pixels = same hash = 0 ms OCR;
    // changed pixels = new hash = server OCRs fresh (~0.5-1 s GPU)
    if (!screenDiffCanvasRef.current) {
      screenDiffCanvasRef.current = document.createElement("canvas");
      screenDiffCanvasRef.current.width = 32;
      screenDiffCanvasRef.current.height = 24;
    }
    const dctx = screenDiffCanvasRef.current.getContext("2d", { willReadFrequently: true });
    dctx.drawImage(vid, 0, 0, 32, 24);
    const px = dctx.getImageData(0, 0, 32, 24).data;
    let sig = 0;
    for (let i = 0, j = 0; i < 32 * 24; i++, j += 4) {
      sig = (sig * 31 + ((px[j] * 299 + px[j + 1] * 587 + px[j + 2] * 114) / 1000)) | 0;
    }
    return { image: b64, b64, hash: `s${sig >>> 0}` };
  };

  /* ---- submit a new user turn ---- */
  const submitChat = useCallback(
    (rawText, screenFrame = null) => {
      const text = (rawText || "").trim();
      if (!text) return;
      hardStop();
      activeTurnIdRef.current += 1;
      diagramTurnRef.current = null;
      setDiagram(null);
      setMessages((m) => [...m, { id: nextId(), role: "user", text, ts: Date.now() }]);
      pushHistory("user", text);

      const ws = wsRef.current;
      if (!ws || ws.readyState !== 1) {
        if (ws && ws.readyState === 0) {
          // Socket is reconnecting: wait 350ms and retry rather than dropping
          // turn — the attached screen frame rides along on the retry too.
          setTimeout(() => submitChat(rawText, screenFrame), 350);
          return;
        }
        showError("सर्वर कनेक्शन नहीं है — रुकिए, फिर से पूछिए।");
        return;
      }
      // dropRef stays true (set by hardStop) until the server's "start" for
      // this reply — any stale frames of the aborted reply are ignored.
      framesRef.current = 0;
      turnStartRef.current = performance.now(); // browser-measured first-audio latency
      activeRef.current = false;
      // SELF-BARGE-IN GUARD: hardStop() just stopped the mic's echo gate; the
      // barge path sees speakingRef=false but an old VAD burst can STILL fire
      // hardStop again AFTER this submission, killing the turn server-side
      // (0 frames, silent reply). Latch barge-in for a moment so our own turn
      // can never cut itself off (a real user interrupt re-arms it via text).
      openAssistantId.current = null;
      assistantTextRef.current = "";
      setTyping(true);
      const payload = {
        text,
        history: historyRef.current,
        nfeStep: CFG.chatStep,
        clientTurnId: activeTurnIdRef.current,
      };
      // Push-to-see: the held button already captured + warmed the frame.
      // Attach it (with wait_ms) so the server waits briefly for the in-flight
      // describe instead of answering blind. Legacy continuous-share path
      // (CFG.pushMode off) still captures at question time below.
      if (screenFrame) {
        payload.screen = {
          image: screenFrame.b64,
          hash: screenFrame.hash,
          wait_ms: 2000, // release-time warm-up gets a bounded window to land
        };
        console.info(
          `[screen] push frame attached (${Math.round(screenFrame.b64.length * 3 / 4 / 1024)} KB, hash ${screenFrame.hash}, hold ${screenFrame.heldMs}ms)`,
        );
      } else if (!CFG.pushMode && sharingRef.current) {
        // PRODUCTION RULE (legacy continuous mode): capture at question time —
        // the tutor always sees the screen as it is RIGHT NOW, automatically.
        const shot = grabScreen();
        if (shot) {
          payload.screen = shot;
          lastSentHashRef.current = shot.hash;
          console.info(`[screen] frame attached to turn (${Math.round(shot.b64.length * 3 / 4 / 1024)} KB, hash ${shot.hash})`);
        } else {
          console.warn("[screen] sharing is ON but grabScreen() returned null — video not ready");
        }
      }
      ws.send(chatMessage({
        ...payload,
        screen: payload.screen,
      }));
    },
    [hardStop]
  );

  /* Push-to-see turn: text + the frame captured during the hold. */
  const submitPushChat = useCallback(
    (text, frame, heldMs) => submitChat(text, { ...frame, heldMs }),
    [submitChat],
  );

  apiRef.current = {
    submitChat,
    hardStop,
    // Push-to-see: press starts a capture session, release sends the turn.
    startPush: () => startPushCapture(),
    releasePush: () => releasePush(),
    pushActiveRef,
    // Legacy continuous-share toggle (SCREEN_PUSH_MODE=0)
    toggleScreenShare: () => startPushCapture(),
    setSpeaking: (v) => {
      speakingRef.current = v;
      setSpeaking(v);
    },
    setUserTalking,
    userTalkingRef,
    // toggleMic lives in a ref (set once by the mic effect); apiRef.current is
    // rebuilt on every render, so we expose a stable dispatcher instead.
    toggleMic: () => micToggleRef.current && micToggleRef.current(),
  };

  /* ---------- ASR readiness: server preloads Whisper at boot ---------- */
  const [asrReady, setAsrReady] = useState(false);
  const [asrReconnecting, setAsrReconnecting] = useState(false);
  // Server rejected the last utterance (speaker-gate or noise blip) — show a
  // transient chip so it's clear WHY nothing happened, then auto-dismiss.
  const [asrRejected, setAsrRejected] = useState("");
  const asrRejTimerRef = useRef(0);
  useEffect(() => () => clearTimeout(asrRejTimerRef.current), []); // unmount cleanup
  useEffect(() => {
    let alive = true;
    let t = 0;
    const check = () => {
      fetch(CONFIG_URL)
        .then((r) => (r.ok ? r.json() : null))
        .then((c) => {
          if (alive && c && c.asr_ready) {
            setAsrReady(true);
            clearInterval(t);
          }
        })
        .catch(() => {});
    };
    check();
    t = setInterval(check, 3000); // re-poll until the warmup finishes
    return () => { alive = false; clearInterval(t); };
  }, []);

  /* ---------- WebSocket ---------- */
  useEffect(() => {
    let closed = false;
    let ttsAttempts = 0;
    let ttsHb = 0;
    let ttsLastPong = 0;
    // playback watchdog: if we claim to be speaking but nothing is actually
    // playing and no new frame has arrived, clear the stuck state so the UI
    // and the echo guard don't stay wrong.
    let speakWatchdog = 0;
    const connect = () => {
      const ws = new WebSocket(WS_URL);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;
      ws.onopen = () => {
        ttsAttempts = 0;
        ttsLastPong = Date.now();
        setConnected(true);
        // heartbeat: detect a dead connection in ~15s instead of uvicorn's 40s
        ttsHb = setInterval(() => {
          try {
            ws.send(pingMessage());
          } catch { /* closed — onclose handles it */ }
          if (Date.now() - ttsLastPong > 15000) {
            try { ws.close(); } catch { /* noop */ } // force reconnect
          }
        }, 5000);
      };
      ws.onerror = () => setConnected(false);
      ws.onclose = () => {
        setConnected(false);
        if (ttsHb) { clearInterval(ttsHb); ttsHb = 0; }
        if (!closed) {
          ttsAttempts += 1;
          // exponential backoff, capped: 1.5s -> 3s -> 6s -> 12s -> 15s
          const delay = Math.min(CFG.wsReconnectMs * 2 ** Math.min(ttsAttempts - 1, 3), 15000);
          setTimeout(connect, delay);
        }
      };
      ws.onmessage = (ev) => {
        const api = apiRef.current;
        if (typeof ev.data === "string") {
          let m;
          try {
            m = JSON.parse(ev.data);
          } catch {
            return;
          }
          if (m.type === "pong") { // heartbeat reply
            ttsLastPong = Date.now();
            return;
          }
          if (m.type === "start") {
            dropRef.current = false;
            activeRef.current = true;
            framesRef.current = 0;
            drainRestartRef.current = false;
            drainStartedRef.current = false;
            setTyping(false);
            // Mark the reply "speaking" from the START of the stream, not
            // when the first sample plays. The mic echo-gates and the server
            // VAD assistant-gate key off this flag; during the jitter-buffer
            // wait (no audio yet) they must already be closed, or the mic
            // hears speaker echo and barge-in kills the reply before it plays.
            if (!speakingRef.current) {
              speakingRef.current = true;
              api.setSpeaking(true);
            }
          } else if (m.type === "window") {
            // Debug: how the LLM stream was divided into TTS chunks + their
            // generation cost. RTF > 1 means that chunk generates slower than
            // it plays — the direct cause of mid-reply underruns.
            console.info(
              `[voice] TTS window #${m.n}: ${m.chars} ch, ${m.steps} steps, speed ${m.speed}` +
              ` | audio ${m.audio_s}s / gen ${m.gen_s}s (RTF ${m.rtf})` +
              ` | "${m.text}"`,
            );
          } else if (m.type === "diagram") {
            if (dropRef.current) return;
            if (m.client_turn_id && m.client_turn_id !== String(activeTurnIdRef.current)) return;
            // Whole-board (legacy) or window delta (watcher sidecar) — both merge by id.
            const incoming = m.diagram?.elements?.length ? m.diagram.elements : (m.elements?.length ? m.elements : null);
            if (!incoming) return;
            if (m.turn_id) diagramTurnRef.current = m.turn_id;
            console.info(`[diagram] ${m.mode === "append" ? `delta window #${m.window_n ?? "?"}` : "board"}: +${incoming.length} element(s)`);
            setDiagram((prev) => {
              const seen = new Set();
              const merged = [];
              for (const el of [...(prev?.elements || []), ...incoming]) {
                if (!el || !el.id || seen.has(el.id)) continue;
                seen.add(el.id);
                merged.push(el);
              }
              return { elements: merged.slice(-40) };
            });
          } else if (m.type === "diagram_error") {
            if (!dropRef.current) console.info("[diagram] visual explanation unavailable", m.message || "");
          } else if (m.type === "text") {
            if (dropRef.current) return; // stale sentence of an aborted reply
            setTyping(false);
            setMessages((ms) => {
              let list = ms;
              if (!openAssistantId.current) {
                const b = { id: nextId(), role: "assistant", text: "", ts: Date.now() };
                openAssistantId.current = b.id;
                list = [...ms, b];
              }
              return list.map((x) =>
                x.id === openAssistantId.current
                  ? { ...x, text: x.text + m.text }
                  : x
              );
            });
            assistantTextRef.current += m.text;
          } else if (m.type === "error") {
            if (assistantTextRef.current) pushHistory("assistant", assistantTextRef.current);
            assistantTextRef.current = "";
            openAssistantId.current = null;
            activeRef.current = false;
            // a server-side failure (e.g. OmniVoice returned no audio) should
            // not leave the UI stuck in a "speaking" state.
            speakingRef.current = false;
            setSpeaking(false);
            if (speakWatchdog) { clearTimeout(speakWatchdog); speakWatchdog = 0; }
            if (currentSourceRef.current) {
              try { currentSourceRef.current.stop(); } catch { /* noop */ }
              currentSourceRef.current = null;
            }
            pendingRef.current = [];
            showError("बोलने में त्रुटि: " + m.message);
          } else if (m.type === "done") {
            if (dropRef.current) {
              // closing done of the reply we interrupted — discard silently
              openAssistantId.current = null;
              assistantTextRef.current = "";
              activeRef.current = false;
              return;
            }
            console.info(
              `[voice] reply done: ${m.frames} frame(s) | server total ${m.elapsed}s | first audio ${m.first_audio ?? "?"}s | TTS RTF ${m.rtf ?? "?"}`,
            );
            // attach the server-measured response time to the finished bubble
            const doneId = openAssistantId.current;
            if (doneId && m.elapsed != null) {
              setMessages((ms) =>
                ms.map((x) => (x.id === doneId ? { ...x, elapsed: m.elapsed } : x))
              );
            }
            if (assistantTextRef.current) pushHistory("assistant", assistantTextRef.current);

            assistantTextRef.current = "";
            openAssistantId.current = null;
            activeRef.current = false;
            // Reply stream is complete — if the jitter buffer is still holding
            // frames (never reached jitterFrames), start playback NOW so the
            // tail isn't stranded silent until the next turn.
            if (!speakingRef.current && pendingRef.current.length > 0) {
              speakingRef.current = true;
              api.setSpeaking(true);
              drain();
            }
            // audio may still be playing its last frame — aura keeps pulsing
            // until drain() ends; only force-stop if nothing is left.
            if (!currentSourceRef.current && pendingRef.current.length === 0) {
              api.setSpeaking(false);
            }
          }
        } else {
          if (dropRef.current) return; // stale audio frame of an aborted reply
          framesRef.current += 1;
          if (framesRef.current === 1 && turnStartRef.current) {
            console.info(
              `[voice] first audio received ${(performance.now() - turnStartRef.current).toFixed(0)}ms after submit (browser-measured, incl. network)`,
            );
          }
          pendingRef.current.push(new Blob([ev.data], { type: "audio/wav" }));
          // Jitter buffer: don't start playing on the first frame alone. The
          // first window is a tiny ~1s fragment; starting immediately means
          // playback runs dry before window #2 is generated (RTF > 1 on
          // MPS/CPU) — heard as mid-word cutoffs. Wait until jitterFrames
          // are queued, or the reply stream ends (done/error), to start.
          const wantStart =
            CFG.jitterFrames <= 0 ||
            pendingRef.current.length >= CFG.jitterFrames ||
            drainRestartRef.current ||
            !activeRef.current;
          if (!speakingRef.current && wantStart) {
            drainRestartRef.current = false;
            speakingRef.current = true;
            api.setSpeaking(true);
            drain();
          }
          // If the "start" event already opened the speaking state, just kick
          // playback when the jitter condition is met.
          else if (speakingRef.current && !currentSourceRef.current && !drainStartedRef.current && wantStart) {
            drainStartedRef.current = true;
            drain().finally(() => { drainStartedRef.current = false; });
          }
          // keep the playback watchdog honest: a new frame just landed
          if (speakWatchdog) {
            clearTimeout(speakWatchdog);
            speakWatchdog = 0;
          }
          // arm the watchdog the first time we start playing — if the queue
          // empties and nothing replays within ~2.5 s, something is stuck.
          if (!speakWatchdog && currentSourceRef.current) {
            speakWatchdog = setTimeout(() => {
              if (speakingRef.current && !currentSourceRef.current && pendingRef.current.length === 0) {
                speakingRef.current = false;
                setSpeaking(false);
                console.warn("[voice] speaking stuck with no active source — cleared");
              }
              speakWatchdog = 0;
            }, 2500);
          }
        }
      };
    };
    connect();
    return () => {
      closed = true;
      if (speakWatchdog) { clearTimeout(speakWatchdog); speakWatchdog = 0; }
      const ws = wsRef.current;
      if (ws) ws.close();
    };
  }, [drain]);

  /* ---------- Mic: realtime conversation -----------------------------
     Echo-free voice input (OpenAI/Gemini-style):
       1. Mic ON  -> getUserMedia with echoCancellation. Chrome subtracts our
          own speaker output from this stream, so the assistant can never
          "hear" itself. The Web Speech API is GONE — its separate capture
          path had no AEC reference, which caused the self-conversation loop.
       2. The AEC-cleaned mic is tapped (AudioWorklet, 16 kHz) and streamed to
          the server's /ws/asr (faster-whisper). Partial transcripts come back
          as live captions; the authoritative "final" is sent as the user turn.
       3. VAD (energy) drives turn-taking: sustained voice (~250 ms) opens an
          utterance, ~0.5 s of silence closes it, barge-in still cuts the
          assistant on real sustained speech. While our reply is audibly
          playing, mic frames are dropped, so residual echo can never even be
          sent to the transcriber.
  -------------------------------------------------------------------- */
  useEffect(() => {
    let rafVad = 0;
    let pcmTimer = 0;
    let closingAsr = false;
    let asrClosedCount = 0;
    let asrAttempts = 0;
    let asrHb = 0;
    let asrLastPong = 0;
    let asrWarned = false;
    let pcmBuf = []; // Float32Array pieces waiting to be flushed to /ws/asr
    let pcmLen = 0;
    // Silence-tail hold: after "early_end" we STOP streaming mic PCM so the
    // server's early-decode buffer stays frozen (exact reuse at "end"). Frames
    // arriving during the tail are parked here and restored if speech resumes.
    let tailHold = false;
    let tailBuf = [];
    // Server-side Silero VAD mode: the SERVER decides when an utterance opens
    // (neural speech/noise classification — fans/doors/keys can't open turns).
    // The browser becomes a dumb continuous streamer + keeps energy barge-in.
    let serverMode = CFG.asrVadMode === "server";
    let prevAssistant = false; // speaking-state edge -> /ws/asr "assistant" gate
    let lastPartial = ""; // most recent live caption (fallback if final is empty)
    let specSentFor = ""; // text already submitted speculatively (guards double-send)
    // Pre-roll ring buffer: ALWAYS keep the last ~350 ms of non-echo mic PCM.
    // When the VAD opens an utterance (~250 ms sustain delay) we replay this
    // first, so the first 2-3 words are never eaten by the open latency.
    const PRE_ROLL_LEN = Math.floor(16000 * 0.35); // 350 ms @ 16 kHz
    let preRoll = new Float32Array(PRE_ROLL_LEN);
    let preRollPos = 0;
    let preRollLen = 0;
    const preRollPush = (arr) => {
      for (let i = 0; i < arr.length; i++) {
        preRoll[preRollPos] = arr[i];
        preRollPos = (preRollPos + 1) % PRE_ROLL_LEN;
        if (preRollLen < PRE_ROLL_LEN) preRollLen += 1;
      }
    };
    const preRollRead = () => {
      if (!preRollLen) return null;
      const out = new Float32Array(preRollLen);
      const start = preRollLen === PRE_ROLL_LEN ? preRollPos : 0;
      for (let i = 0; i < preRollLen; i++) out[i] = preRoll[(start + i) % PRE_ROLL_LEN];
      return out;
    };
    const preRollClear = () => { preRollLen = 0; };

    const asrSendJson = (obj) => {
      const ws = asrWsRef.current;
      if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj));
    };
    const asrSendPcm = () => {
      // Server-VAD mode makes the browser a dumb continuous streamer: flush
      // whenever the socket is open, EVEN before Silero opens an utterance.
      // streamingRef only turns true on the server's vad_start, which cannot
      // arrive until the server has HEARD audio — gating the flush on it here
      // deadlocks the whole path (no PCM sent -> Silero never opens -> no
      // vad_start -> no transcription at all).
      const canStream = streamingRef.current || serverMode;
      // During a decode (asrBusy) the server worker is busy but still QUEUES
      // incoming PCM — in server mode keep flushing so the first words of the
      // user's next sentence are never dropped (the server's Silero pre-roll
      // covers the seam). Only the legacy client-VAD path pauses while a
      // final is on its way.
      const notBusy = serverMode || !asrBusyRef.current;
      if (pcmLen && canStream && asrOpenRef.current && notBusy) {
        const joined = new Float32Array(pcmLen);
        let o = 0;
        for (const p of pcmBuf) {
          joined.set(p, o);
          o += p.length;
        }
        try {
          asrWsRef.current.send(joined.buffer);
        } catch { /* socket closed mid-flush — reconnect handles it */ }
      }
      pcmBuf = [];
      pcmLen = 0;
    };

    const connectAsr = () => {
      const ws = new WebSocket(ASR_URL);
      ws.binaryType = "arraybuffer";
      asrWsRef.current = ws;
      ws.onopen = () => {
        asrOpenRef.current = true;
        asrClosedCount = 0;
        asrAttempts = 0;
        asrLastPong = Date.now();
        // opt into server-side Silero VAD (the server then owns open/close)
        if (serverMode) asrSendJson({ type: "mode", vad: "server" });
        // heartbeat: detect a dead ASR socket in ~15s
        asrHb = setInterval(() => {
          try {
            ws.send(JSON.stringify({ type: "ping" }));
          } catch { /* closed — onclose handles it */ }
          if (Date.now() - asrLastPong > 15000) {
            try { ws.close(); } catch { /* noop */ }
          }
        }, 5000);
      };
      ws.onclose = () => {
        asrOpenRef.current = false;
        asrBusyRef.current = false;
        streamingRef.current = false;
        pcmBuf = [];
        pcmLen = 0;
        if (asrHb) { clearInterval(asrHb); asrHb = 0; }
        if (listeningRef.current && !closingAsr) {
          asrClosedCount += 1;
          asrAttempts += 1;
          // reconnect immediately (bounded) so a momentary server restart or
          // network blip doesn't leave the mic open and silent.
          const delay = Math.min(CFG.wsReconnectMs * 2 ** Math.min(asrAttempts - 1, 3), 15000);
          setTimeout(connectAsr, delay);
          // transient indicator while we're down — clearer than a silent mic.
          if (asrAttempts <= 3) {
            setInterim("");
            setAsrReconnecting(true);
          }
          // The /ws/asr endpoint only exists on a server running the current
          // code — tell the user to restart it instead of failing silently,
          // but only after a few reconnect attempts so a transient drop isn't
          // surfaced as a server error.
          if (asrClosedCount >= 3 && !asrWarned) {
            asrWarned = true;
            showError("वॉयस इंजन (Whisper) से कनेक्ट नहीं हो पा रहा — कृपया सर्वर रीस्टार्ट करें (python voice_api.py)।");
          }
        }
      };
      ws.onmessage = (ev) => {
        let m;
        try {
          m = JSON.parse(ev.data);
        } catch {
          return;
        }
        if (m.type === "pong") { // heartbeat reply
          asrLastPong = Date.now();
          if (asrReconnecting) {
            setAsrReconnecting(false);
            setInterim("");
          }
          return;
        }
        if (m.type === "vad_start") {
          // Server Silero opened an utterance — mirror client state so the
          // legacy hooks (interim, watchdogs) stay coherent.
          streamingRef.current = true;
          vadRef.current.streamingRefCur = true;
          vadRef.current.textHeard = false;
          specSentFor = "";
          setInterim("");
          return;
        }
        if (m.type === "vad_end") {
          // Server closed it: speculative/final transcripts follow next
          streamingRef.current = false;
          vadRef.current.streamingRefCur = false;
          asrBusyRef.current = true;
          return;
        }
        if (m.type === "rejected") {
          // Server dropped the utterance ("speaker" = voiceprint mismatch from
          // the gate, "blip" = too short). Show why, clear any stale caption
          // fallback so it can never fire a phantom turn, release the mic.
          clearTimeout(asrRejTimerRef.current);
          setAsrRejected(m.message || "blip");
          asrRejTimerRef.current = setTimeout(() => setAsrRejected(""), 2500);
          lastPartial = "";
          asrBusyRef.current = false;
          streamingRef.current = false;
          vadRef.current.streamingRefCur = false;
          setInterim("");
          return;
        }
        if (m.type === "partial") {
          // live caption from Whisper; also confirms real speech for barge-in
          const t = (m.text || "").trim();
          if (t) {
            lastPartial = t;
            vadRef.current.textHeard = true;
            setInterim(t);
          }
        } else if (m.type === "speculative") {
          // Fast greedy decode of the finished utterance: start the LLM turn
          // NOW. The authoritative "final" follows and reconciles.
          if (!asrBusyRef.current) return; // stale spec after a watchdog release — ignore
          const t = (m.text || "").trim();
          if (pushActiveRef.current || pushReleasePendingRef.current) {
            // Holding X (or just released, turn assembling): ABSORB the
            // question — it becomes the push turn's text on release instead of
            // firing a screenless reply now.
            if (t) pushTextRef.current = (pushTextRef.current ? pushTextRef.current + " " : "") + t;
            asrBusyRef.current = false;
            specSentFor = "";
            lastPartial = "";
            setInterim("");
            return;
          }
          if (CFG.specChat && t && t.replace(/\s/g, "").length >= CFG.sendMinChars) {
            vadRef.current.textHeard = true;
            setInterim("");
            specSentFor = t;
            lastPartial = ""; // final reconciliation only if no spec turn went out
            asrBusyRef.current = false; // allow the next utterance to open while the final lands
            apiRef.current.submitChat(t);
          }
        } else if (m.type === "final") {
          // short clips can come back empty — fall back to the last live caption
          const t = (m.text || "").trim() || lastPartial;
          lastPartial = "";
          asrBusyRef.current = false;
          streamingRef.current = false;
          if (pushActiveRef.current) {
            // Holding X: absorb the beam final too (replaces/merges with the
            // speculative guess) — no turn until the key comes up.
            if (t) pushTextRef.current = (pushTextRef.current ? pushTextRef.current + " " : "") + t;
            specSentFor = "";
            setInterim("");
            return;
          }
          if (pushReleasePendingRef.current) {
            // X was JUST released but the turn is still assembling (waiting
            // for the first frame). Attach this transcript to the push turn's
            // text instead of firing a separate screenless reply now.
            if (t) pushTextRef.current = (pushTextRef.current ? pushTextRef.current + " " : "") + t;
            specSentFor = "";
            setInterim("");
            return;
          }
          if (specSentFor) {
            // A speculative turn already went out: reconcile only on real divergence.
            const norm = (s) => s.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").trim();
            const same = norm(t) === norm(specSentFor);
            const tokensA = new Set(norm(specSentFor).split(/\s+/).filter(Boolean));
            const tokensB = new Set(norm(t).split(/\s+/).filter(Boolean));
            const common = [...tokensA].filter((w) => tokensB.has(w)).length;
            const beamLen = t.replace(/\s/g, "").length;
            const specLen = specSentFor.replace(/\s/g, "").length;
            // Only reconcile UPWARD: the beam may add/correct text the greedy
            // pass missed, but a SHORTER beam guess on a short clip is just
            // noise-vs-noise — resubmitting there barge-ins our own reply.
            const diverged =
              !same &&
              tokensB.size > 0 &&
              beamLen >= specLen &&
              (common / Math.max(1, tokensA.size) < 0.5 || beamLen > specLen * 1.6);
            if (diverged) {
              specSentFor = "";
              setInterim(t);
              apiRef.current.submitChat(t); // beam heard something meaningfully different
            } else {
              specSentFor = ""; // close enough — the speculative reply stands
            }
          } else if (t && t.replace(/\s/g, "").length >= CFG.sendMinChars) {
            vadRef.current.textHeard = true;
            setInterim(t);
            apiRef.current.submitChat(t); // the turn goes out with the real transcript
          } else {
            setInterim("");
          }
        } else if (m.type === "error") {
          asrBusyRef.current = false;
          streamingRef.current = false;
          vadRef.current.streamingRefCur = false; // keep the VAD hold-mirror in sync
          specSentFor = ""; // never let a dead utterance's guard leak into the next one
          if (asrReconnecting) setAsrReconnecting(false);
          setInterim("");
          showError("बोलने की पहचान में त्रुटि: " + (m.message || ""));
        }
      };
    };

    const ticksFor = (ms) => Math.max(1, Math.round(ms / CFG.vadTickMs));

    /* Tap handler: called with 16 kHz Float32 frames whenever the user's mic
       burst is open. While OUR reply is audibly playing (or its tail is still
       ringing out) the frames are dropped — belt-and-suspenders over Chrome's
       AEC so the assistant's own voice can never reach the transcriber. */
    const onFrame = (arr) => {
      const v = vadRef.current;
      const now = performance.now();
      // Always push into rolling pre-roll (350ms of audio) so when the user
      // interrupts (barges in), their initial words are NEVER lost!
      preRollPush(arr);

      if (serverMode) {
        // Server VAD owns turn-taking: stream continuously so Silero can detect speech
        if (!asrOpenRef.current) return;
        pcmBuf.push(arr);
        pcmLen += arr.length;
        return;
      }

      // In client mode, drop frames only while audio is playing and no utterance is open
      const echo = speakingRef.current || (!streamingRef.current && now - v.lastSpeakAt <= CFG.speakTailMs);
      if (echo && !streamingRef.current) {
        return;
      }

      if (tailHold) {
        tailBuf.push(arr); // park tail frames; restored if speech resumes
        return;
      }
      if (!streamingRef.current) return;
      pcmBuf.push(arr);
      pcmLen += arr.length;
    };

    // Leaky-integrator voice metric with hysteresis.
    //
    // The old gate was brittle: one tick below threshold reset the open counter,
    // so a brief mouth noise or a mic-gain pump could kill an utterance that was
    // already mid-word. The new gate smooths short-term RMS into `v.voiceEnergy`
    // and opens only when that smoothed value has been above a *lower* open
    // threshold for vadSustainMs, then HOLDS the mic open down to an even lower
    // hold threshold (hysteresis), so brief dips never steal the next phoneme.
    // Barge-in still uses the original strong-energy path so the assistant can be
    // cut reliably on real sustained speech.
    const vadTick = () => {
      if (!listeningRef.current) return;
      const mic = engine.readMic();
      const v = vadRef.current;
      const now = performance.now();

      // thresholds recomputed per tick so a late /api/config fetch is honored
      const sustainTicks = ticksFor(CFG.vadSustainMs); // real voice energy length
      const textTicks = ticksFor(CFG.vadTextMs); // words + energy confirm
      const failsafeTicks = ticksFor(CFG.vadFailsafeMs); // sustained-energy failsafe

      // smoothed short-term energy (leaky integrator, ~1 s time constant at
      // 50 ms ticks). Keeps momentary dips from resetting the open decision.
      const alpha = 0.08;
      v.voiceEnergy = v.voiceEnergy == null
        ? mic.rms
        : v.voiceEnergy * (1 - alpha) + mic.rms * alpha;

      // hysteresis thresholds are recomputed from the live noise floor each tick
      // so a late /api/config fetch is honored without restarting the mic.
      const openThresh = Math.max(v.noise * CFG.vadGateMult * 0.9, CFG.vadThresholdMin * 0.9);
      const holdThresh = Math.max(v.noise * CFG.vadGateMult * 0.45, CFG.vadThresholdMin * 0.45);
      const strongThresh = v.threshold * 1.6;

      // hot = clearly above the open gate this tick
      const hot = mic.rms > openThresh;
      const inHold = v.streamingRefCur || v.voiceSilentSince > 0
        ? mic.rms > holdThresh : false;
      const sustainedVoice = v.hotTicks >= sustainTicks;

      // Remember when OUR reply is audibly playing — the frame-drop gate in
      // onFrame keys off this deterministic app state.
      if (speakingRef.current) v.lastSpeakAt = now;

      // Assistant-audio gate for server VAD: tell /ws/asr when OUR TTS plays
      // so Silero never opens an utterance on AEC residue of our own voice.
      if (serverMode && speakingRef.current !== prevAssistant) {
        prevAssistant = speakingRef.current;
        asrSendJson({ type: "assistant", active: prevAssistant });
      }

      // Aura: show the user is talking (from the raw mic analyser).
      if (mic.rms > v.noise * 0.8 && mic.rms > CFG.vadNoiseFloor) {
        if (!apiRef.current.userTalkingRef.current) apiRef.current.setUserTalking(true);
      } else if (mic.rms < v.noise * 0.6) {
        if (apiRef.current.userTalkingRef.current) apiRef.current.setUserTalking(false);
      }

      // strong energy (well above the gate) — barge-in path only
      const strong = mic.rms > strongThresh;
      if (hot || (v.streamingRefCur && inHold)) {
        v.hotTicks += 1;
        v.lastUtt = now; // energy-backed speech time (drives end-of-speech)
      } else {
        // do NOT hard-reset while we're holding an open utterance — allow brief
        // dips below the open line but above the hold line
        if (!v.streamingRefCur || mic.rms <= holdThresh) v.hotTicks = Math.max(0, v.hotTicks - 1);
      }
      if (strong) v.strongTicks += 1;
      else v.strongTicks = 0;
      if (sustainedVoice) v.anyVoice = true;

      // voice heard = open energy OR recognizer already delivered words OR we
      // previously latched that real voice existed this burst
      const voiceHeard = sustainedVoice || v.textHeard || v.anyVoice;

      // Ambient noise-floor tracking ONLY while nobody has spoken recently.
      if (!voiceHeard && now - v.lastUtt > CFG.vadRecActiveMs + 200) {
        v.noise = v.noise * 0.998 + mic.rms * 0.002;
        v.threshold = Math.max(v.noise * CFG.vadGateMult, CFG.vadThresholdMin);
      }

      const assistantBusy = speakingRef.current || activeRef.current;
      v.streamingRefCur = streamingRef.current; // used by the hold logic above

      // ---- OPEN an utterance on real sustained voice ------------------
      // (client energy VAD path only — in server mode Silero decides)
      if (!serverMode && sustainedVoice && !streamingRef.current && !asrBusyRef.current && asrOpenRef.current) {
        streamingRef.current = true;
        v.voiceSilentSince = 0;
        v.hotTicks = 0; // reset open-counter once the utterance is open (hold mode)
        specSentFor = ""; // new utterance: drop any stale speculative-turn guard
        asrSendJson({ type: "start" });
        v.recordingSince = now;
        // Replay the pre-roll right after "start" so the ~250 ms the VAD
        // needed to open the utterance is NOT lost from the transcript.
        const pr = preRollRead();
        if (pr) {
          pcmBuf.unshift(pr);
          pcmLen += pr.length;
        }
        preRollClear();
      }

      // ---- BARGE-IN (real sustained speech cuts the assistant) --------
      // Grace window: for ~1.2 s after WE submit a turn, ignore barge-in —
      // the tail of the user's own just-ended utterance (or its echo) can
      // otherwise kill the turn server-side and the reply comes back EMPTY
      // (the mysterious 0-frame replies). Real interrupts still work: 1.2 s
      // is shorter than the first TTS window, and sustained speech AFTER the
      // grace cuts normally.
      const bargeGrace = turnStartRef.current && (performance.now() - turnStartRef.current) < 1200;
      if (assistantBusy && !bargeGrace && !v.bargeLatched) {
        const enough =
          (v.textHeard && v.hotTicks >= textTicks) || v.strongTicks >= failsafeTicks;
        if (enough) {
          v.bargeLatched = true;
          v.lastSpeakAt = 0; // Clear echo tail so speech immediately after barge-in is NOT dropped!
          apiRef.current.hardStop();
          if (serverMode) {
            asrSendJson({ type: "assistant", active: false });
          } else if (!streamingRef.current && asrOpenRef.current) {
            streamingRef.current = true;
            v.voiceSilentSince = 0;
            v.hotTicks = 0;
            specSentFor = "";
            asrSendJson({ type: "start" });
            const pr = preRollRead();
            if (pr) {
              pcmBuf.unshift(pr);
              pcmLen += pr.length;
            }
            preRollClear();
          }
        }
      }
      // latch releases once the user stops producing sound (next burst can cut)
      if (v.bargeLatched && !hot && v.strongTicks === 0) v.bargeLatched = false;

      // ---- END-OF-SPEECH: silence after a real burst closes it ---------
      const recActive = now - v.lastUtt < CFG.vadRecActiveMs;
      // If no speculative transcript arrives shortly after "end" (slow backend
      // or disabled), release the mic anyway so the next burst works.
      if (!streamingRef.current && !asrBusyRef.current && !specSentFor &&
          now - v.uttEndedAt > (CFG.specChat ? 120 : 0)) {
        asrBusyRef.current = false;
        streamingRef.current = false;
      }

      if (!serverMode && streamingRef.current && voiceHeard && !assistantBusy) {
        if (hot || inHold || recActive) {
          if (v.voiceSilentSince) {
            v.voiceSilentSince = 0; // resumed within the silence tail
            asrSendJson({ type: "resume" }); // void any early pre-decode
            if (tailHold) {
              tailHold = false; // resume streaming; restore parked frames first
              pcmBuf.unshift(...tailBuf);
              pcmLen += tailBuf.reduce((n, p) => n + p.length, 0);
              tailBuf = [];
            }
          }
        } else {
          if (!v.voiceSilentSince) {
            v.voiceSilentSince = now;
            // silence just started -> server pre-decodes NOW while this tail
            // counts down; the "end" below then reuses that decode instantly.
            // Stop streaming PCM so the server's decode buffer stays frozen.
            tailHold = true;
            tailBuf = [];
            asrSendJson({ type: "early_end" });
          }
          else if (now - v.voiceSilentSince > CFG.autoSendMs) {
            v.voiceSilentSince = 0;
            v.hotTicks = 0;
            v.textHeard = false;
            v.anyVoice = false;
            v.streamingRefCur = false;
            streamingRef.current = false;
            v.uttEndedAt = now;
            asrBusyRef.current = true;
            tailHold = false; // discard parked tail frames (post-speech silence)
            tailBuf = [];
            asrSendJson({ type: "end" });
            // Watchdog: if neither a speculative nor the final transcript
            // arrives (lost frame/drop), don't leave the mic dead — release
            // after 12 s so the next burst works.
            setTimeout(() => {
              if (asrBusyRef.current) {
                asrBusyRef.current = false;
                streamingRef.current = false;
                v.streamingRefCur = false;
                setInterim("");
              }
            }, 12000);
          }
        }
      }
    };

    const enableMic = async () => {
      const v = vadRef.current;
      serverMode = CFG.asrVadMode === "server"; // re-read config each session
      prevAssistant = false;
      tailHold = false;
      tailBuf = [];
      v.noise = CFG.vadNoiseFloor;
      v.threshold = Math.max(CFG.vadNoiseFloor * CFG.vadGateMult, CFG.vadThresholdMin);
      v.hotTicks = 0;
      v.strongTicks = 0;
      v.textHeard = false;
      v.anyVoice = false;
      v.bargeLatched = false;
      v.voiceSilentSince = 0;
      v.uttEndedAt = 0;
      v.lastUtt = 0;
      v.lastSpeakAt = 0;
      v.voiceEnergy = null;
      v.streamingRefCur = false;
      v.recordingSince = 0;
      streamingRef.current = false;
      asrBusyRef.current = false;
      asrWarned = false;
      lastPartial = "";
      preRollClear(); // fresh pre-roll per listening session — no stale audio
      setInterim("");
      listeningRef.current = true;
      setListening(true);
      rafVad = setInterval(vadTick, CFG.vadTickMs);
      pcmTimer = setInterval(asrSendPcm, 25); // flush mic PCM every 25 ms (was 50 — halves mic-side latency)
      const ok = await engine.startMic(); // AEC-enabled mic (inside the click)
      if (!ok) {
        disableMic();
        showError("माइक अनुमति नहीं मिली। ब्राउज़र में माइक की अनुमति दें और दोबारा दबाएँ।");
        return;
      }
      connectAsr();
      const tapped = await engine.startTap(onFrame);
      if (!tapped.ok) {
        disableMic();
        const msg =
          tapped.error
            ? `ऑडियो स्ट्रीमिंग उपलब्ध नहीं है — ${tapped.error}. Chrome/Edge आज़माएँ।`
            : "इस ब्राउज़र में ऑडियो स्ट्रीमिंग उपलब्ध नहीं है — Chrome/Edge आज़माएँ।";
        showError(msg);
      }
    };

    const disableMic = () => {
      clearInterval(rafVad);
      clearInterval(pcmTimer);
      listeningRef.current = false;
      setListening(false);
      apiRef.current.setUserTalking(false);
      closingAsr = true;
      const ws = asrWsRef.current;
      if (ws) {
        try {
          ws.onclose = null;
          ws.close();
        } catch { /* noop */ }
      }
      asrWsRef.current = null;
      asrOpenRef.current = false;
      asrBusyRef.current = false;
      streamingRef.current = false;
      pcmBuf = [];
      pcmLen = 0;
      tailHold = false;
      tailBuf = [];
      engine.stopTap();
      engine.stopMic();
      setInterim("");
    };

    micToggleRef.current = () => {
      if (listeningRef.current) disableMic();
      else enableMic();
    };
    return () => {
      disableMic();
    };
  }, []);

  /* ---------- Hold X = push-to-see (screen context on demand) ----------
     keydown X starts the capture+warm hold, keyup releases and sends the turn
     with the freshest frame plus anything spoken into the mic while holding.
     Ignored while typing in the input/textarea; window blur releases safely
     (a lost keyup must never leave a stuck hold). */
  useEffect(() => {
    if (!CFG.pushMode) return;
    const isX = (e) => e.key === "x" || e.key === "X";
    const isTyping = () => {
      const el = document.activeElement;
      return el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable);
    };
    const down = (e) => {
      if (e.repeat || !isX(e) || isTyping()) return;
      e.preventDefault();
      apiRef.current.startPush && apiRef.current.startPush();
    };
    const up = (e) => {
      if (!isX(e) || !apiRef.current.pushActiveRef?.current) return;
      e.preventDefault();
      apiRef.current.releasePush && apiRef.current.releasePush();
    };
    const blur = () => {
      // lost keyup (alt-tab, minimize) — release so the hold can't stick
      if (apiRef.current.pushActiveRef?.current) apiRef.current.releasePush();
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    window.addEventListener("blur", blur);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
      window.removeEventListener("blur", blur);
    };
  }, []);

  const sendText = useCallback((event) => {
    event.preventDefault();
    const text = input.trim();
    if (!text) return;
    submitChat(text);
    setInput("");
  }, [input, submitChat]);

  /* ---------- single-screen shell handlers (UI only — voice logic untouched) ---------- */
  const clearChat = useCallback(() => {
    hardStop();
    historyRef.current = [];
    openAssistantId.current = null;
    assistantTextRef.current = "";
    setDiagram(null);
    setMessages([{ id: nextId(), role: "assistant", text: GREETING }]);
  }, [hardStop]);

  const handleToggleMic = useCallback(() => {
    apiRef.current.toggleMic && apiRef.current.toggleMic();
  }, []);

  const shareDown = useCallback(() => {
    if (!CFG.pushMode) {
      if (sharingRef.current) stopScreenCapture();
      else startScreenShare();
      return;
    }
    apiRef.current.startPush && apiRef.current.startPush();
  }, [startScreenShare, stopScreenCapture]);

  const shareUp = useCallback(() => {
    if (CFG.pushMode && apiRef.current.pushActiveRef?.current) apiRef.current.releasePush();
  }, []);

  /* ---------- UI: single screen + sidebar + Gemini bottom bar ---------- */
  return (
    <div className="relative flex h-screen overflow-hidden bg-white text-slate-900">
      <AuroraBg listening={listening} speaking={speaking} userTalking={userTalking} />
      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        connected={connected}
        listening={listening}
        asrReady={asrReady}
        messageCount={messages.length}
        hasDiagram={!!diagram}
        onClear={clearChat}
        onToggleMic={handleToggleMic}
        onShareScreen={shareDown}
        sharing={sharing}
        visionEnabled={CFG.visionEnabled}
      />
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-20 bg-slate-900/20 min-md:hidden"
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}

      <div className="relative z-10 flex min-w-0 flex-1 flex-col">
        <header className="flex items-center gap-3 border-b border-slate-200/70 bg-white/70 px-4 py-3 backdrop-blur sm:px-6">
          <button
            onClick={() => setSidebarOpen((v) => !v)}
            className="inline-flex h-9 w-9 items-center justify-center rounded-full text-slate-500 transition hover:bg-slate-100 hover:text-[#ff5a5f]"
            aria-label={sidebarOpen ? "Hide sidebar" : "Show sidebar"}
          >
            <FaBars className="h-4 w-4" />
          </button>
          <div className="min-w-0">
            <h1 className="truncate text-sm font-bold tracking-wide text-slate-900">
              Saathi <span className="font-normal text-slate-400">· Hindi voice tutor</span>
            </h1>
          </div>
          <div className="ml-auto flex items-center gap-2 text-xs font-medium text-slate-500">
            {asrRejected && (
              <span className="hidden rounded-full bg-amber-50 px-2.5 py-1 text-[11px] font-semibold text-amber-700 sm:block">
                {asrRejected}
              </span>
            )}
            <span
              className={`rounded-full px-2.5 py-1 text-[11px] font-semibold ${
                typing
                  ? "bg-[#ff5a5f]/10 text-[#ff5a5f]"
                  : speaking
                    ? "bg-[#ff5a5f]/10 text-[#ff5a5f]"
                    : listening
                      ? userTalking
                        ? "bg-emerald-50 text-emerald-700"
                        : "bg-slate-100 text-slate-600"
                      : "bg-slate-100 text-slate-500"
              }`}
            >
              {typing
                ? "Thinking…"
                : speaking
                  ? "Speaking…"
                  : listening
                    ? userTalking
                      ? "Listening…"
                      : asrReady
                        ? "Mic live"
                        : "Warming up…"
                    : connected
                      ? "Ready"
                      : "Connecting…"}
            </span>
          </div>
        </header>

        <main
          className={`flex min-h-0 flex-1 gap-4 overflow-hidden px-4 pt-4 sm:px-6 ${
            diagram ? "lg:flex-row" : "flex-col"
          }`}
        >
          <section
            ref={chatEl}
            className={`flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto pb-2 ${
              diagram ? "lg:max-w-[46%]" : "mx-auto w-full max-w-3xl"
            }`}
            aria-live="polite"
            aria-label="Chat transcript"
          >
            {messages.map((message) => {
              const isUser = message.role === "user";
              const isError = message.role === "error";
              return (
                <article
                  key={message.id}
                  className={`max-w-[88%] rounded-2xl px-4 py-3 text-sm leading-6 shadow-sm ${
                    isError
                      ? "self-start border border-rose-200 bg-rose-50 text-rose-700"
                      : isUser
                        ? "self-end bg-[#ff5a5f] text-white"
                        : "self-start border border-slate-200 bg-white text-slate-700"
                  }`}
                >
                  <div
                    className={`mb-1 text-[0.62rem] font-bold tracking-[0.12em] uppercase ${
                      isUser ? "text-white/80" : isError ? "text-rose-500" : "text-[#ff5a5f]"
                    }`}
                  >
                    {isError ? "Notice" : isUser ? "You" : "Tutor"}
                  </div>
                  <p className="whitespace-pre-wrap">{message.text}</p>
                  {message.elapsed != null && (
                    <span className="mt-2 block text-[0.65rem] opacity-60">
                      {fmtElapsed(message.elapsed)}
                    </span>
                  )}
                </article>
              );
            })}
            {interim && (
              <div className="self-end max-w-[88%] rounded-2xl border border-dashed border-[#ff5a5f]/40 bg-[#ff5a5f]/5 px-4 py-3 text-sm leading-6 text-slate-700 shadow-sm">
                <div className="mb-1 text-[0.62rem] font-bold tracking-[0.12em] text-[#ff5a5f] uppercase">
                  You’re saying…
                </div>
                <p className="whitespace-pre-wrap">{interim}</p>
              </div>
            )}
            {typing && (
              <div className="self-start rounded-2xl border border-slate-200 bg-white px-4 py-3 text-xs text-slate-400">
                Tutor is thinking…
              </div>
            )}
          </section>

          {diagram ? (
            <aside className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-[0_24px_70px_rgba(255,90,95,0.12)] [animation:diagram-reveal_420ms_cubic-bezier(0.16,1,0.3,1)_both] motion-reduce:animate-none">
              <div className="flex items-center justify-between border-b border-slate-200/70 px-5 py-3">
                <div>
                  <span className="block text-[0.64rem] font-bold tracking-[0.14em] text-[#ff5a5f] uppercase">
                    Visual explanation · {diagram?.elements?.length || 0} shapes
                  </span>
                  <h2 className="mt-0.5 text-[1rem] font-semibold tracking-[-0.02em] text-slate-900">
                    Let’s map it out
                  </h2>
                </div>
                <div className="flex items-center gap-2">
                  <span className="h-2 w-2 rounded-full bg-[#ff5a5f]" aria-hidden="true" />
                  <button
                    onClick={() => setDiagram(null)}
                    className="inline-flex h-8 w-8 items-center justify-center rounded-full text-slate-400 transition hover:bg-slate-100 hover:text-slate-700"
                    aria-label="Close whiteboard"
                  >
                    <FaXmark className="h-4 w-4" />
                  </button>
                </div>
              </div>
              <div className="min-h-0 flex-1">
                <Suspense
                  fallback={
                    <div className="grid h-full min-h-[320px] place-items-center text-[0.82rem] tracking-wide text-slate-500">
                      Preparing the whiteboard…
                    </div>
                  }
                >
                  <DiagramWhiteboard diagram={diagram} />
                </Suspense>
              </div>
            </aside>
          ) : null}
        </main>

        <BottomBar
          input={input}
          setInput={setInput}
          sendText={sendText}
          connected={connected}
          listening={listening}
          micBusy={micBusy}
          speaking={speaking}
          userTalking={userTalking}
          sharing={sharing}
          visionEnabled={CFG.visionEnabled}
          onToggleMic={handleToggleMic}
          onShareDown={shareDown}
          onShareUp={shareUp}
        />
      </div>
    </div>
  );
}
