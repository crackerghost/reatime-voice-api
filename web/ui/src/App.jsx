import { useCallback, useEffect, useRef, useState } from "react";
import { FaClock, FaMicrophone, FaPaperPlane, FaStop, FaTrash, FaVolumeHigh, FaWandMagicSparkles } from "react-icons/fa6";
import AuraGlobe from "./AuraGlobe.jsx";
import { engine } from "./audioEngine.js";

const GREETING = "नमस्ते! मैं आपका हिंदी असिस्टेंट हूँ। माइक दबाएँ या लिखकर पूछिए — और जब मैं बोलूँ तो बीच में कुछ भी बोलिए, मैं तुरंत रुक जाऊँगा।";
const WS_URL = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/tts`;
const ASR_URL = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/asr`;
const CONFIG_URL = `${location.protocol}//${location.host}/api/config`;

/* Live tunables, fetched once from the backend /api/config (which reads them
   from .env). Everything here mirrors an env var server-side, so tuning
   happens in .env — never by editing this file. Defaults below match the
   current behaviour and apply until the fetch resolves. */
const CFG = {
  chatStep: 12, // nfe_step the UI sends for chat replies
  maxHistory: 12,
  wsReconnectMs: 1500,
  recRestartMs: 400,
  vadTickMs: 50,
  autoSendMs: 450,
  sendMinChars: 2, // min non-space chars to treat as real words
  vadNoiseFloor: 0.005,
  vadThresholdMin: 0.014,
  vadGateMult: 3.4, // voice gate = noise * this (or threshold_min, whichever is higher)
  vadSustainMs: 250, // energy this long = real voice (barge / arm send)
  vadTextMs: 150, // …or recognizer words + this much energy confirm
  vadFailsafeMs: 900, // sustained energy failsafe (recognizer lagging)
  speakTailMs: 700, // ignore recognition this long after OUR speaker audio stops (echo)
  vadRecActiveMs: 400, // recognizer counts as active within this window
  bargeIdleMs: 900, // recognizer-idle safety-net send delay
};
const num = (v, d) => (v === undefined || v === null || Number.isNaN(Number(v)) ? d : Number(v));
const mergeCfg = (c) => {
  if (!c) return;
  CFG.chatStep = num(c.chat_step, CFG.chatStep);
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
  CFG.bargeIdleMs = num(c.barge_idle_ms, CFG.bargeIdleMs);
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

  // ---- mutable runtime state (safe across renders) ----
  const wsRef = useRef(null);
  const chatEl = useRef(null);
  const apiRef = useRef({});
  const historyRef = useRef([]);
  const pendingRef = useRef([]);
  const currentSourceRef = useRef(null);
  const speakingRef = useRef(false);
  const dropRef = useRef(false);
  const activeRef = useRef(false); // a reply is being streamed from the server
  const framesRef = useRef(0);
  const openAssistantId = useRef(null);
  const assistantTextRef = useRef("");
  const asrWsRef = useRef(null); // /ws/asr connection (streaming faster-whisper)
  const asrOpenRef = useRef(false);
  const asrBusyRef = useRef(false); // a "final" transcript is on its way
  const streamingRef = useRef(false); // mic PCM currently being sent to the ASR
  // VAD state — thresholds/initial values come from CFG (env-driven).
  const vadRef = useRef({
    noise: CFG.vadNoiseFloor, // adaptive ambient floor (updated when idle)
    threshold: Math.max(CFG.vadNoiseFloor * CFG.vadGateMult, CFG.vadThresholdMin),
    hotTicks: 0, // consecutive ticks above threshold (debounce noise blips)
    textHeard: false, // recognizer delivered real words during this burst
    anyVoice: false, // analyser saw real voice at least once this burst
    bargeLatched: false, // already cut the assistant during this burst
    voiceSilentSince: 0, // ms timestamp when the burst ended
    lastUtt: 0, // recognizer last heard anything (ms)
    lastSpeakAt: 0, // when OUR speaker output last had real audio (echo guard)
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
    return new Promise((resolve) => {
      const src = ctx.createBufferSource();
      src.buffer = buf;
      engine.connectSpeak(src);
      currentSourceRef.current = src;
      src.onended = () => {
        currentSourceRef.current = null;
        resolve();
      };
      src.start();
    });
  }, []);

  const drain = useCallback(async () => {
    while (pendingRef.current.length > 0) {
      if (dropRef.current) {
        pendingRef.current.length = 0;
        return;
      }
      const blob = pendingRef.current.shift();
      try {
        await playBlob(blob);
      } catch {
        /* ignore one bad frame */
      }
    }
    speakingRef.current = false;
    setSpeaking(false);
  }, [playBlob]);

  /* ---- hard stop: instant audio cut + server cancel (barge-in) ---- */
  const hardStop = useCallback(() => {
    dropRef.current = true; // drop stale frames/text until the next "start"
    if (currentSourceRef.current) {
      try {
        currentSourceRef.current.stop();
      } catch { /* noop */ }
      currentSourceRef.current = null;
    }
    pendingRef.current.length = 0;
    if (wsRef.current && wsRef.current.readyState === 1) {
      wsRef.current.send(JSON.stringify({ type: "stop" }));
    }
    speakingRef.current = false;
    setSpeaking(false);
  }, []);

  /* ---- submit a new user turn ---- */
  const submitChat = useCallback(
    (rawText) => {
      const text = (rawText || "").trim();
      if (!text) return;
      hardStop();
      setMessages((m) => [...m, { id: nextId(), role: "user", text, ts: Date.now() }]);
      pushHistory("user", text);

      const ws = wsRef.current;
      if (!ws || ws.readyState !== 1) {
        showError("सर्वर कनेक्शन नहीं है — रुकिए, फिर से पूछिए।");
        return;
      }
      // dropRef stays true (set by hardStop) until the server's "start" for
      // this reply — any stale frames of the aborted reply are ignored.
      framesRef.current = 0;
      activeRef.current = false;
      openAssistantId.current = null;
      assistantTextRef.current = "";
      setTyping(true);
      ws.send(JSON.stringify({ type: "chat", text, history: historyRef.current, nfe_step: CFG.chatStep }));
    },
    [hardStop]
  );

  apiRef.current = {
    submitChat,
    hardStop,
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

  /* ---------- WebSocket ---------- */
  useEffect(() => {
    let closed = false;
    const connect = () => {
      const ws = new WebSocket(WS_URL);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;
      ws.onopen = () => setConnected(true);
      ws.onerror = () => setConnected(false);
      ws.onclose = () => {
        setConnected(false);
        if (!closed) setTimeout(connect, CFG.wsReconnectMs);
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
          if (m.type === "start") {
            dropRef.current = false;
            activeRef.current = true;
            framesRef.current = 0;
            setTyping(false);
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
          } else if (m.type === "done") {
            if (dropRef.current) {
              // closing done of the reply we interrupted — discard silently
              openAssistantId.current = null;
              assistantTextRef.current = "";
              activeRef.current = false;
              return;
            }
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
            // audio may still be playing its last frame — aura keeps pulsing
            // until drain() ends; only force-stop if nothing is left.
            if (!currentSourceRef.current && pendingRef.current.length === 0) {
              api.setSpeaking(false);
            }
          } else if (m.type === "error") {
            if (assistantTextRef.current) pushHistory("assistant", assistantTextRef.current);
            assistantTextRef.current = "";
            openAssistantId.current = null;
            activeRef.current = false;
            api.setSpeaking(false);
            showError("बोलने में त्रुटि: " + m.message);
          }
        } else {
          if (dropRef.current) return; // stale audio frame of an aborted reply
          framesRef.current += 1;
          pendingRef.current.push(new Blob([ev.data], { type: "audio/wav" }));
          if (!speakingRef.current) {
            speakingRef.current = true;
            api.setSpeaking(true);
            drain();
          }
        }
      };
    };
    connect();
    return () => {
      closed = true;
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
    let pcmBuf = []; // Float32Array pieces waiting to be flushed to /ws/asr
    let pcmLen = 0;

    const asrSendJson = (obj) => {
      const ws = asrWsRef.current;
      if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj));
    };
    const asrSendPcm = () => {
      if (pcmLen && streamingRef.current && asrOpenRef.current && !asrBusyRef.current) {
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
      };
      ws.onclose = () => {
        asrOpenRef.current = false;
        asrBusyRef.current = false;
        streamingRef.current = false;
        pcmBuf = [];
        pcmLen = 0;
        if (listeningRef.current && !closingAsr) setTimeout(connectAsr, CFG.wsReconnectMs);
      };
      ws.onmessage = (ev) => {
        let m;
        try {
          m = JSON.parse(ev.data);
        } catch {
          return;
        }
        const v = vadRef.current;
        if (m.type === "partial") {
          // live caption from Whisper; also confirms real speech for barge-in
          const t = (m.text || "").trim();
          if (t) {
            v.textHeard = true;
            setInterim(t);
          }
        } else if (m.type === "final") {
          const t = (m.text || "").trim();
          asrBusyRef.current = false;
          streamingRef.current = false;
          if (t && t.replace(/\s/g, "").length >= CFG.sendMinChars) {
            v.textHeard = true;
            setInterim(t);
            apiRef.current.submitChat(t); // the turn goes out with the real transcript
          } else {
            setInterim("");
          }
        } else if (m.type === "error") {
          asrBusyRef.current = false;
          streamingRef.current = false;
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
      if (!streamingRef.current) return;
      const v = vadRef.current;
      if (speakingRef.current || performance.now() - v.lastSpeakAt <= CFG.speakTailMs) return;
      pcmBuf.push(arr);
      pcmLen += arr.length;
    };

    const vadTick = () => {
      if (!listeningRef.current) return;
      const mic = engine.readMic();
      const v = vadRef.current;
      const now = performance.now();
      // thresholds recomputed per tick so a late /api/config fetch is honored
      const sustainTicks = ticksFor(CFG.vadSustainMs); // real voice energy length
      const textTicks = ticksFor(CFG.vadTextMs); // words + energy confirm
      const failsafeTicks = ticksFor(CFG.vadFailsafeMs); // sustained-energy failsafe
      const hot = mic.rms > v.threshold; // above the voice gate this tick

      // Remember when OUR reply is audibly playing — the frame-drop gate in
      // onFrame keys off this deterministic app state.
      if (speakingRef.current) v.lastSpeakAt = now;

      // Aura: show the user is talking (from the raw mic analyser).
      if (mic.rms > v.noise * 0.8 && mic.rms > CFG.vadNoiseFloor) {
        if (!apiRef.current.userTalkingRef.current) apiRef.current.setUserTalking(true);
      } else if (mic.rms < v.noise * 0.6) {
        if (apiRef.current.userTalkingRef.current) apiRef.current.setUserTalking(false);
      }

      // Debounce: noise blips are short; real voice sustains. Count ticks.
      if (hot) {
        v.hotTicks += 1;
        v.lastUtt = now; // energy-backed speech time (drives end-of-speech)
      } else v.hotTicks = 0;
      const sustainedVoice = v.hotTicks >= sustainTicks;
      if (sustainedVoice) v.anyVoice = true; // latch: analyser really heard us
      const voiceHeard = sustainedVoice || v.textHeard || v.anyVoice;

      // Ambient noise-floor tracking ONLY while nobody has spoken recently.
      if (!voiceHeard && now - v.lastUtt > CFG.vadRecActiveMs + 200) {
        v.noise = v.noise * 0.998 + mic.rms * 0.002;
        v.threshold = Math.max(v.noise * CFG.vadGateMult, CFG.vadThresholdMin);
      }

      const assistantBusy = speakingRef.current || activeRef.current;

      // ---- OPEN an utterance on real sustained voice ------------------
      if (sustainedVoice && !streamingRef.current && !asrBusyRef.current && asrOpenRef.current) {
        streamingRef.current = true;
        v.voiceSilentSince = 0;
        asrSendJson({ type: "start" });
      }

      // ---- BARGE-IN (real sustained speech cuts the assistant) --------
      if (assistantBusy && !v.bargeLatched) {
        const enough =
          (v.textHeard && v.hotTicks >= textTicks) || v.hotTicks >= failsafeTicks;
        if (enough) {
          v.bargeLatched = true; // latch: one cut per burst
          apiRef.current.hardStop();
        }
      }
      // latch releases once the user stops producing sound (next burst can cut)
      if (v.bargeLatched && !hot) v.bargeLatched = false;

      // ---- END-OF-SPEECH: silence after a real burst closes it ---------
      // End = the analyser is cold for ~autoSendMs after real energy. Whisper
      // lag no longer matters: the transcript request is sent at "end" and the
      // authoritative "final" arrives right after.
      const recActive = now - v.lastUtt < CFG.vadRecActiveMs;
      if (streamingRef.current && voiceHeard && !assistantBusy) {
        if (hot || recActive) {
          v.voiceSilentSince = 0; // still talking
        } else {
          if (!v.voiceSilentSince) v.voiceSilentSince = now;
          else if (now - v.voiceSilentSince > CFG.autoSendMs) {
            v.voiceSilentSince = 0;
            v.hotTicks = 0;
            v.textHeard = false;
            v.anyVoice = false;
            streamingRef.current = false;
            asrBusyRef.current = true; // wait for the server's "final"
            asrSendJson({ type: "end" });
          }
        }
      }
    };

    const enableMic = async () => {
      const v = vadRef.current;
      v.noise = CFG.vadNoiseFloor;
      v.threshold = Math.max(CFG.vadNoiseFloor * CFG.vadGateMult, CFG.vadThresholdMin);
      v.hotTicks = 0;
      v.textHeard = false;
      v.anyVoice = false;
      v.bargeLatched = false;
      v.voiceSilentSince = 0;
      v.lastUtt = 0;
      v.lastSpeakAt = 0;
      streamingRef.current = false;
      asrBusyRef.current = false;
      setInterim("");
      listeningRef.current = true;
      setListening(true);
      rafVad = setInterval(vadTick, CFG.vadTickMs);
      pcmTimer = setInterval(asrSendPcm, 50);
      const ok = await engine.startMic(); // AEC-enabled mic (inside the click)
      if (!ok) {
        disableMic();
        showError("माइक अनुमति नहीं मिली। ब्राउज़र में माइक की अनुमति दें और दोबारा दबाएँ।");
        return;
      }
      connectAsr();
      const tapped = await engine.startTap(onFrame);
      if (!tapped) {
        disableMic();
        showError("इस ब्राउज़र में ऑडियो स्ट्रीमिंग उपलब्ध नहीं है — Chrome/Edge आज़माएँ।");
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

  /* ---------- UI ---------- */
  return (
    <div className="relative min-h-screen overflow-hidden bg-slate-50 text-slate-800">
      {/* soft professional background glows */}
      <div className="pointer-events-none absolute -top-48 left-1/2 h-120 w-120 -translate-x-1/2 rounded-full bg-indigo-200/40 blur-3xl" />
      <div className="pointer-events-none absolute -bottom-24 -right-24 h-80 w-80 rounded-full bg-cyan-100/60 blur-3xl" />
      <div className="pointer-events-none absolute -bottom-24 -left-24 h-80 w-80 rounded-full bg-fuchsia-100/50 blur-3xl" />

      <div className="relative mx-auto flex min-h-screen w-full max-w-2xl flex-col px-4 pb-4">
        {/* header */}
        <header className="flex items-center justify-between py-4">
          <div className="flex items-center gap-3">
            <span className="flex h-10 w-10 items-center justify-center rounded-2xl bg-gradient-to-br from-indigo-500 via-violet-500 to-fuchsia-500 text-white shadow-lg shadow-indigo-200">
              {listening ? <FaMicrophone className="h-4 w-4" /> : <FaWandMagicSparkles className="h-4 w-4" />}
            </span>
            <div>
              <h1 className="text-lg font-semibold leading-tight tracking-tight">वॉयस ट्यूटर</h1>
              <p className="text-xs leading-tight text-slate-400">रियल-टाइम वॉयस कन्वर्सेशन • हिंदी में</p>
            </div>
          </div>
          <span
            className={`flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium ${
              connected ? "bg-emerald-50 text-emerald-600" : "bg-rose-50 text-rose-500"
            }`}
          >
            <span className={`h-1.5 w-1.5 rounded-full ${connected ? "bg-emerald-500" : "bg-rose-500"}`} />
            {connected ? (listening ? "सुन रहा हूँ" : "तैयार") : "जुड़ रहा हूँ…"}
          </span>
        </header>

        {/* aura globe + live caption */}
        <main className="flex flex-1 flex-col items-center pt-2">
          <div className="relative h-[46vh] min-h-[420px] w-full max-w-2xl">
            <AuraGlobe listening={listening} speaking={speaking} userTalking={userTalking} />

            <div className="pointer-events-none absolute inset-x-0 bottom-4 flex flex-col items-center gap-2">
              {typing && (
                <span className="rounded-full bg-white/85 px-4 py-1.5 text-xs text-slate-500 shadow-sm ring-1 ring-slate-200/80 backdrop-blur">
                  सोच रहा हूँ…
                </span>
              )}
              {(speaking || activeRef.current) && !typing && (
                <span className="flex animate-pulse items-center gap-1.5 rounded-full bg-indigo-50 px-4 py-1.5 text-xs font-medium text-indigo-600 shadow-sm ring-1 ring-indigo-100">
                  <FaVolumeHigh className="h-3 w-3" />
                  बोल रहा हूँ — बीच में बोलिए, रुक जाऊँगा
                </span>
              )}
              {listening && !speaking && (
                <span className="rounded-full bg-white/85 px-4 py-1.5 text-xs text-slate-500 shadow-sm ring-1 ring-slate-200/80 backdrop-blur">
                  {userTalking ? "सुन रहा हूँ…" : "बोलिए…"}
                </span>
              )}
              {interim && (
                <span className="max-w-lg truncate rounded-full bg-cyan-50/90 px-4 py-1.5 text-xs italic text-cyan-600 shadow-sm ring-1 ring-cyan-100/80">
                  “{interim}”
                </span>
              )}
            </div>
          </div>

          {/* chat */}
          <section className="mt-2 w-full max-w-2xl">
            <div className="flex h-72 flex-col overflow-hidden rounded-3xl bg-white/90 shadow-lg shadow-slate-200/70 ring-1 ring-slate-200/80 backdrop-blur">
              <div ref={chatEl} className="flex-1 space-y-2.5 overflow-y-auto px-4 py-4">
                {messages.map((m) =>
                  m.role === "assistant" ? (
                    <div key={m.id} className="flex justify-start">
                      <div className="flex max-w-[85%] flex-col">
                        <div className="whitespace-pre-wrap rounded-2xl rounded-bl-sm bg-slate-100 px-3.5 py-2 text-sm leading-relaxed text-slate-700">
                          {m.text}
                        </div>
                        {m.elapsed != null && (
                          <div className="mt-1 flex items-center gap-2 px-1 text-[10px] text-slate-400">
                            <span className="flex items-center gap-1">
                              <FaClock className="h-2.5 w-2.5" />
                              {fmtElapsed(m.elapsed)}
                            </span>
                            <span>{fmtClock(m.ts)}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  ) : m.role === "user" ? (
                    <div key={m.id} className="flex justify-end">
                      <div className="flex max-w-[85%] flex-col items-end">
                        <div className="whitespace-pre-wrap rounded-2xl rounded-br-sm bg-gradient-to-br from-indigo-500 to-violet-500 px-3.5 py-2 text-sm leading-relaxed text-white shadow-sm">
                          {m.text}
                        </div>
                        <div className="mt-1 px-1 text-[10px] text-slate-400">{fmtClock(m.ts)}</div>
                      </div>
                    </div>
                  ) : (
                    <div key={m.id} className="flex justify-center">
                      <div className="rounded-xl bg-rose-50 px-3 py-1.5 text-xs text-rose-600 ring-1 ring-rose-100">
                        {m.text}
                      </div>
                    </div>
                  )
                )}
                {typing && (
                  <div className="flex justify-start">
                    <div className="flex items-center gap-1 rounded-2xl rounded-bl-sm bg-slate-100 px-4 py-3">
                      {[0, 1, 2].map((i) => (
                        <span
                          key={i}
                          className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-400"
                          style={{ animationDelay: `${i * 0.15}s` }}
                        />
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* composer */}
              <div className="flex items-center gap-2 border-t border-slate-100 p-3">
                <input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      submitChat(input);
                      setInput("");
                    }
                  }}
                  placeholder="लिखकर पूछिए…"
                  className="h-11 flex-1 rounded-full border border-slate-200 bg-slate-50 px-4 text-sm outline-none transition placeholder:text-slate-400 focus:border-indigo-300 focus:bg-white focus:ring-2 focus:ring-indigo-100"
                />
                <button
                  onClick={() => {
                    apiRef.current.toggleMic && apiRef.current.toggleMic();
                  }}
                  title={listening ? "माइक बंद करें" : "बोलने के लिए माइक चालू करें"}
                  disabled={micBusy}
                  className={`flex h-11 w-11 items-center justify-center rounded-full text-lg shadow-md transition disabled:opacity-50 ${
                    listening
                      ? "animate-pulse bg-rose-500 text-white shadow-rose-200"
                      : "bg-white text-slate-600 ring-1 ring-slate-200 hover:ring-indigo-300"
                  }`}
                >
                  {micBusy ? (
                    <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-current border-t-transparent" />
                  ) : listening ? (
                    <FaStop className="h-4 w-4" />
                  ) : (
                    <FaMicrophone className="h-4 w-4" />
                  )}
                </button>
                <button
                  onClick={() => {
                    submitChat(input);
                    setInput("");
                  }}
                  title="भेजें"
                  className="flex h-11 w-11 items-center justify-center rounded-full bg-gradient-to-br from-indigo-500 to-fuchsia-500 text-white shadow-lg shadow-indigo-200 transition hover:brightness-110 active:scale-95"
                >
                  <FaPaperPlane className="h-4 w-4 -translate-x-px" />
                </button>
              </div>
            </div>
            <div className="mt-2 flex items-center justify-between px-2 pb-1">
              <p className="flex items-center gap-1.5 text-[11px] text-slate-400">
                <FaMicrophone className="h-3 w-3 text-slate-400" />
                माइक चालू करके बोलिए — बीच में बोलने पर मैं रुक जाता हूँ
              </p>
              <button
                onClick={() => {
                  historyRef.current = [];
                  setMessages([{ id: nextId(), role: "assistant", text: GREETING, ts: Date.now() }]);
                  hardStop();
                }}
                className="flex items-center gap-1 text-[11px] font-medium text-slate-400 transition hover:text-rose-400"
              >
                <FaTrash className="h-2.5 w-2.5" />
                साफ़ करें
              </button>
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}
