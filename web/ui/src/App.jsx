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
  autoSendMs: 750, // end-of-speech silence tail; <500ms chops speech into fragments that each fire a reply
  specChat: true, // server sends a fast "speculative" transcript; reply starts on it, final reconciles
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
  CFG.specChat = c.spec_chat !== undefined ? !!c.spec_chat : CFG.specChat; // live tunable
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
  const turnStartRef = useRef(0); // browser-side: when the current turn was submitted (first-audio stopwatch)
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
    speakingRef.current = false;
    setSpeaking(false);
  }, [playBuf]);

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
      turnStartRef.current = performance.now(); // browser-measured first-audio latency
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

  /* ---------- ASR readiness: server preloads Whisper at boot ---------- */
  const [asrReady, setAsrReady] = useState(false);
  const [asrReconnecting, setAsrReconnecting] = useState(false);
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
            ws.send(JSON.stringify({ type: "ping" }));
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
            assistantTextRef.current += m.text;          } else if (m.type === "error") {
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
          if (framesRef.current === 1 && turnStartRef.current) {
            console.info(
              `[voice] first audio received ${(performance.now() - turnStartRef.current).toFixed(0)}ms after submit (browser-measured, incl. network)`,
            );
          }
          pendingRef.current.push(new Blob([ev.data], { type: "audio/wav" }));
          if (!speakingRef.current) {
            speakingRef.current = true;
            api.setSpeaking(true);
            drain();
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
        asrClosedCount = 0;
        asrAttempts = 0;
        asrLastPong = Date.now();
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
          if (specSentFor) {
            // A speculative turn already went out: reconcile only on real divergence.
            const norm = (s) => s.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").trim();
            const same = norm(t) === norm(specSentFor);
            const tokensA = new Set(norm(specSentFor).split(/\s+/).filter(Boolean));
            const tokensB = new Set(norm(t).split(/\s+/).filter(Boolean));
            const common = [...tokensA].filter((w) => tokensB.has(w)).length;
            const diverged =
              !same &&
              tokensB.size > 0 &&
              (common / Math.max(1, tokensA.size) < 0.5 || t.length > specSentFor.length * 1.6);
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
      // Echo guard: drop mic frames while OUR audio is playing. In the
      // silence tail after it, only drop while no user utterance is open —
      // once the utterance IS open the user is definitely talking, and
      // cutting the tail there would swallow their first words.
      const echo = speakingRef.current ||
        (!streamingRef.current && now - v.lastSpeakAt <= CFG.speakTailMs);
      if (echo) {
        preRollClear(); // never replay our own voice as ASR input
        return;
      }
      preRollPush(arr); // always keep the rolling pre-roll (first words)
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
      if (sustainedVoice && !streamingRef.current && !asrBusyRef.current && asrOpenRef.current) {
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
      if (assistantBusy && !v.bargeLatched) {
        const enough =
          (v.textHeard && v.hotTicks >= textTicks) || v.strongTicks >= failsafeTicks;
        if (enough) {
          v.bargeLatched = true;
          apiRef.current.hardStop();
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

      if (streamingRef.current && voiceHeard && !assistantBusy) {
        if (hot || inHold || recActive) {
          v.voiceSilentSince = 0; // still talking (or holding through a dip)
        } else {
          if (!v.voiceSilentSince) v.voiceSilentSince = now;
          else if (now - v.voiceSilentSince > CFG.autoSendMs) {
            v.voiceSilentSince = 0;
            v.hotTicks = 0;
            v.textHeard = false;
            v.anyVoice = false;
            v.streamingRefCur = false;
            streamingRef.current = false;
            v.uttEndedAt = now;
            asrBusyRef.current = true;
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
            {connected ? (listening ? (asrReady ? "सुन रहा हूँ" : "वॉर्म-अप…") : "तैयार") : "जुड़ रहा हूँ…"}
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
