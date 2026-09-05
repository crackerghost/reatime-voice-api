/* Minimal WebAudio hub: one AudioContext, analysers for the TTS output and for
   the microphone, plus RMS + pitch(ish) readers that drive the AuraGlobe and
   the voice-activity (barge-in) detector. */

let ctx = null;
let speakAnalyser = null;
let micAnalyser = null;
let micStream = null;
let micSrc = null;
let tap = null;

function ensureNodes() {
  if (!speakAnalyser) speakAnalyser = ctx.createAnalyser();
  speakAnalyser.fftSize = 2048;
  speakAnalyser.smoothingTimeConstant = 0.55;
  if (!micAnalyser) micAnalyser = ctx.createAnalyser();
  micAnalyser.fftSize = 2048;
  micAnalyser.smoothingTimeConstant = 0.4;
}

function read(analyser) {
  if (!analyser) return { rms: 0, pitch: 0 };
  const buf = new Float32Array(analyser.fftSize);
  analyser.getFloatTimeDomainData(buf);
  let sum = 0;
  let crossings = 0;
  let prev = buf[0];
  for (let i = 1; i < buf.length; i++) {
    const s = buf[i];
    sum += s * s;
    if ((prev < 0 && s >= 0) || (prev >= 0 && s < 0)) crossings++;
    prev = s;
  }
  const rms = Math.sqrt(sum / buf.length);
  const zcr = crossings / buf.length;
  let pitch = 0;
  if (rms > 0.004) {
    // zero-crossing rate -> rough fundamental (Hz), clamped to a voice range
    pitch = (zcr * ctx.sampleRate) / 2;
    if (pitch < 70 || pitch > 480) pitch = 0;
  }
  return { rms, pitch };
}

export const engine = {
  get ctx() {
    return ctx;
  },
  unlock() {
    if (!ctx) {
      ctx = new (window.AudioContext || window.webkitAudioContext)();
      ensureNodes();
    }
    if (ctx.state === "suspended") ctx.resume().catch(() => {});
    return ctx;
  },
  /* Route a TTS buffer source through the "speaking" analyser to the speakers. */
  connectSpeak(source) {
    if (!speakAnalyser) return;
    source.connect(speakAnalyser);
    speakAnalyser.connect(ctx.destination);
  },
  readSpeak() {
    return read(speakAnalyser);
  },
  async startMic() {
    try {
      this.unlock();
      if (micStream) return true;
      micStream = await navigator.mediaDevices.getUserMedia({
        // echoCancellation makes Chrome subtract our own speaker output from
        // this stream — the AEC-cleaned audio is what gets transcribed, so the
        // assistant can never "hear" itself. (OpenAI/Gemini realtime use this
        // same getUserMedia + AEC path; the Web Speech API does NOT, which is
        // why it echoed.)
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      micSrc = ctx.createMediaStreamSource(micStream);
      micSrc.connect(micAnalyser); // not connected to destination (no self-monitor)
      return true;
    } catch {
      return false;
    }
  },
  stopMic() {
    if (micStream) {
      micStream.getTracks().forEach((t) => t.stop());
      micStream = null;
    }
    if (micSrc) {
      try { micSrc.disconnect(); } catch { /* noop */ } // detach node too — mic re-enable rebuilds it
    }
    micSrc = null;
  },
  /* Tap the AEC-cleaned mic stream as 16 kHz mono frames for streaming ASR. */
  async startTap(onFrame) {
    if (!micSrc) return false;
    if (tap) return true;
    try {
      // Load the worklet from a real same-origin file (web/ui/public -> dist
      // root). A Blob URL fallback is kept for setups where the file is
      // missing — Chrome can refuse Blob worklet modules under some CSPs,
      // which would otherwise silently kill STT.
      const src =
        "class PcmTap extends AudioWorkletProcessor{\n" +
        "constructor(opts){super();const outRate=(opts&&opts.processorOptions&&opts.processorOptions.outRate)||16000;this.step=outRate/sampleRate;this.phase=0;}\n" +
        "process(inputs){const ch=inputs[0]&&inputs[0][0];if(!ch||ch.length===0)return true;const out=[];for(let i=0;i<ch.length;i++){this.phase+=this.step;if(this.phase>=1){this.phase-=1;out.push(ch[i]);}}if(out.length)this.port.postMessage(new Float32Array(out));return true;}\n" +
        "}\nregisterProcessor(\"pcm-tap\",PcmTap);";
      let url = `${location.origin}/pcm-tap.js`;
      try {
        await ctx.audioWorklet.addModule(url);
      } catch {
        url = URL.createObjectURL(new Blob([src], { type: "application/javascript" }));
        await ctx.audioWorklet.addModule(url);
        URL.revokeObjectURL(url);
      }
      tap = new AudioWorkletNode(ctx, "pcm-tap", {
        processorOptions: { outRate: 16000 },
      });
      tap.port.onmessage = (e) => onFrame(e.data);
      micSrc.connect(tap);
      return true;
    } catch {
      return false;
    }
  },
  stopTap() {
    if (tap) {
      try {
        tap.disconnect();
      } catch { /* noop */ }
      tap.port.onmessage = null;
      tap = null;
    }
  },
  readMic() {
    return read(micAnalyser);
  },
  get micRunning() {
    return !!micStream;
  },
};
