# Push-to-See Architecture — screen understanding without continuous sharing

**New interaction (default `SCREEN_PUSH_MODE=1`):**
Hold the screen button → the client captures frames *while you hold* → each
frame is described by Qwen2.5-VL **in the background during the hold** →
release (or 5 s cap) auto-sends the turn with the freshest frame → the reply
starts ~1 s after release with screen context at (almost) zero extra latency.

The vision compute now overlaps **the hold**, not the reply.

```
 OLD (continuous share)                          NEW (push-to-see)
 ──────────────────────────────                  ──────────────────────────────
 [share ON forever]                              user presses button
   tick every 1.2 s                              ├─ t=0      getDisplayMedia + frame #1
   ├─ downsample 32×24                           ├─ t=0      /api/vision warm-up #1  ─┐
   ├─ block-diff vs prev                         ├─ t=400    frame #2 (if changed)     │ VLM describe
   ├─ changed? → JPEG encode                     ├─ t=800    frame #3 …                │ runs DURING
   └─ background /api/vision                     │  …        (one in-flight max)      ─┘ the hold
      (queued behind slow                        user releases (t ≤ 5000)
       local VLM 7-15 s)                         ├─ freshest frame + wait_ms=2000
 [turn sent]                                     ├─ server: OCR cache hit/inline (≤300 ms)
   └─ chat pays leftover                         └─ LLM + TTS as usual
      vision latency                                 → first audio ≈ same as text-only
 [GPU busy between turns]                            → GPU idle when button is up
```

## Timeline decomposition (per turn, on a Kaggle T4)

Current text-only turn (from your logs): **first audio ≈ 0.76–0.81 s**

| Stage | Now (text-only) | With push-to-see |
|---|---|---|
| getDisplayMedia handshake | — | **0 ms** (requested in the *press* handler, overlaps hold) |
| First JPEG capture + upload (~70 KB @ 960 px q0.65) | — | ~20–40 ms (in hold) |
| Qwen2.5-VL-3B describe (4-bit, 384 px, 60 tok) | — | ~1.5–3 s (**in hold**, overlapped) |
| RapidOCR inline (cache hit after warm-up) | 0–300 ms | ~0 ms (hash already warm) |
| LLM first sentence (Groq, gpt-oss-120b low) | ~0.35 s | ~0.35 s |
| TTS first window (step 6, ~4 ch) | ~0.4 s | ~0.4 s |
| **First audio after release** | **~0.8 s** | **~0.8–1.1 s** (rarely 1.6 s if the very last warm-up missed) |

**Key numbers:**

- Vision latency removed from the reply path: **−1.5 to −3 s per screen-question**
  (previously the first screen question paid the full local describe; with
  continuous sharing the cache was warm but the GPU was perpetually busy,
  inflating TTS RTF and causing the 25 s→53 s warm-up queue growth you hit).
- GPU time reclaimed: continuous sharing issued a warm-up every ~1.2–4 s
  (~7 s of VLM GPU each, plus OCR) = **30–60 % of the T4 was doing vision
  work while nobody was talking**. Push-to-see does at most 2–4 describes
  per *press* and zero otherwise.
- Idle cost: **0** — no stream, no fetches, no GPU while the button is up.

## Is it realtime ("press → web AI → described with my audio")?

Yes, with the right backend. The critical budget: press-to-voice-response
**≤ 6.5 s** on T4, **≤ 2.5 s** with a hosted vision API.

| Backend | Describe time | Realtime feel |
|---|---|---|
| DashScope/OpenRouter hosted VL (7B/72B) | 0.6–1.5 s (network) | ✅ excellent — release→audio ≈ 1.5–2 s |
| Qwen2.5-VL-3B 4-bit on T4 (Kaggle) | 1.5–3 s | ✅ good — hold 2–3 s and the describe finishes *before* you release |
| Qwen2.5-VL-3B fp16 on T4 | 3–5 s | ⚠️ hold the full 5 s, or the release waits ≤2 s (wait_ms) |
| VL on CPU / M4 MPS | 8–15 s | ❌ not realtime locally — use the hosted API |

Latency model: `T_response ≈ max(0, T_describe − T_hold) + T_ocr + T_llm + T_tts_first`.
With `T_hold = 3 s` and `T_describe = 2.5 s`, the describe is fully hidden —
the response pipeline looks **identical to a text-only turn (~0.8 s after
release)**, while the model knows exactly what is on your screen.

## Why this beats continuous sharing

1. **No phantom GPU contention.** Background warm-ups used to run against the
   TTS (`GPU_GENERATE_LOCK`) — a describe landing mid-reply stretched the
   first TTS window. Push-to-see keeps describes away from replies by design.
2. **Intent is free.** A press *is* the screen-intent signal: no
   `_SCREEN_INTENT_RE` guessing, no missed frames, no "share your screen"
   dead-ends. The turn always carries a frame.
3. **Fresher than the loop.** The legacy loop captured at question time from a
   video element that might be 1.2 s stale and diff-gated; push captures every
   400 ms and ships the *release-instant* frame.
4. **Privacy** — the screen only leaves the machine while the button is held.

## Hardened for lowest latency (audit results)

- **Frame-bearing turns always inject screen context** — the intent regex can
  never downgrade a push turn to a blind text-only reply (`voice_api.py`,
  `need_screen` override).
- **Hold warm-ups are deduped by hash** — an unchanged screen during a hold
  describes once, not every 400 ms tick.
- **First-capture retry** — if the video element isn't rendering yet, ticks
  retry at 100 ms instead of silently losing the whole warm-up window.
- **Missed-warm-up safety net** — on release, a frame that was never warmed is
  described cache-first so the turn's bounded wait has something to catch.
- **Reply priority preserved** — background describes already pause while a
  chat turn synthesizes (`_screen_busy_count` gate) and OCR carries the exact
  text inline with a 300 ms budget.

## Files & knobs

- `web/ui/src/App.jsx` — `startPushCapture` / `pushCaptureTick` / `releasePush`
  (pointer-down/up on the screen button, 5 s cap, warm-ups during hold).
- `voice_api.py` — `/api/vision` accepts `force`; WS `screen.wait_ms` extends
  the bounded wait for an in-flight describe; `WS chat: screen context ready
  in X.XXs` log line = your push-path latency measurement.
- `.env` — `SCREEN_PUSH_MODE=1` (default) | `SCREEN_PUSH_MAX_MS=5000` |
  `SCREEN_PUSH_TICK_MS=400`. `SCREEN_PUSH_MODE=0` restores the legacy loop.

## How to measure (do this on Kaggle, not on the Mac)

1. `SCREEN_PUSH_MODE=1` server on the T4, UI on your Mac (Chrome).
2. Hold the button ~3 s while describing your question out loud, release.
3. Watch the server log:
   ```
   [vision] describe in 2.31s          ← during the hold (overlapped)
   WS chat: screen context ready in 0.08s (OCR 412 chars, VLM 210 chars)
   WS chat first audio frame sent 0.79s after request
   ```
   If `screen context ready` stays < 0.3 s, vision is fully off the critical
   path — you're realtime. Browser console shows `[push] hold 3000ms -> turn …`.
