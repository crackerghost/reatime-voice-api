"""Debug probe: connect to /ws/tts, send a CHAT request (same as the browser),
print the exact event order (start / text / window / audio-bytes / done).

Run:  ./omnivoice-env/bin/python ws_probe.py "नमस्ते"
"""
import asyncio
import json
import sys
import time

import websockets

URL = "ws://127.0.0.1:8000/ws/tts"


async def main(text: str) -> None:
    t0 = time.perf_counter()
    async with websockets.connect(URL, max_size=10 * 1024 * 1024) as ws:
        await ws.send(json.dumps({
            "type": "chat",
            "text": text,
            "history": [],
            "nfe_step": 8,
        }))
        print(f"[{time.perf_counter()-t0:6.2f}s] sent chat request")
        frames = 0
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=90)
                ts = time.perf_counter() - t0
                if isinstance(msg, (bytes, bytearray)):
                    frames += 1
                    print(f"[{ts:6.2f}s] AUDIO frame #{frames}: {len(msg)} bytes")
                else:
                    m = json.loads(msg)
                    if m.get("type") == "window":
                        print(f"[{ts:6.2f}s] WINDOW  #{m['n']}: {m['chars']} ch, audio {m['audio_s']}s / gen {m['gen_s']}s (RTF {m['rtf']}) | {m['text'][:40]!r}")
                    else:
                        extra = {k: v for k, v in m.items() if k != "type"}
                        print(f"[{ts:6.2f}s] {m.get('type','?').upper():7s} {extra if extra else ''}")
                    if m.get("type") in ("done", "error"):
                        break
        except asyncio.TimeoutError:
            print("TIMEOUT waiting for events")
        print(f"TOTAL audio frames: {frames}")


if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else "नमस्ते, आप कैसे हो?"
    asyncio.run(main(text))
