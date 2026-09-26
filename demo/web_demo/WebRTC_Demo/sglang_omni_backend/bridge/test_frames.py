"""Offline check of the frame path (--forward-images) against a stub /v1/realtime server that grants images.

Servers without per-unit image input (sglang.input_image.append) reject the frame event, so the flag defaults to off.
This stub accepts exactly the #350 event shape {type, event_id, image, sglang: {t_ms}} and records it.
Run: python -m bridge.test_frames
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import wave

import aiohttp
import websockets
from aiohttp import web

from .__main__ import parse_args
from .app import Bridge

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 200 + b"\xff\xd9"  # stand-in body; the stub only checks the SOI marker


async def stub(ws, seen: list) -> None:
    await ws.send(json.dumps({"type": "session.created", "event_id": "e0", "session": {"id": "stub"}}))
    async for raw in ws:
        event = json.loads(raw)
        seen.append(event)
        if event["type"] == "session.update":
            granted = {"native_unit_ms": 1000, "input_modalities": ["audio", "image"],
                       "input_image_format": {"types": ["image/jpeg", "image/png"], "max_bytes": 524288, "max_per_unit": 1}}
            await ws.send(json.dumps({"type": "session.updated", "event_id": "e1", "session": {"sglang": {"granted": granted}}}))
        elif event["type"] == "sglang.input_image.append":
            assert set(event) == {"type", "event_id", "image", "sglang"} and set(event["sglang"]) == {"t_ms"}, event.keys()
            assert base64.b64decode(event["image"], validate=True)[:2] == b"\xff\xd8"
            await ws.send(json.dumps({"type": "sglang.input_image.accepted", "event_id": "e2"}))
        elif event["type"] == "session.close":
            await ws.send(json.dumps({"type": "session.closed", "event_id": "e3", "reason": "client_closed"}))
            return


def wav_b64(seconds: float) -> str:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
        w.writeframes(b"\x10\x00" * int(16000 * seconds))
    return base64.b64encode(buffer.getvalue()).decode()


async def run(forward: bool) -> list:
    seen: list = []
    async with websockets.serve(lambda ws: stub(ws, seen), "127.0.0.1", 18399):
        argv = ["--port", "18390", "--upstream", "http://127.0.0.1:18399", "--init-delay-s", "0"]
        if forward:
            argv.append("--forward-images")
        bridge = Bridge(parse_args(argv))
        runner = web.AppRunner(bridge.main_app())
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 18390).start()
        async with aiohttp.ClientSession() as http:
            base = "http://127.0.0.1:18390"
            assert (await (await http.post(f"{base}/omni/init_sys_prompt", json={})).json())["success"]
            await http.post(f"{base}/omni/streaming_prefill", json={"audio": wav_b64(1.0)})
            await http.post(f"{base}/omni/streaming_prefill", json={"image": base64.b64encode(JPEG).decode(), "image_audio_id": 2})
            await http.post(f"{base}/omni/streaming_prefill", json={"audio": wav_b64(1.0)})
            await http.post(f"{base}/omni/stop")
        await runner.cleanup()
    return seen


async def main() -> None:
    for forward in (False, True):
        seen = await run(forward)
        images = [e for e in seen if e["type"] == "sglang.input_image.append"]
        audio = [e for e in seen if e["type"] == "input_audio_buffer.append"]
        print(json.dumps({"forward_images": forward, "images_sent": len(images), "image_t_ms": [e["sglang"]["t_ms"] for e in images],
                          "audio_packets": len(audio), "seqs_contiguous": [e["sglang"]["seq"] for e in audio] == list(range(len(audio)))}))
        assert len(images) == (1 if forward else 0)
    print("FRAMES OK")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    asyncio.run(main())
