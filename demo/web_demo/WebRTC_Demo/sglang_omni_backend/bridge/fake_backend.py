"""Headless stand-in for the MiniCPM-o WebRTC backend, speaking its inference-service contract.

It replays a WAV in real time exactly the way omni_backend_code drives an inference service:
  * registry: serves POST /api/inference/register and GET /api/inference/services (in-memory,
    like inference_service_manager.py) so the bridge's registrar can be exercised;
  * health: GET http://ip:port+1/health (heartbeat_monitor.py:133);
  * init:   POST http://ip:model_port/omni/init_sys_prompt (model_call.py:53-57);
  * duplex: every 1000 ms of captured audio -> POST /omni/streaming_prefill {audio: base64 WAV 16 kHz PCM16}
            then immediately a new POST /omni/streaming_generate {mode: duplex} (omni_stream.py:726-737),
            both fire-and-forget, SSE parsed on blank lines (httpUtil.py:581-594);
  * simplex: energy VAD stand-in: voiced 1 s chunks are prefilled, the tail with last_chunk=true,
            then one generate that is read until done (omni_stream.py:657-725); prefill is skipped while
            the reply plays (model_call.py:83);
  * stop:   POST http://ip:model_port+1/omni/stop (model_call.py:232-247).

With --chunk-ms 80 --generate-loop it behaves like the latency-patched backend (deploy/backend-latency.patch):
80 ms chunks sent in order, and one duplex generate with keep_open=true, reopened as soon as it ends.

Usage: python -m bridge.fake_backend --wav input_16k.wav --out-dir runs/x [--mode duplex|simplex] [--sessions 2]
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import time
import wave
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web

from .audio import parse_wav, peak_rms


def wav_b64(pcm: bytes, rate: int = 16000) -> str:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def speech_end_s(pcm: bytes, rate: int = 16000, frame_ms: int = 20, threshold: int = 500) -> float:
    step = rate * frame_ms // 1000 * 2
    last = 0.0
    for offset in range(0, len(pcm), step):
        peak, rms = peak_rms(pcm[offset : offset + step])
        if rms > threshold:
            last = (offset + step) / 2 / rate
    return last


class Registry:
    def __init__(self) -> None:
        self.services: dict[str, dict[str, Any]] = {}

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/api/inference/register", self.register)
        app.router.add_get("/api/inference/services", self.list)
        app.router.add_delete("/api/inference/unregister/{sid}", self.unregister)
        return app

    async def register(self, request: web.Request) -> web.Response:
        body = await request.json()
        sid = f"{body['ip']}:{body['port']}"
        self.services[sid] = {**body, "service_id": sid, "status": "available"}
        return web.json_response({"service_id": sid, "message": "ok"})

    async def list(self, request: web.Request) -> web.Response:
        return web.json_response({"services": list(self.services.values()), "total": len(self.services)})

    async def unregister(self, request: web.Request) -> web.Response:
        self.services.pop(request.match_info["sid"], None)
        return web.json_response({"message": "ok"})


class Run:
    def __init__(self, base: str, control: str, http: aiohttp.ClientSession, label: str) -> None:
        self.base, self.control, self.http, self.label = base, control, http, label
        self.t0 = 0.0
        self.chunks: list[dict[str, Any]] = []
        self.audio = bytearray()
        self.text: list[str] = []
        self.generates: list[dict[str, Any]] = []
        self.prefills: list[dict[str, Any]] = []
        self.playing_until = 0.0
        self.break_after_first_audio = False
        self.break_sent: float | None = None

    def now(self) -> float:
        return time.monotonic() - self.t0

    async def prefill(self, pcm: bytes, last_chunk: bool = False) -> None:
        sent = self.now()
        async with self.http.post(
            f"{self.base}/omni/streaming_prefill",
            json={"session_id": self.label, "audio": wav_b64(pcm), "image": None, "image_audio_id": len(self.prefills) + 1, "round": 1, "last_chunk": last_chunk},
        ) as r:
            body = await r.json()
        self.prefills.append({"t": round(sent, 3), "ms": round((self.now() - sent) * 1000, 1), "ok": r.status == 200 and body.get("success")})

    async def send_break(self) -> None:
        # omni_break goes to model_port+1 (model_call.py:211-214)
        async with self.http.post(f"{self.control}/omni/break") as r:
            self.break_reply = await r.json()

    async def generate(self, mode: str, keep_open: bool = False) -> None:
        record = {"t": round(self.now(), 3), "mode": mode, "chunks": 0, "end": None}
        self.generates.append(record)
        body = {"session_id": self.label, "mode": mode, "stream": True}
        if keep_open:
            body["keep_open"] = True
        async with self.http.post(
            f"{self.base}/omni/streaming_generate", json=body,
            timeout=aiohttp.ClientTimeout(total=None),
        ) as r:
            buffer = ""
            async for piece in r.content.iter_any():
                buffer += piece.decode("utf-8", errors="ignore")
                while "\n\n" in buffer:
                    message, buffer = buffer.split("\n\n", 1)
                    message = message.strip()
                    if message.startswith("data: "):
                        message = message[6:]
                    if not message:
                        continue
                    if '"done": true' in message or '"done":true' in message:  # model_call.py:397
                        record["end"] = json.loads(message)
                        return
                    data = json.loads(message)
                    chunk = data.get("chunk_data") or {}
                    entry = {"t": round(self.now(), 3), "gen": len(self.generates) - 1}
                    if isinstance(chunk.get("wav"), str):
                        pcm = base64.b64decode(chunk["wav"])
                        self.audio.extend(pcm)
                        entry["audio_ms"] = len(pcm) / 2 / chunk.get("sample_rate", 24000) * 1000
                        entry["rate"] = chunk.get("sample_rate")
                        self.playing_until = max(self.playing_until, self.now()) + entry["audio_ms"] / 1000
                        if self.break_after_first_audio and self.break_sent is None:
                            self.break_sent = self.now()
                            asyncio.get_running_loop().create_task(self.send_break())
                    if chunk.get("text"):
                        entry["text"] = chunk["text"]
                        self.text.append(chunk["text"])
                    record["chunks"] += 1
                    self.chunks.append(entry)


async def one_session(args: argparse.Namespace, http: aiohttp.ClientSession, pcm: bytes, index: int) -> dict[str, Any]:
    base = f"http://{args.bridge_host}:{args.bridge_port}"
    control = f"http://{args.bridge_host}:{args.bridge_port + 1}"
    run = Run(base, control, http, f"fake-{index}")
    run.break_after_first_audio = args.break_after_first_audio
    async with http.get(f"{control}/health") as r:
        health = {"status": r.status, **(await r.json())}
    init_sent = time.monotonic()
    async with http.post(
        f"{base}/omni/init_sys_prompt",
        json={"highRefresh": False, "highImage": False, "timbreId": None, "timbreBase64": None, "media_type": None,
              "audio_prompt_text": None, "task_prompt_text": None, "timbre_id": None, "checkpoint_id": None, "language": "en"},
    ) as r:
        init = await r.json()
    init_ms = (time.monotonic() - init_sent) * 1000
    if not init.get("success"):
        return {"session": index, "init": init}

    speech = pcm + b"\0\0" * int(args.trailing_silence_s * 16000)
    end_s = speech_end_s(pcm)
    tasks: set[asyncio.Task] = set()

    def spawn(coro) -> None:
        task = asyncio.create_task(coro)
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    run.t0 = time.monotonic()
    chunk_s = args.chunk_ms / 1000
    chunk_bytes = int(args.chunk_ms * 32)  # upstream: 1000 ms at 16 kHz PCM16 (omni_stream.py target_duration=1000)
    total = len(speech) // chunk_bytes + (1 if len(speech) % chunk_bytes else 0)
    voiced_pending = False
    prefill_chain: asyncio.Task | None = None

    async def ordered_prefill(previous: asyncio.Task | None, chunk: bytes) -> None:
        if previous is not None:
            await asyncio.gather(previous, return_exceptions=True)
        await run.prefill(chunk)

    async def generate_loop() -> None:
        while True:
            await run.generate("duplex", keep_open=True)

    for k in range(total):
        # the backend holds a chunk until chunk_ms of audio has been captured
        await asyncio.sleep(max(0.0, (k + 1) * chunk_s - run.now()))
        chunk = speech[k * chunk_bytes : (k + 1) * chunk_bytes]
        if args.mode == "duplex" and args.generate_loop:
            prefill_chain = asyncio.create_task(ordered_prefill(prefill_chain, chunk))
            tasks.add(prefill_chain)
            prefill_chain.add_done_callback(tasks.discard)
            if k == 0:
                spawn(generate_loop())
        elif args.mode == "duplex":
            spawn(run.prefill(chunk))
            spawn(run.generate("duplex"))
        else:
            if run.now() < run.playing_until:
                continue  # simplex: prefill ignored while the reply plays (model_call.py:83)
            _, rms = peak_rms(chunk)
            if rms > 300:
                voiced_pending = True
                spawn(run.prefill(chunk))
            elif voiced_pending:
                voiced_pending = False
                await run.prefill(chunk, last_chunk=True)
                spawn(run.generate("simplex"))
    await asyncio.sleep(args.drain_s)
    for task in list(tasks):
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    async with http.post(f"{control}/omni/stop") as r:
        stop = await r.json()
    async with http.get(f"{base}/bridge/status") as r:
        status = await r.json()

    out_dir = Path(args.out_dir) / f"session{index}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_dir / "output.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(bytes(run.audio))
    audio_chunks = [c for c in run.chunks if "audio_ms" in c]
    first_audio_after_end = next((c["t"] for c in audio_chunks if c["t"] > end_s), None)
    peak, rms = peak_rms(bytes(run.audio))
    result = {
        "session": index,
        "mode": args.mode,
        "health": health,
        "init_ms": round(init_ms, 1),
        "init": init,
        "speech_end_s": round(end_s, 3),
        "first_audio_s": audio_chunks[0]["t"] if audio_chunks else None,
        "first_audio_after_speech_end_ms": round((first_audio_after_end - end_s) * 1000, 1) if first_audio_after_end else None,
        "audio_chunks": len(audio_chunks),
        "audio_out_s": round(len(run.audio) / 48000, 2),
        "audio_rms": round(rms, 1),
        "sample_rates": sorted({c.get("rate") for c in audio_chunks}),
        "transcript": "".join(run.text),
        "prefills": len(run.prefills),
        "prefill_failures": sum(1 for p in run.prefills if not p["ok"]),
        "prefill_ms_max": max((p["ms"] for p in run.prefills), default=None),
        "generates": len(run.generates),
        "generate_end_reasons": _count(g["end"].get("ended_by") if g["end"] else "cancelled" for g in run.generates),
        "break_sent_s": run.break_sent,
        "audio_after_break_s": round(sum(c["audio_ms"] for c in audio_chunks if run.break_sent and c["t"] > run.break_sent + 0.05) / 1000, 2),
        "stop": stop,
        "bridge_last_session": (status.get("history") or [None])[-1],
    }
    (out_dir / "chunks.json").write_text(json.dumps({"chunks": run.chunks, "generates": run.generates, "prefills": run.prefills}, indent=1))
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _count(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return out


async def main_async(args: argparse.Namespace) -> int:
    registry_runner = None
    if args.registry_port:
        registry = Registry()
        registry_runner = web.AppRunner(registry.app(), access_log=None)
        await registry_runner.setup()
        await web.TCPSite(registry_runner, "127.0.0.1", args.registry_port).start()
        deadline = time.monotonic() + 30
        while not registry.services and time.monotonic() < deadline:
            await asyncio.sleep(0.5)
        print(json.dumps({"registered": list(registry.services.values())}))
    pcm, rate = parse_wav(Path(args.wav).read_bytes())
    assert rate == 16000, rate
    results = []
    async with aiohttp.ClientSession() as http:
        for index in range(1, args.sessions + 1):
            result = await one_session(args, http, pcm, index)
            short = {k: result.get(k) for k in ("session", "mode", "init_ms", "speech_end_s", "first_audio_s", "first_audio_after_speech_end_ms",
                                                 "audio_chunks", "audio_out_s", "sample_rates", "transcript", "prefills", "prefill_failures",
                                                 "prefill_ms_max", "generates", "generate_end_reasons", "break_sent_s", "audio_after_break_s")}
            short["stop_ok"] = (result.get("stop") or {}).get("success")
            print(json.dumps(short, ensure_ascii=False))
            results.append(result)
            await asyncio.sleep(1.0)
    if registry_runner:
        await registry_runner.cleanup()
    ok = all(r.get("audio_chunks") and r.get("transcript") and (r.get("stop") or {}).get("success") for r in results)
    return 0 if ok else 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bridge-host", default="127.0.0.1")
    p.add_argument("--bridge-port", type=int, default=18270)
    p.add_argument("--registry-port", type=int, default=0, help="serve a fake /api/inference registry here")
    p.add_argument("--wav", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--mode", choices=("duplex", "simplex"), default="duplex")
    p.add_argument("--sessions", type=int, default=2)
    p.add_argument("--trailing-silence-s", type=float, default=8.0)
    p.add_argument("--drain-s", type=float, default=3.0)
    p.add_argument("--chunk-ms", type=float, default=1000.0, help="prefill chunk length (upstream backend 1000, patched backend 80)")
    p.add_argument("--generate-loop", action="store_true", help="duplex: one keep_open generate at a time, reopened when it ends (patched backend)")
    p.add_argument("--break-after-first-audio", action="store_true", help="POST /omni/break once the first reply audio arrives")
    raise SystemExit(asyncio.run(main_async(p.parse_args())))


if __name__ == "__main__":
    main()
