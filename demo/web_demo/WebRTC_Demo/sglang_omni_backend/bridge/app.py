"""HTTP face of the bridge: the inference-service contract the MiniCPM-o WebRTC backend expects.

Contract (see BRIDGE.md for file:line references into OpenSQZ/MiniCPM-V-CookBook):
  main port P (registered as both ``port`` and ``model_port``):
    POST /omni/init_sys_prompt      -> open one /v1/realtime session
    POST /omni/streaming_prefill    -> {audio: base64 WAV 16 kHz PCM16 | image: base64 JPEG}
    POST /omni/streaming_generate   -> SSE: data: {"chunk_idx", "chunk_data": {"wav", "sample_rate", "text"}} ... data: {"done": true}
    GET  /health
  control port P+1 (the backend health-checks and breaks/stops here):
    GET  /health, POST /omni/break, POST /omni/stop
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import hmac
import json
import logging
import time
from typing import Any

import aiohttp
from aiohttp import web

from .audio import INPUT_RATE, WavError, linear_resample, parse_wav
from .realtime import OutputItem, RealtimeSession

log = logging.getLogger("bridge.app")


class Bridge:
    def __init__(self, args: Any) -> None:
        self.args = args
        self.session: RealtimeSession | None = None
        self.session_lock = asyncio.Lock()
        self.service_id = f"{args.advertise_ip}:{args.port}"
        self.registered = False
        self.http: aiohttp.ClientSession | None = None
        self.history: list[dict[str, Any]] = []
        self._upstream_ok: tuple[float, bool] = (0.0, False)
        self._bg: list[asyncio.Task] = []

    # ------------------------------------------------------------------ app wiring
    def main_app(self) -> web.Application:
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.router.add_post("/omni/init_sys_prompt", self.init_sys_prompt)
        app.router.add_post("/omni/streaming_prefill", self.streaming_prefill)
        app.router.add_post("/omni/streaming_generate", self.streaming_generate)
        app.router.add_post("/omni/break", self.omni_break)
        app.router.add_post("/omni/stop", self.omni_stop)
        app.router.add_get("/health", self.health)
        app.router.add_get("/bridge/status", self.status)
        return app

    def control_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/health", self.health)
        app.router.add_get("/", self.health)
        app.router.add_post("/omni/break", self.omni_break)
        app.router.add_post("/omni/stop", self.omni_stop)
        return app

    async def start_background(self) -> None:
        self.http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        if self.args.register_url:
            self._bg.append(asyncio.create_task(self._registrar()))
        self._bg.append(asyncio.create_task(self._idle_watchdog()))

    async def shutdown(self) -> None:
        for task in self._bg:
            task.cancel()
        await asyncio.gather(*self._bg, return_exceptions=True)
        if self.session is not None:
            await self._end_session("bridge_shutdown")
        if self.args.register_url and self.registered and self.http is not None:
            try:
                async with self.http.delete(f"{self.args.register_url}/api/inference/unregister/{self.service_id}") as r:
                    log.info("unregistered %s: HTTP %s", self.service_id, r.status)
            except Exception as exc:
                log.warning("unregister failed: %r", exc)
        if self.http is not None:
            await self.http.close()

    # ------------------------------------------------------------------ registration
    async def _registrar(self) -> None:
        """Register once, then re-register only if the backend lost us (it keeps services in memory).

        Re-registering an existing entry would reset it to AVAILABLE and drop a live user's
        lock (inference_service_manager.py register_service), so we check the list first.
        """
        payload = {
            "ip": self.args.advertise_ip,
            "port": self.args.port,
            "model_port": self.args.port,
            "service_name": self.args.service_name,
            "model_type": self.args.model_type,
            "session_type": self.args.session_type,
        }
        base = self.args.register_url
        while True:
            try:
                async with self.http.get(f"{base}/api/inference/services") as r:
                    services = (await r.json()).get("services", []) if r.status == 200 else None
                present = services is not None and any(s.get("service_id") == self.service_id for s in services)
                if not present:
                    async with self.http.post(f"{base}/api/inference/register", json=payload) as r:
                        text = await r.text()
                        self.registered = r.status == 200
                        log.info("register %s -> HTTP %s %s", payload, r.status, text[:200])
            except Exception as exc:
                log.warning("registration check failed (%s): %r", base, exc)
            await asyncio.sleep(self.args.register_interval_s)

    # ------------------------------------------------------------------ health
    async def _upstream_healthy(self) -> bool:
        stamp, ok = self._upstream_ok
        if time.monotonic() - stamp < 2.0:
            return ok
        try:
            async with self.http.get(f"{self.args.upstream_http}/health", timeout=aiohttp.ClientTimeout(total=3)) as r:
                ok = r.status == 200
        except Exception:
            ok = False
        self._upstream_ok = (time.monotonic(), ok)
        return ok

    async def health(self, request: web.Request) -> web.Response:
        ok = await self._upstream_healthy()
        body = {
            "status": "healthy" if ok else "unhealthy",
            "message": "sglang-omni realtime bridge",
            "backend": "sglang-omni-bridge",
            "upstream": self.args.upstream_ws,
            "upstream_ok": ok,
            "session_active": self.session is not None,
            "registered": self.registered,
        }
        return web.json_response(body, status=200 if ok else 503)

    async def status(self, request: web.Request) -> web.Response:
        current = self.session.summary() if self.session else None
        return web.json_response({"current": current, "history": self.history[-10:]})

    # ------------------------------------------------------------------ session lifecycle
    async def _gate_busy(self) -> bool:
        if not self.args.gate_status_url:
            return False
        try:
            async with self.http.get(self.args.gate_status_url, timeout=aiohttp.ClientTimeout(total=2)) as r:
                return bool((await r.json()).get("busy"))
        except Exception:
            return False

    async def _end_session(self, reason: str) -> dict[str, Any] | None:
        session, self.session = self.session, None
        if session is None:
            return None
        await session.close(reason)
        summary = session.summary()
        self.history.append(summary)
        return summary

    async def init_sys_prompt(self, request: web.Request) -> web.Response:
        body = await _json(request)
        async with self.session_lock:
            if self.session is not None:
                # One session at a time: the backend locks this service per user, so a new init
                # means the previous user is gone (their stop may have been lost).
                await self._end_session("superseded_by_init")
            if await self._gate_busy():
                return web.json_response({"success": False, "detail": "public gate session active"}, status=503)
            session = RealtimeSession(
                self.args.upstream_ws,
                backend_session=body.get("session_id"),
                instructions=self.args.instructions,
                forward_images=self.args.forward_images,
                silence_grace_ms=self.args.silence_grace_ms,
                silence_grace_small_ms=self.args.silence_grace_small_ms,
            )
            session.align_units = not self.args.no_unit_align
            if self.args.dump_dir:
                session.enable_dump(self.args.dump_dir)
            try:
                granted = await session.open()
            except Exception as exc:
                log.error("upstream session open failed: %r", exc)
                await session.close("open_failed")
                return web.json_response({"success": False, "detail": f"upstream open failed: {exc}"}, status=502)
            self.session = session
        # The backend relays <state><model_init_success> to the page as soon as this returns (or as soon as the
        # page joins the room, whichever is later). If the page's peer connection is not up yet the message is
        # lost and the page never starts the mic; the C++ wrapper's init takes seconds, which hides the race.
        # With LiveKit credentials we hold the reply until the room's participants are ACTIVE; otherwise we
        # hold it for a fixed --init-delay-s.
        waited = await self._wait_page_ready() if self.args.livekit_url else None
        if waited is None and self.args.init_delay_s > 0:
            await asyncio.sleep(self.args.init_delay_s)
        ignored = sorted(k for k, v in body.items() if v not in (None, "", False) and k not in ("language",))
        log.info("init_sys_prompt -> %s (page ready after %s s; ignored fields: %s)", session.id, waited, ignored)
        return web.json_response(
            {
                "success": True,
                "message": "sglang-omni realtime session opened",
                "session_id": session.id,
                "duplex_mode": True,
                "granted_unit_ms": granted.get("native_unit_ms"),
                "ignored_fields": ignored,
            }
        )

    # ------------------------------------------------------------------ LiveKit readiness
    def _livekit_token(self, video: dict[str, Any]) -> str:
        def b64(raw: bytes) -> str:
            return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

        now = int(time.time())
        header = b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
        claims = {"iss": self.args.livekit_key, "sub": "sglang-omni-bridge", "nbf": now - 10, "exp": now + 600, "video": video}
        payload = b64(json.dumps(claims).encode())
        signature = hmac.new(self.args.livekit_secret.encode(), f"{header}.{payload}".encode(), hashlib.sha256).digest()
        return f"{header}.{payload}.{b64(signature)}"

    async def _room_service(self, method: str, body: dict[str, Any], video: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.args.livekit_url}/twirp/livekit.RoomService/{method}"
        headers = {"Authorization": f"Bearer {self._livekit_token(video)}"}
        async with self.http.post(url, json=body, headers=headers, timeout=aiohttp.ClientTimeout(total=3)) as r:
            return await r.json()

    async def _wait_page_ready(self) -> float | None:
        """Wait until the newest LiveKit room has >= 2 participants, all ACTIVE (robot + page); None on timeout."""
        start = time.monotonic()
        while time.monotonic() - start < self.args.page_ready_timeout_s:
            try:
                rooms = (await self._room_service("ListRooms", {}, {"roomList": True})).get("rooms") or []
                if rooms:
                    room = max(rooms, key=lambda r: int(r.get("creation_time") or 0))["name"]
                    listed = await self._room_service("ListParticipants", {"room": room}, {"roomAdmin": True, "room": room})
                    people = listed.get("participants") or []
                    if len(people) >= 2 and all(p.get("state") == "ACTIVE" for p in people):
                        await asyncio.sleep(self.args.page_ready_margin_s)
                        return round(time.monotonic() - start, 2)
            except Exception as exc:
                log.debug("livekit readiness poll failed: %r", exc)
            await asyncio.sleep(0.25)
        log.warning("page not ACTIVE in LiveKit after %.0f s; replying to init anyway", self.args.page_ready_timeout_s)
        return None

    async def streaming_prefill(self, request: web.Request) -> web.Response:
        body = await _json(request)
        session = self.session
        if session is None or session.closed.is_set():
            raise web.HTTPBadRequest(text=json.dumps({"detail": "no active session; call /omni/init_sys_prompt first"}))
        session.last_backend_call_s = time.monotonic()
        audio_ms = 0.0
        if body.get("audio"):
            try:
                pcm, rate = parse_wav(base64.b64decode(body["audio"]))
            except (WavError, binascii.Error, ValueError) as exc:
                raise web.HTTPBadRequest(text=json.dumps({"detail": f"audio decode failed: {exc}"}))
            if rate != INPUT_RATE:
                pcm = linear_resample(pcm, rate, INPUT_RATE)
            audio_ms = len(pcm) / 32.0
            await session.push_audio(pcm)
        if body.get("image"):
            await session.push_image(body["image"])
        if not body.get("audio") and not body.get("image"):
            raise web.HTTPBadRequest(text=json.dumps({"detail": "audio or image required"}))
        return web.json_response(
            {
                "success": True,
                "session_id": session.id,
                "audio_duration_seconds": audio_ms / 1000,
                "timeline_ms": session.packetizer.sent_ms,
                "backend": "sglang-omni-bridge",
            }
        )

    async def streaming_generate(self, request: web.Request) -> web.StreamResponse:
        body = await _json(request)
        session = self.session
        if session is None or session.closed.is_set():
            raise web.HTTPBadRequest(text=json.dumps({"detail": "no active session"}))
        mode = body.get("mode") or "duplex"
        # Patched backend: its only open generate, reopened as soon as it ends, so keep it open for a long
        # window instead of ending it after every idle second (upstream backend: one generate per chunk).
        keep_open = bool(body.get("keep_open"))
        idle_window_s = self.args.keep_open_window_s if keep_open else self.args.duplex_window_s
        session.last_backend_call_s = time.monotonic()
        session.metrics.generate_calls += 1
        session.generation += 1
        generation = session.generation
        if mode == "simplex":
            session.request_eager_fill()
        response = web.StreamResponse(
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )
        await response.prepare(request)
        started = time.monotonic()
        chunk_idx = 0
        saw_response = session.response_active
        ended_by = "window"

        async def emit(obj: dict[str, Any]) -> None:
            await response.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode())

        try:
            while True:
                now = time.monotonic()
                elapsed = now - started
                if session.closed.is_set() or self.session is not session:
                    ended_by = "session_closed"
                    break
                if session.generation != generation:
                    ended_by = "superseded"
                    break
                try:
                    item: OutputItem = await asyncio.wait_for(session.out.get(), 0.05)
                except asyncio.TimeoutError:
                    item = None
                if item is not None:
                    if item.kind == "audio":
                        await emit(
                            {
                                "chunk_idx": chunk_idx,
                                "chunk_data": {
                                    "wav": base64.b64encode(item.payload).decode("ascii"),
                                    "sample_rate": 24000,
                                },
                            }
                        )
                        session.note_first_audio(item, time.monotonic())
                        session.metrics.audio_ms_out += len(item.payload) / 48.0
                        chunk_idx += 1
                        saw_response = True
                    elif item.kind == "text":
                        if item.payload:
                            await emit({"chunk_idx": chunk_idx, "chunk_data": {"text": item.payload}})
                            chunk_idx += 1
                        saw_response = True
                    elif item.kind == "response_done":
                        if mode == "simplex" and saw_response:
                            ended_by = "end_of_turn"
                            break
                    continue
                # Queue empty: decide whether this call's window is over.
                if mode == "duplex":
                    if elapsed >= idle_window_s and not session.response_active:
                        break
                    if elapsed >= self.args.duplex_max_s:
                        ended_by = "duplex_cap"
                        break
                else:
                    if not saw_response and not session.response_active and elapsed >= self.args.simplex_wait_s:
                        ended_by = "no_response"
                        break
                    if elapsed >= self.args.simplex_max_s:
                        ended_by = "simplex_cap"
                        break
            await emit(
                {
                    "done": True,
                    "is_listen": not session.response_active,
                    "end_of_turn": ended_by == "end_of_turn",
                    "chunks_received": chunk_idx,
                    "ended_by": ended_by,
                }
            )
        except (ConnectionResetError, asyncio.CancelledError):
            # Backend dropped the stream (break or stop); nothing to clean up here.
            pass
        return response

    async def omni_break(self, request: web.Request) -> web.Response:
        session = self.session
        dropped = session.interrupt() if session else 0
        if session:
            session.generation += 1  # end the current generate stream now
        return web.json_response({"success": True, "message": "muted current response", "state": "break", "dropped": dropped})

    async def omni_stop(self, request: web.Request) -> web.Response:
        async with self.session_lock:
            summary = await self._end_session("backend_stop")
        return web.json_response(
            {"success": True, "message": "session closed", "state": "session_stop", "session_id": (summary or {}).get("id")}
        )

    async def _backend_still_locked(self) -> bool:
        """True while the backend still holds this service for a user (it releases on stop/logout/lock timeout)."""
        if not self.args.register_url:
            return False
        try:
            async with self.http.get(f"{self.args.register_url}/api/inference/services") as r:
                services = (await r.json()).get("services", [])
        except Exception:
            return False
        return any(s.get("service_id") == self.service_id and s.get("status") == "busy" for s in services)

    async def _idle_watchdog(self) -> None:
        """End a session whose backend went away without calling /omni/stop.

        Simplex callers legitimately stay silent (no prefill) for as long as the user is quiet, so idleness
        alone is not enough: while the backend still shows this service as locked, the session is kept up to
        idle_cap_s.
        """
        while True:
            await asyncio.sleep(2.0)
            session = self.session
            if session is None:
                continue
            idle = time.monotonic() - session.last_backend_call_s
            expired = idle > self.args.idle_timeout_s and (
                idle > self.args.idle_cap_s or not await self._backend_still_locked()
            )
            if session.closed.is_set() or expired:
                async with self.session_lock:
                    if self.session is session:
                        reason = session.close_reason or f"idle_{int(idle)}s"
                        log.info("watchdog ends session %s (%s)", session.id, reason)
                        await self._end_session(reason)


async def _json(request: web.Request) -> dict[str, Any]:
    if not request.can_read_body:
        return {}
    try:
        data = await request.json()
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}
