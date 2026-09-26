"""One sglang-omni ``/v1/realtime`` session driven on behalf of one backend session."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import websockets

from .audio import PACKET_MS, Packetizer

log = logging.getLogger("bridge.realtime")

MAX_MESSAGE_BYTES = 16 * 1024 * 1024
TEXT_DELTAS = ("response.output_audio_transcript.delta", "response.output_text.delta")


@dataclass
class OutputItem:
    kind: str  # "audio" | "text" | "response_done"
    response_id: str | None
    payload: bytes | str | None
    received_s: float
    unit_id: str | None = None


@dataclass
class Metrics:
    opened_s: float = 0.0
    prefill_calls: int = 0
    prefill_audio_ms: float = 0.0
    image_calls: int = 0
    images_forwarded: int = 0
    silence_packets: int = 0
    align_pad_ms: float = 0.0
    real_packets: int = 0
    generate_calls: int = 0
    units: int = 0
    responses: int = 0
    audio_deltas: int = 0
    audio_ms_out: float = 0.0
    audio_ms_muted: float = 0.0
    breaks: int = 0
    # Per response: server receive time of its first audio delta, SSE write time of that delta,
    # and the backend arrival time of the last prefill before the response started.
    first_audio: list[dict[str, Any]] = field(default_factory=list)
    # silence fills: [timeline_ms, packets, ms since last real chunk, eager]
    fills: list[list[Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)


class RealtimeSession:
    """Owns the websocket to our server, the input timeline and the output queue.

    Input: every backend prefill is appended to one contiguous 16 kHz timeline
    cut into 80 ms packets and sent at once (seq, t_start_ms = audio time sent; a chunk
    that is not a multiple of 80 ms ends with one shorter packet, see Packetizer). The
    patched backend sends 80 ms chunks as the mic delivers them; the upstream backend sends
    1 s chunks about once a second. Neither sends anything during silence (simplex VAD)
    or before the page's init message. A pacer therefore advances the timeline with
    silent packets only when real audio has stopped: anchored at the last real chunk
    (wall time, timeline end), it keeps the timeline at
    ``anchor_end + (now - anchor_wall) - grace``, where grace is ``silence_grace_ms``
    for 1 s chunks and ``silence_grace_small_ms`` for short ones. While chunks keep
    arriving on time this target stays behind the timeline, so no silence is ever
    spliced into live speech.

    Output: audio/transcript deltas are queued; the HTTP layer drains the queue
    into whichever ``/omni/streaming_generate`` stream is current (with the patched
    backend there is always exactly one, kept open).
    """

    def __init__(
        self,
        url: str,
        *,
        backend_session: str | None,
        instructions: str | None,
        forward_images: bool,
        silence_grace_ms: float,
        silence_grace_small_ms: float = 500.0,
        output_modalities: tuple[str, ...] = ("audio",),
    ) -> None:
        self.url = url
        self.id = "br_" + uuid.uuid4().hex[:10]
        self.backend_session = backend_session
        self.instructions = instructions
        self.forward_images = forward_images
        self.silence_grace_ms = silence_grace_ms
        self.silence_grace_small_ms = silence_grace_small_ms
        self.last_chunk_ms = 1000.0  # duration of the last real prefill chunk
        self.output_modalities = output_modalities
        self.packetizer = Packetizer()
        self.out: asyncio.Queue[OutputItem] = asyncio.Queue()
        self.ws: Any = None
        self.granted: dict[str, Any] | None = None
        self.metrics = Metrics()
        self.closed = asyncio.Event()
        self.close_reason: str | None = None
        self.response_active = False
        self.current_response: str | None = None
        self.muted: set[str] = set()
        self.eager_fill = False
        self.generation = 0  # id of the current streaming_generate call
        self.last_backend_call_s = time.monotonic()
        self.last_prefill_s: float | None = None
        self._anchor_wall: float | None = None  # wall time of the last real chunk (or session open)
        self._anchor_ms = 0.0  # timeline end right after that chunk
        self.align_units = True
        self.unit_ms = 1000
        self._needs_align = True  # next real chunk starts on a model-unit boundary
        self._send_lock = asyncio.Lock()
        self._tasks: list[asyncio.Task] = []
        self._seen_first_audio: set[str] = set()
        self.dump_dir: str | None = None
        self._timeline: bytearray | None = None  # set by enable_dump
        self._out_audio: bytearray | None = None
        self.arrivals: list[list[float]] = []  # [s since open, audio ms, timeline_ms before push]
        self.server_events: list[list[Any]] = []  # [s since open, type, unit_id] (input acks omitted)

    # ------------------------------------------------------------------ lifecycle
    async def open(self, timeout_s: float = 15.0) -> dict[str, Any]:
        self.ws = await websockets.connect(
            self.url, max_size=MAX_MESSAGE_BYTES, compression=None, ping_interval=None, open_timeout=timeout_s
        )
        created = await self._recv_until("session.created", timeout_s)
        session: dict[str, Any] = {"output_modalities": list(self.output_modalities)}
        if self.instructions:
            session["instructions"] = self.instructions
        await self._send("session.update", session=session)
        updated = await self._recv_until("session.updated", timeout_s)
        self.granted = ((updated.get("session") or {}).get("sglang") or {}).get("granted") or {}
        self.unit_ms = int(self.granted.get("native_unit_ms") or 1000)
        self._anchor_wall = time.monotonic()
        self._opened_mono = self._anchor_wall
        self.metrics.opened_s = time.time()
        self._tasks = [
            asyncio.create_task(self._receiver(), name=f"{self.id}-recv"),
            asyncio.create_task(self._pacer(), name=f"{self.id}-pacer"),
        ]
        log.info("session %s open (server id %s, backend session %s)", self.id, (created.get("session") or {}).get("id"), self.backend_session)
        return self.granted

    async def close(self, reason: str, timeout_s: float = 3.0) -> None:
        if self.close_reason is None:
            self.close_reason = reason
        if self.ws is not None and not self.closed.is_set():
            try:
                await self._send("session.close")
                await asyncio.wait_for(self.closed.wait(), timeout_s)
            except Exception:  # server gone or slow: fall through to a hard close
                pass
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:
                pass
        self.closed.set()
        log.info("session %s closed (%s) %s", self.id, self.close_reason, json.dumps(self.summary()))
        if self.dump_dir:
            self._write_dump()

    def _write_dump(self) -> None:
        import os
        import wave

        path = os.path.join(self.dump_dir, self.id)
        os.makedirs(path, exist_ok=True)
        for name, data, rate in (("input_timeline.wav", self._timeline, 16000), ("output.wav", self._out_audio, 24000)):
            with wave.open(os.path.join(path, name), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(rate)
                w.writeframes(bytes(data or b""))
        with open(os.path.join(path, "summary.json"), "w") as f:
            json.dump({**self.summary(), "arrivals": self.arrivals, "fills": self.metrics.fills, "server_events": self.server_events}, f)

    def summary(self) -> dict[str, Any]:
        m = self.metrics
        return {
            "id": self.id,
            "backend_session": self.backend_session,
            "reason": self.close_reason,
            "opened_at": self.metrics.opened_s,
            "prefill_calls": m.prefill_calls,
            "prefill_audio_s": round(m.prefill_audio_ms / 1000, 2),
            "image_calls": m.image_calls,
            "images_forwarded": m.images_forwarded,
            "real_packets": m.real_packets,
            "silence_packets": m.silence_packets,
            "align_pad_s": round(m.align_pad_ms / 1000, 2),
            "timeline_s": round(self.packetizer.sent_ms / 1000, 2),
            "generate_calls": m.generate_calls,
            "units": m.units,
            "responses": m.responses,
            "audio_out_s": round(m.audio_ms_out / 1000, 2),
            "audio_muted_s": round(m.audio_ms_muted / 1000, 2),
            "breaks": m.breaks,
            "prefill_gap_ms": _gap_stats(self.arrivals),
            "first_audio": m.first_audio,
            "fill_runs": _merge_fills(m.fills),
            "errors": m.errors[:5],
        }

    # ------------------------------------------------------------------ input
    def enable_dump(self, directory: str) -> None:
        self.dump_dir = directory
        self._timeline = bytearray()
        self._out_audio = bytearray()

    async def push_audio(self, pcm16_16k: bytes) -> None:
        if len(self.arrivals) < 5000:
            self.arrivals.append(
                [round(time.monotonic() - self._opened_mono, 3), len(pcm16_16k) / 32.0, self.packetizer.sent_ms + self.packetizer.carried_ms]
            )
        self.metrics.prefill_calls += 1
        self.metrics.prefill_audio_ms += len(pcm16_16k) / 32.0
        self.last_prefill_s = time.monotonic()
        self.eager_fill = False
        self.last_chunk_ms = len(pcm16_16k) / 32.0
        if self.align_units and self._needs_align and self.last_chunk_ms >= 0.9 * self.unit_ms:
            # Upstream backend only: it cuts audio into 1000 ms chunks, the model into 1000 ms units. Starting
            # a run of chunks on a unit boundary makes every chunk complete exactly one unit on arrival, instead
            # of leaving each unit waiting (up to 1 s, 0.5 s on average) for the next chunk. Short chunks
            # (patched backend) complete units on their own, so they are never padded.
            position = self.packetizer.sent_ms + self.packetizer.carried_ms
            pad_ms = (-position) % self.unit_ms
            if pad_ms >= 1:
                pad = self.packetizer.push(b"\0\0" * int(round(pad_ms * 16)))
                self.metrics.align_pad_ms += pad_ms
                await self._send_packets(pad)
        self._needs_align = False
        packets = self.packetizer.push(pcm16_16k)
        self.metrics.real_packets += len(packets)
        await self._send_packets(packets)
        self._anchor_wall = time.monotonic()
        self._anchor_ms = self.packetizer.sent_ms + self.packetizer.carried_ms

    async def push_image(self, image_b64: str) -> None:
        self.metrics.image_calls += 1
        if not self.forward_images:
            return
        # The backend pairs a frame with the audio chunk that follows it (same image_audio_id, model_call.py:80-94),
        # so the frame is stamped at the current end of the input timeline: the start of the next unit.
        t_ms = float(self.packetizer.sent_ms + self.packetizer.carried_ms)
        await self._send("sglang.input_image.append", image=image_b64, sglang={"t_ms": t_ms})
        self.metrics.images_forwarded += 1

    def request_eager_fill(self) -> None:
        """Simplex end of speech: fill silence up to the wall clock now so the model can take its turn."""
        self.eager_fill = True

    async def _send_packets(self, packets: list[tuple[int, float, bytes]]) -> None:
        for seq, t_start_ms, body in packets:
            if self._timeline is not None:
                self._timeline.extend(body)
            await self._send(
                "input_audio_buffer.append",
                audio=base64.b64encode(body).decode("ascii"),
                sglang={"seq": seq, "t_start_ms": t_start_ms},
            )

    async def _pacer(self) -> None:
        try:
            while not self.closed.is_set():
                await asyncio.sleep(0.04)
                if self.eager_fill:
                    grace = 0.0
                elif self.last_chunk_ms >= 500:
                    grace = self.silence_grace_ms
                else:
                    grace = self.silence_grace_small_ms
                target_ms = self._anchor_ms + (time.monotonic() - self._anchor_wall) * 1000 - grace
                deficit = target_ms - self.packetizer.sent_ms
                if deficit >= PACKET_MS:
                    count = int(deficit // PACKET_MS)
                    self.metrics.silence_packets += count
                    self._needs_align = True
                    if len(self.metrics.fills) < 200:
                        self.metrics.fills.append(
                            [self.packetizer.sent_ms, count, round((time.monotonic() - self._anchor_wall) * 1000), self.eager_fill]
                        )
                    await self._send_packets(self.packetizer.silence(count))
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.warning("session %s pacer stopped: %r", self.id, exc)

    # ------------------------------------------------------------------ output
    def interrupt(self) -> int:
        """Backend break: mute the response in flight (our protocol has no cancel) and drop queued audio."""
        self.metrics.breaks += 1
        if self.current_response and self.response_active:
            self.muted.add(self.current_response)
        dropped = 0
        kept: list[OutputItem] = []
        while not self.out.empty():
            item = self.out.get_nowait()
            if item.kind == "response_done":
                kept.append(item)
            else:
                dropped += 1
        for item in kept:
            self.out.put_nowait(item)
        return dropped

    async def _receiver(self) -> None:
        try:
            async for raw in self.ws:
                now = time.monotonic()
                event = json.loads(raw)
                kind = event.get("type")
                ext = event.get("sglang") or {}
                if kind != "sglang.input_audio.accepted" and len(self.server_events) < 20000:
                    self.server_events.append([round(now - self._opened_mono, 3), kind, ext.get("unit_id") or event.get("unit_id")])
                if kind == "response.created":
                    self.response_active = True
                    self.current_response = (event.get("response") or {}).get("id")
                    self.metrics.responses += 1
                elif kind == "response.output_audio.delta":
                    pcm = base64.b64decode(event["delta"])
                    rid = event.get("response_id")
                    ms = len(pcm) / 48.0
                    if rid in self.muted:
                        self.metrics.audio_ms_muted += ms
                        continue
                    self.metrics.audio_deltas += 1
                    if self._out_audio is not None:
                        self._out_audio.extend(pcm)
                    self.out.put_nowait(OutputItem("audio", rid, pcm, now, ext.get("unit_id")))
                elif kind in TEXT_DELTAS:
                    rid = event.get("response_id")
                    if rid in self.muted:
                        continue
                    self.out.put_nowait(OutputItem("text", rid, event.get("delta") or "", now))
                elif kind == "response.done":
                    rid = (event.get("response") or {}).get("id")
                    self.response_active = False
                    self.muted.discard(rid)
                    self.out.put_nowait(OutputItem("response_done", rid, None, now))
                elif kind == "sglang.unit.done":
                    self.metrics.units += 1
                elif kind == "error":
                    self.metrics.errors.append(event.get("error") or event)
                    log.warning("session %s server error: %s", self.id, json.dumps(event)[:400])
                    if ext.get("fatal"):
                        self.close_reason = self.close_reason or "server_fatal_error"
                        break
                elif kind == "session.closed":
                    self.close_reason = self.close_reason or f"server_closed:{event.get('reason')}"
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.close_reason = self.close_reason or f"ws_error:{type(exc).__name__}"
        finally:
            self.closed.set()

    def note_first_audio(self, item: OutputItem, written_s: float) -> None:
        if item.response_id in self._seen_first_audio:
            return
        self._seen_first_audio.add(item.response_id)
        log.info(
            "session %s first audio of %s (%s): server->bridge at -%.0f ms, written to backend SSE now",
            self.id, item.response_id, item.unit_id, (written_s - item.received_s) * 1000,
        )
        self.metrics.first_audio.append(
            {
                "response_id": item.response_id,
                "unit_id": item.unit_id,
                "received_at": round(time.time() - (time.monotonic() - item.received_s), 3),
                "bridge_hold_ms": round((written_s - item.received_s) * 1000, 1),
                "since_last_prefill_ms": round((item.received_s - self.last_prefill_s) * 1000, 1)
                if self.last_prefill_s
                else None,
            }
        )

    # ------------------------------------------------------------------ wire
    async def _send(self, event_type: str, **payload: Any) -> None:
        event = {"type": event_type, "event_id": uuid.uuid4().hex, **payload}
        async with self._send_lock:
            await self.ws.send(json.dumps(event))

    async def _recv_until(self, event_type: str, timeout_s: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while True:
            raw = await asyncio.wait_for(self.ws.recv(), max(0.01, deadline - time.monotonic()))
            event = json.loads(raw)
            if event.get("type") == event_type:
                return event
            if event.get("type") == "error":
                raise RuntimeError(f"server error before {event_type}: {json.dumps(event)[:300]}")


def _merge_fills(fills: list[list[Any]]) -> list[list[Any]]:
    """Collapse consecutive pacer fills into runs: [timeline_start_ms, duration_ms, eager]."""
    runs: list[list[Any]] = []
    for start, count, _, eager in fills:
        if runs and abs(runs[-1][0] + runs[-1][1] - start) < 1e-6 and runs[-1][2] == eager:
            runs[-1][1] += count * PACKET_MS
        else:
            runs.append([start, count * PACKET_MS, eager])
    return runs


def _gap_stats(arrivals: list[list[float]]) -> dict[str, Any] | None:
    gaps = [round((b[0] - a[0]) * 1000) for a, b in zip(arrivals, arrivals[1:])]
    if not gaps:
        return None
    ordered = sorted(gaps)
    return {"n": len(gaps), "p50": ordered[len(ordered) // 2], "max": ordered[-1], "over_1300": sum(g > 1300 for g in gaps)}
