"""PCM helpers: WAV parsing, a linear fallback resampler and the 80 ms packetizer.

Only the standard library is used. On the normal path nothing is resampled:
the backend already sends 16 kHz PCM16 WAV (it resamples 48 kHz WebRTC audio
with scipy ``resample_poly`` before calling us, model_call.py:89,283), and it
resamples our 24 kHz output to 48 kHz itself from the ``sample_rate`` field we
return (omni_stream.py:591-601). The linear resampler below only runs if some
caller sends a WAV that is not 16 kHz.
"""

from __future__ import annotations

import array
import struct
import sys

INPUT_RATE = 16_000
PACKET_MS = 80
PACKET_SAMPLES = INPUT_RATE * PACKET_MS // 1000  # 1280
PACKET_BYTES = PACKET_SAMPLES * 2


class WavError(ValueError):
    pass


def parse_wav(data: bytes) -> tuple[bytes, int]:
    """Return (mono PCM16 little-endian bytes, sample rate) from a RIFF/WAVE blob.

    Accepts PCM (format 1), IEEE float32 (format 3) and WAVE_FORMAT_EXTENSIBLE
    wrapping either. Multi-channel audio is averaged to mono.
    """
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise WavError("not a RIFF/WAVE blob")
    offset = 12
    fmt = None
    payload = None
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        size = struct.unpack_from("<I", data, offset + 4)[0]
        body = data[offset + 8 : offset + 8 + size]
        if chunk_id == b"fmt ":
            fmt = body
        elif chunk_id == b"data":
            payload = body
            break
        offset += 8 + size + (size & 1)
    if fmt is None or payload is None or len(fmt) < 16:
        raise WavError("missing fmt or data chunk")
    tag, channels, rate, _, _, bits = struct.unpack_from("<HHIIHH", fmt)
    if tag == 0xFFFE and len(fmt) >= 26:
        tag = struct.unpack_from("<H", fmt, 24)[0]
    if tag == 1 and bits == 16:
        samples = array.array("h")
        samples.frombytes(payload[: len(payload) // 2 * 2])
        if sys.byteorder != "little":
            samples.byteswap()
        values = list(samples)
    elif tag == 3 and bits == 32:
        floats = array.array("f")
        floats.frombytes(payload[: len(payload) // 4 * 4])
        if sys.byteorder != "little":
            floats.byteswap()
        values = [max(-32768, min(32767, round(v * 32767))) for v in floats]
    else:
        raise WavError(f"unsupported WAV encoding tag={tag} bits={bits}")
    if channels > 1:
        values = [
            round(sum(values[i : i + channels]) / channels) for i in range(0, len(values) - channels + 1, channels)
        ]
    return to_pcm16(values), rate


def to_pcm16(values: list[int]) -> bytes:
    out = array.array("h", values)
    if sys.byteorder != "little":
        out.byteswap()
    return out.tobytes()


def from_pcm16(pcm: bytes) -> list[int]:
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) // 2 * 2])
    if sys.byteorder != "little":
        samples.byteswap()
    return list(samples)


def linear_resample(pcm: bytes, source_rate: int, target_rate: int) -> bytes:
    """Linear interpolation resampler (same method as tools/realtime_web_demo/test_protocol_client.py).

    Fallback only; it has no anti-alias filter, so downsampling aliases.
    """
    if source_rate == target_rate or not pcm:
        return pcm
    samples = from_pcm16(pcm)
    length = max(1, round(len(samples) * target_rate / source_rate))
    ratio = source_rate / target_rate
    last = len(samples) - 1
    out = []
    for index in range(length):
        position = min(index * ratio, last)
        left = int(position)
        right = min(left + 1, last)
        weight = position - left
        out.append(round(samples[left] * (1 - weight) + samples[right] * weight))
    return to_pcm16(out)


def peak_rms(pcm: bytes) -> tuple[int, float]:
    samples = from_pcm16(pcm)
    if not samples:
        return 0, 0.0
    peak = max(abs(v) for v in samples)
    rms = (sum(v * v for v in samples) / len(samples)) ** 0.5
    return peak, rms


class Packetizer:
    """Cuts a PCM16 stream into 80 ms packets on one sample-contiguous timeline.

    ``seq`` counts packets from 0 and ``t_start_ms`` is the audio time already sent, as the server
    requires (``input media time must be sample-contiguous``). A backend chunk (1000 ms) is not a
    multiple of 80 ms, so with ``flush=True`` the remainder goes out at once as one shorter packet
    (40 ms for a 1000 ms chunk) instead of waiting for the next chunk: holding it back would keep the
    model unit that ends in it open for another second.
    """

    def __init__(self) -> None:
        self.carry = b""
        self.seq = 0
        self.sent_samples = 0

    @property
    def sent_ms(self) -> float:
        """Audio time already emitted as packets."""
        return self.sent_samples / (INPUT_RATE / 1000)

    @property
    def carried_ms(self) -> float:
        return len(self.carry) / 2 / INPUT_RATE * 1000

    def _emit(self, body: bytes) -> tuple[int, float, bytes]:
        packet = (self.seq, self.sent_ms, body)
        self.seq += 1
        self.sent_samples += len(body) // 2
        return packet

    def push(self, pcm: bytes, flush: bool = True) -> list[tuple[int, float, bytes]]:
        buffer = self.carry + pcm
        packets = []
        offset = 0
        while len(buffer) - offset >= PACKET_BYTES:
            packets.append(self._emit(buffer[offset : offset + PACKET_BYTES]))
            offset += PACKET_BYTES
        rest = buffer[offset:]
        if flush and rest:
            packets.append(self._emit(rest))
            rest = b""
        self.carry = rest
        return packets

    def silence(self, count: int) -> list[tuple[int, float, bytes]]:
        """Emit ``count`` 80 ms silent packets; a carried partial packet goes out first, zero-padded."""
        packets = []
        for _ in range(count):
            body = self.carry + b"\0" * (PACKET_BYTES - len(self.carry))
            self.carry = b""
            packets.append(self._emit(body))
        return packets
