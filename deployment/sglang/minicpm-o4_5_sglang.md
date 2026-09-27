# MiniCPM-o 4.5 SGLang-Omni Full-Duplex Deployment Guide

[SGLang-Omni](https://github.com/sgl-project/sglang-omni) serves the native full-duplex mode of MiniCPM-o 4.5: the model listens and speaks at the same time, decides once per second whether to stay silent or talk, and can take one camera frame per second. The service is the WebSocket endpoint `/v1/realtime`, with 16 kHz audio in and 24 kHz speech plus text out. On the 727 Full-Duplex-Bench v1.0 samples, 95.75 % of its per-unit listen/speak decisions match the official reference implementation.

For every configuration field, the full protocol and tuning advice, see the SGLang-Omni page [MiniCPM-o 4.5 Full-Duplex](https://github.com/sgl-project/sglang-omni/blob/main/docs/cookbook/minicpm_o_full_duplex.md).

## 1. Environment Setup

### 1.1 Install SGLang-Omni

> [!NOTE]
> Full-duplex serving needs SGLang-Omni installed from source (it is not in the v0.1.6 release). The pinned stack is SGLang 0.5.20, FlashInfer 0.6.18 and transformers 5.12.1.

```bash
git clone https://github.com/sgl-project/sglang-omni.git
cd sglang-omni
pip install --upgrade pip uv
uv venv .venv -p 3.12
source .venv/bin/activate
uv pip install --prerelease=allow -e .
```

Verify the installation:

```bash
python -c "import sglang; print(sglang.__version__)"   # 0.5.20
```

### 1.2 Download the Model

```bash
hf download openbmb/MiniCPM-o-4_5 --revision 503e754207c94da6bb26850b4469f367c9ea3582 --local-dir MiniCPM-o-4_5
```

All numbers below were measured with this revision (about 20 GB).

### 1.3 GPU Memory

The four stages share one GPU. An H200 (141 GB) runs the default configuration and uses about 72 GB. A 48 GB card (tested: RTX A6000) needs explicit KV pool sizes as in 2.2 and uses about 32 GB, plus about 1.2 GB with image input.

## 2. Launch the Service

### 2.1 Start the API Server

From the `sglang-omni` directory:

```bash
sgl-omni serve --config examples/full_duplex/minicpmo.yaml --model-path ./MiniCPM-o-4_5 --enable-realtime --port 18260
```

**Parameters:**
- `--config`: the full-duplex pipeline configuration, `examples/full_duplex/minicpmo.yaml`
- `--model-path`: model path; overrides `model_path` in the YAML
- `--enable-realtime`: mounts the WebSocket endpoint `/v1/realtime`; required for full duplex
- `--port`: server port, 8000 by default; 18260 matches the demo defaults in section 4

Check that the server is ready:

```bash
curl -s http://127.0.0.1:18260/health
curl -s http://127.0.0.1:18260/v1/realtime/capabilities
```

The first session after every start compiles kernels and is tens of seconds slower; run one warm-up session before real use.

### 2.2 Common Settings

Copy the YAML and edit it. For a 48 GB card:

```yaml
config_cls: MiniCPMODuplexPipelineConfig
model_path: /path/to/MiniCPM-o-4_5
stages:
  thinker:
    engine:
      kv_cache_bytes: 4GiB
  talker:
    engine:
      kv_cache_bytes: 2GiB
```

Other common fields:

- `reference_audio`: reference voice WAV, by default the checkpoint's `assets/HT_ref_audio.wav`; shared by all sessions on the server
- `max_sessions`: concurrent sessions (default 2); further connections get HTTP 503
- `stages.thinker.engine.context_length`: context per session (default 8192, at most 40960)

## 3. Calling the Service

### 3.1 Full-Duplex Speech

Protocol flow: connect to `ws://127.0.0.1:18260/v1/realtime`, send `session.update` (optionally with `instructions` as the system prompt), then send `input_audio_buffer.append` in real time (base64 mono PCM16 at 16 kHz, 80 ms per packet, with `sglang: {seq, t_start_ms}`) while receiving `response.output_audio.delta` (24 kHz PCM16) and `response.output_audio_transcript.delta` (text). At the end of input, send `sglang.input_audio.end`, wait for `sglang.input_audio.drained`, then send `session.close`.

The script below streams a WAV file to the server in real time, prints the text and saves the reply. It needs only `websockets`:

```python
"""Stream a WAV (and optional 1 fps JPEG frames) through a MiniCPM-o full-duplex session."""

import argparse
import asyncio
import base64
import json
import uuid
import wave
from pathlib import Path

import websockets

RATE = 16000
PACKET_MS = 80
PACKET_BYTES = RATE * 2 * PACKET_MS // 1000  # 2560 bytes of PCM16


def read_pcm16(path):
    with wave.open(str(path), "rb") as f:
        if (f.getframerate(), f.getnchannels(), f.getsampwidth()) != (RATE, 1, 2):
            raise SystemExit("input must be a 16 kHz mono 16-bit PCM WAV")
        return f.readframes(f.getnframes())


async def run(args):
    # Trailing silence gives the model time to answer after the last word.
    pcm = read_pcm16(args.input) + bytes(RATE * 2 * args.tail_s)
    frames = sorted(args.frames.glob("*.jpg")) if args.frames else []
    reply = bytearray()
    flags, last = {}, {}

    async with websockets.connect(args.url, max_size=16 * 1024 * 1024) as ws:

        def flag(name):
            return flags.setdefault(name, asyncio.Event())

        async def send(kind, **body):
            await ws.send(json.dumps({"type": kind, "event_id": uuid.uuid4().hex, **body}))

        async def receive():
            async for raw in ws:
                event = json.loads(raw)
                kind = event["type"]
                last[kind] = event
                if kind == "response.output_audio.delta":
                    reply.extend(base64.b64decode(event["delta"]))
                elif kind in ("response.output_audio_transcript.delta", "response.output_text.delta"):
                    print(event["delta"], end="", flush=True)
                elif kind == "response.done":
                    print()
                elif kind == "error":
                    print("[error]", event["error"])
                    if event.get("sglang", {}).get("fatal"):
                        raise RuntimeError(event["error"]["message"])
                flag(kind).set()

        receiver = asyncio.create_task(receive())

        async def until(name):
            waiter = asyncio.create_task(flag(name).wait())
            await asyncio.wait({waiter, receiver}, return_when=asyncio.FIRST_COMPLETED)
            waiter.cancel()
            if receiver.done():
                receiver.result()  # re-raises a fatal server error

        await until("session.created")
        session = {"output_modalities": ["audio"]}
        if args.instructions:
            session["instructions"] = args.instructions
        await send("session.update", session=session)
        await until("session.updated")
        granted = last["session.updated"]["session"]["sglang"]["granted"]
        if frames and "image" not in granted["input_modalities"]:
            raise SystemExit("this server does not accept image input")

        loop = asyncio.get_running_loop()
        start = loop.time()
        for seq, offset in enumerate(range(0, len(pcm), PACKET_BYTES)):
            await asyncio.sleep(max(0.0, start + seq * PACKET_MS / 1000 - loop.time()))
            t_ms = seq * PACKET_MS
            unit = t_ms // 1000
            if unit < len(frames) and t_ms - unit * 1000 < PACKET_MS:
                # One frame per 1 s unit, sent with the first packet of that unit.
                image = base64.b64encode(frames[unit].read_bytes()).decode()
                await send("sglang.input_image.append", image=image, sglang={"t_ms": float(unit * 1000)})
            audio = base64.b64encode(pcm[offset : offset + PACKET_BYTES]).decode()
            await send("input_audio_buffer.append", audio=audio, sglang={"seq": seq, "t_start_ms": float(t_ms)})

        await send("sglang.input_audio.end")
        await until("sglang.input_audio.drained")
        await send("session.close")
        await until("session.closed")
        receiver.cancel()

    with wave.open(str(args.output), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(granted["output_audio_format"]["rate"])
        f.writeframes(bytes(reply))
    print(f"wrote {len(reply) / 2 / granted['output_audio_format']['rate']:.2f} s of audio to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:18260/v1/realtime")
    parser.add_argument("--input", type=Path, required=True, help="16 kHz mono PCM16 WAV")
    parser.add_argument("--output", type=Path, default=Path("reply.wav"))
    parser.add_argument("--frames", type=Path, help="directory of JPEG frames, one per second, in name order")
    parser.add_argument("--instructions", help="system prompt; the server default is used when omitted")
    parser.add_argument("--tail-s", type=int, default=4, help="seconds of silence appended after the input")
    asyncio.run(run(parser.parse_args()))
```

Save it as `duplex_client.py` and run:

```bash
ffmpeg -i question.mp3 -ac 1 -ar 16000 -c:a pcm_s16le input.wav
python duplex_client.py --input input.wav --output reply.wav
```

The script appends 4 s of silence to the input: the model answers only after it hears silence.

### 3.2 Full Duplex with Images

When `session.sglang.granted.input_modalities` in `session.updated` contains `"image"`, each 1 s unit can carry one frame: send `sglang.input_image.append` with `image` as base64 JPEG or PNG bytes (no `data:` prefix, at most 512 KiB) and `sglang.t_ms` as the frame's media time. The frame belongs to unit `floor(t_ms / 1000)` and must arrive before the audio that completes that unit. The script above sends each unit's frame just before the unit's first audio packet.

Extract the audio and one frame per second from a video (short side 448 px; the example assumes a landscape video):

```bash
ffmpeg -i clip.mp4 -ac 1 -ar 16000 -c:a pcm_s16le input.wav
mkdir -p frames
ffmpeg -i clip.mp4 -vf "fps=1,scale=-2:448" -q:v 4 frames/%04d.jpg
python duplex_client.py --input input.wav --frames frames --output reply.wav
```

Each frame adds 66 tokens to the thinker context. With the default 8192 context, a session with one frame per second runs out after about 100 s; the server then sends a `context_exhausted` error and closes the session. For longer video conversations set `context_length` to 32768 (about 7 minutes) and raise the thinker's `kv_cache_bytes` accordingly (144 KiB per token per session).

## 4. Web Demo

### 4.1 SGLang-Omni Browser Demo

```bash
python tools/realtime_web_demo/serve.py --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080/`, set the WebSocket URL to `ws://localhost:18260/v1/realtime`, connect and start the microphone; the Camera button is enabled when the server accepts images. Browsers allow the microphone and camera only on `localhost` or HTTPS, so forward the ports over SSH when the server is remote:

```bash
ssh -L 8080:127.0.0.1:8080 -L 18260:127.0.0.1:18260 <gpu-host>
```

### 4.2 Official WebRTC Demo

The [WebRTC Demo](../../demo/web_demo/WebRTC_Demo/README.md) can use SGLang-Omni instead of `llama-server` as its inference service: a bridge process registers with the demo backend as the `o45-cpp` service, and the front end is unchanged. Setup is described in [`sglang_omni_backend/README.md`](../../demo/web_demo/WebRTC_Demo/sglang_omni_backend/README.md). The bridge connects to `http://127.0.0.1:18260` by default, handles one call at a time and forwards camera frames with `--forward-images`.

## 5. Notes

1. **GPU memory**: on any card smaller than an H200, set `kv_cache_bytes` for the thinker and the talker; otherwise the thinker KV pool is too small for a conversation.
2. **Concurrency**: one H200 runs 2 sessions cleanly (0.45 % of units miss their 1 s deadline), 4 with occasional hiccups (1.5 %) and 8 with frequent stalls (22 %). Run more servers for more users.
3. **Session length**: there is no sliding window; the whole conversation stays in context. With the default 8192 context, an audio-only session lasts about 9.5 minutes and a session with one frame per second about 100 s; the server then sends `context_exhausted` and closes the session.
4. **Decoding and persona**: the server always decodes greedily with repetition penalty 1.05 and forces the first 3 units of a session to listen, as the official reference does; this cannot be changed per session. `instructions` must be given in the first `session.update` and cannot change afterwards; the voice comes from the server's `reference_audio`.
5. **Interruptions**: the protocol has no way to cancel a reply; the model decides when to stop. After the user cuts in, the model keeps talking for about 1 more second at the median and then yields; short interjections such as "mm-hm" are usually taken as backchannel and the model continues. A reply usually starts 1.5 to 2 s after the user stops talking.
6. **Audio format**: input must be 16 kHz mono PCM16, `sglang.seq` counts up from 0 without gaps, and `t_start_ms` equals the duration of audio already sent; output is 24 kHz.
7. **Warm-up**: the first session after every restart compiles kernels and is tens of seconds slower.
8. **Scope**: this page covers SGLang-Omni full-duplex serving only. For single-turn image chat with SGLang (`python -m sglang.launch_server` with the OpenAI-compatible API), see the [SGLang documentation](https://docs.sglang.ai/).
