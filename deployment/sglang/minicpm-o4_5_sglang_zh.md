# MiniCPM-o 4.5 SGLang-Omni 全双工部署指南

[SGLang-Omni](https://github.com/sgl-project/sglang-omni) 支持 MiniCPM-o 4.5 的原生全双工模式：模型边听边说，每 1 秒自己决定一次是沉默还是开口，可选每秒输入一帧摄像头画面。服务通过 WebSocket 端点 `/v1/realtime` 提供，输入 16 kHz 音频，输出 24 kHz 语音及文字。在 Full-Duplex-Bench v1.0 的 727 条样本上，逐 unit 听/说决策与官方参考实现 95.75 % 一致。

完整的配置说明、协议细节与调优建议见 SGLang-Omni 文档 [MiniCPM-o 4.5 Full-Duplex](https://github.com/sgl-project/sglang-omni/blob/main/docs/cookbook/minicpm_o_full_duplex.md)。

## 1. 环境准备

### 1.1 安装 SGLang-Omni

> [!NOTE]
> 全双工服务需要从源码安装 SGLang-Omni（v0.1.6 发布版中还没有）。依赖栈固定为 SGLang 0.5.20、FlashInfer 0.6.18、transformers 5.12.1。

```bash
git clone https://github.com/sgl-project/sglang-omni.git
cd sglang-omni
pip install --upgrade pip uv
uv venv .venv -p 3.12
source .venv/bin/activate
uv pip install --prerelease=allow -e .
```

安装完成后可以使用以下命令验证：

```bash
python -c "import sglang; print(sglang.__version__)"   # 0.5.20
```

### 1.2 下载模型

```bash
hf download openbmb/MiniCPM-o-4_5 --revision 503e754207c94da6bb26850b4469f367c9ea3582 --local-dir MiniCPM-o-4_5
```

该 revision 即下文所有数据的实测版本，约 20 GB。

### 1.3 显存

四个 stage 共用一张卡。H200（141 GB）直接使用默认配置，占用约 72 GB。48 GB 的卡（实测 RTX A6000）需要按 2.2 显式设置 KV 池大小，占用约 32 GB，开启图像输入再加约 1.2 GB。

## 2. 启动服务

### 2.1 启动 API 服务

在 `sglang-omni` 目录下运行：

```bash
sgl-omni serve --config examples/full_duplex/minicpmo.yaml --model-path ./MiniCPM-o-4_5 --enable-realtime --port 18260
```

**参数说明：**
- `--config`：全双工流水线配置，`examples/full_duplex/minicpmo.yaml`
- `--model-path`：模型路径，覆盖 YAML 中的 `model_path`
- `--enable-realtime`：挂载 WebSocket 端点 `/v1/realtime`，全双工必须加
- `--port`：服务端口，默认 8000；这里用 18260，与第 4 节的 demo 默认值一致

检查服务是否就绪：

```bash
curl -s http://127.0.0.1:18260/health
curl -s http://127.0.0.1:18260/v1/realtime/capabilities
```

每次启动后的第一个会话要编译 kernel，会慢几十秒，正式使用前先跑一个预热会话。

### 2.2 常用配置

复制一份 YAML 按需修改，例如 48 GB 显卡：

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

其他常用字段：

- `reference_audio`：参考音色 WAV，默认为 checkpoint 自带的 `assets/HT_ref_audio.wav`，服务上所有会话共用
- `max_sessions`：同时在线的会话数（默认 2），超出的连接收到 HTTP 503
- `stages.thinker.engine.context_length`：每个会话的上下文长度（默认 8192，最大 40960）

## 3. 调用服务

### 3.1 全双工语音

协议流程：连接 `ws://127.0.0.1:18260/v1/realtime`，发 `session.update`（可带 `instructions` 作为 system prompt），然后按实时节奏发送 `input_audio_buffer.append`（base64 编码的 16 kHz 单声道 PCM16，80 ms 一包，带 `sglang: {seq, t_start_ms}`），同时接收 `response.output_audio.delta`（24 kHz PCM16）和 `response.output_audio_transcript.delta`（文字）。输入结束时发 `sglang.input_audio.end`，等到 `sglang.input_audio.drained` 后发 `session.close`。

下面的脚本把一个 WAV 文件实时推给服务，打印文字并保存回复音频，只依赖 `websockets`：

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

保存为 `duplex_client.py` 后运行：

```bash
ffmpeg -i question.mp3 -ac 1 -ar 16000 -c:a pcm_s16le input.wav
python duplex_client.py --input input.wav --output reply.wav
```

脚本会在输入末尾补 4 秒静音：模型只在听到静音后才回答。

### 3.2 带图像的全双工

当 `session.updated` 中 `session.sglang.granted.input_modalities` 包含 `"image"` 时，每个 1 秒 unit 可以附一帧画面：发送 `sglang.input_image.append`，`image` 为 base64 编码的 JPEG 或 PNG（不带 `data:` 前缀，不超过 512 KiB），`sglang.t_ms` 为该帧的媒体时间，帧归属 unit `floor(t_ms / 1000)`，要在补齐该 unit 的音频之前发出。上面的脚本在每个 unit 的第一个音频包之前发送对应的帧。

从视频中抽出音频和每秒一帧（短边 448 像素，示例假设横屏视频）：

```bash
ffmpeg -i clip.mp4 -ac 1 -ar 16000 -c:a pcm_s16le input.wav
mkdir -p frames
ffmpeg -i clip.mp4 -vf "fps=1,scale=-2:448" -q:v 4 frames/%04d.jpg
python duplex_client.py --input input.wav --frames frames --output reply.wav
```

每帧给 thinker 增加 66 个 token。默认 8192 上下文下，每秒一帧的会话大约 100 秒就会用完上下文，服务端发出 `context_exhausted` 错误并关闭会话；需要更长的视频对话时，把 `context_length` 调到 32768（约 7 分钟），并相应调大 thinker 的 `kv_cache_bytes`（每会话每 token 144 KiB）。

## 4. Web Demo

### 4.1 SGLang-Omni 自带网页 demo

```bash
python tools/realtime_web_demo/serve.py --host 127.0.0.1 --port 8080
```

打开 `http://127.0.0.1:8080/`，WebSocket 地址填 `ws://localhost:18260/v1/realtime`，连接后打开麦克风；服务端支持图像输入时 Camera 按钮可用。浏览器只在 `localhost` 或 HTTPS 下允许麦克风和摄像头，服务在远端时用 SSH 转发端口：

```bash
ssh -L 8080:127.0.0.1:8080 -L 18260:127.0.0.1:18260 <gpu-host>
```

### 4.2 官方 WebRTC Demo

[WebRTC Demo](../../demo/web_demo/WebRTC_Demo/README_zh.md) 可以用 SGLang-Omni 替代 `llama-server` 作为推理服务：一个 bridge 进程以 `o45-cpp` 服务身份注册到 demo 后端，前端无需改动。部署步骤见 [`sglang_omni_backend/README.md`](../../demo/web_demo/WebRTC_Demo/sglang_omni_backend/README.md)。bridge 默认连接 `http://127.0.0.1:18260`，一次处理一通电话，加 `--forward-images` 时转发摄像头画面。

## 5. 注意事项

1. **显存**：比 H200 小的卡必须设置 thinker 和 talker 的 `kv_cache_bytes`，否则 thinker KV 池会小到无法正常对话。
2. **并发**：单张 H200 上 2 个会话流畅（超过 1 秒期限的 unit 0.45 %），4 个会话偶有卡顿（1.5 %），8 个会话明显卡顿（22 %）。需要更多并发时多起几个服务。
3. **会话时长**：没有滑动窗口，整段对话都留在上下文中。默认 8192 上下文下纯音频约 9.5 分钟，每秒一帧约 100 秒；用完后服务端发 `context_exhausted` 并关闭会话。
4. **解码与人设**：服务端固定为贪心解码、重复惩罚 1.05、会话开头强制听 3 个 unit，与官方参考一致，不能按会话修改。`instructions` 要在第一个 `session.update` 中给出，之后不能更改；音色由服务端的 `reference_audio` 决定。
5. **打断**：协议没有取消回复的操作，是否停止说话由模型自己决定。用户插话后模型中位数再说 1 秒左右让出话轮；"嗯""对"这类简短插话通常被当作附和，模型会继续说。回复一般在用户停止说话后 1.5–2 秒开始。
6. **音频格式**：输入必须是 16 kHz 单声道 PCM16，`sglang.seq` 从 0 连续递增，`t_start_ms` 等于已发送音频的时长；输出为 24 kHz。
7. **预热**：每次重启后的第一个会话要编译 kernel，会慢几十秒。
8. **范围**：本页只介绍 SGLang-Omni 的全双工服务。用 SGLang 做单轮图文问答（`python -m sglang.launch_server` 加 OpenAI 兼容接口）请参考 [SGLang 文档](https://docs.sglang.ai/)。
