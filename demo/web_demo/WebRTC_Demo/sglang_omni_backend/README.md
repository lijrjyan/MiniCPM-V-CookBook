# Serving the WebRTC demo with sglang-omni

This directory lets the WebRTC demo (LiveKit + FastAPI backend + Vue frontend) use an
[sglang-omni](https://github.com/sgl-project/sglang-omni) MiniCPM-o 4.5 native full-duplex server
as its inference service instead of `llama-server`. Nothing in the frontend changes; the backend
gets a small latency patch (already applied on this branch) and a bridge process registers itself
as the `o45-cpp` inference service.

```
browser ──WebRTC──▶ LiveKit ──▶ backend (FastAPI) ──HTTP/SSE──▶ bridge ──/v1/realtime (WebSocket)──▶ sglang-omni
```

## Components

- `bridge/` — Python 3.12, only `aiohttp` and `websockets`. Registers with the backend
  (`POST /api/inference/register`), serves the inference contract (`init_sys_prompt`,
  `streaming_prefill`, `streaming_generate` as SSE, `/omni/break`, `/omni/stop`, `/health`), and
  keeps one `/v1/realtime` session per call: 80 ms PCM16 packets in, 24 kHz audio and transcript
  deltas out. One call at a time. `--forward-images` forwards camera frames as
  `sglang.input_image.append` (needs a server that grants `input_modalities: [audio, image]`).
- `deploy/` — `env.sh` (all ports, paths, `CTR=docker|podman`), `start.sh` / `stop.sh` (LiveKit
  container, backend, both page builds, bridge), `serve_http.mjs` (static page server that proxies
  the API and LiveKit signalling).
- Backend changes on this branch (`omini_backend_code/code`): 80 ms prefill chunks
  (`OMNI_PREFILL_CHUNK_MS`, default 80; 1000 restores upstream), one long-lived duplex
  `streaming_generate` (`OMNI_DUPLEX_KEEP_OPEN=1`), no hold on the first reply chunk, streaming
  resampler with filter state across chunks (`stream_resample.py`), environment variables
  overriding the YAML config, per-chunk logs at debug level.

## Run

1. Start the sglang-omni server (see its `examples/full_duplex/README.md`), e.g. on
   `127.0.0.1:18260` with `--enable-realtime`.
2. Build the frontend once (`o45-frontend`, `pnpm install && pnpm build`; a second build with the
   voice tab enabled can be placed in `dist-simplex`) and create the backend venv
   (`omini_backend_code/requirements.txt`) and the bridge venv (`pip install aiohttp websockets`)
   under this directory as `backend-venv/` and `bridge-venv/`.
3. Optionally write `deploy/env.local.sh` (plain `VAR=value` lines) to override ports, the
   container CLI or `UPSTREAM`.
4. `./deploy/start.sh`, then open `http://127.0.0.1:8088/` (duplex) or `:8089` (simplex). Over
   ssh, forward `8088`, `8089` and `7881` (LiveKit advertises `127.0.0.1:7881` for ICE-TCP).
5. `./deploy/stop.sh` stops everything except the sglang-omni server.

## Notes

- The bridge cannot cancel a reply (the realtime protocol has no cancel); `/omni/break` mutes the
  reply in flight.
- The backend keeps the service locked for 15–20 s after hang-up; a call in that window fails.
- Media is relayed by the container runtime's port publisher (docker-proxy / rootlessport); on a
  loaded host this adds seconds. `LK_NET=host` is experimental.
