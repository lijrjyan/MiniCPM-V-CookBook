"""Run the bridge: ``python -m bridge --register-url http://127.0.0.1:8021``."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal

from aiohttp import web

from .app import Bridge


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1", help="bind address for both ports")
    p.add_argument("--port", type=int, default=18270, help="main port; control port is port+1 (backend contract)")
    p.add_argument("--advertise-ip", default="127.0.0.1", help="ip registered with the backend")
    p.add_argument("--upstream", default="http://127.0.0.1:18260", help="sglang-omni server (NOT the public gate)")
    p.add_argument("--register-url", default="", help="backend base url, e.g. http://127.0.0.1:8021; empty = no registration")
    p.add_argument("--register-interval-s", type=float, default=10.0)
    p.add_argument("--service-name", default="o45-cpp", help="the frontend hard-codes serviceName='o45-cpp'")
    p.add_argument("--model-type", default="release", help="'release' matches both simplex and duplex logins")
    p.add_argument("--session-type", default="release")
    p.add_argument("--instructions", default=None, help="optional session.update instructions (default: server default)")
    p.add_argument("--forward-images", action="store_true", help="forward camera frames as sglang.input_image.append (needs PR #350 server)")
    p.add_argument("--silence-grace-ms", type=float, default=1800.0, help="start filling silence this long after the last real audio chunk arrived (chunks come every ~1000 ms)")
    p.add_argument("--silence-grace-small-ms", type=float, default=500.0, help="same, when the backend sends short chunks (< 500 ms, the latency-patched backend)")
    p.add_argument("--no-unit-align", action="store_true", help="do not pad the timeline so backend 1 s chunks start on model-unit boundaries (short chunks are never padded)")
    p.add_argument("--duplex-window-s", type=float, default=1.0, help="duplex: end an idle generate stream after this long")
    p.add_argument("--keep-open-window-s", type=float, default=20.0, help="duplex with keep_open (patched backend): end an idle generate stream after this long")
    p.add_argument("--duplex-max-s", type=float, default=30.0)
    p.add_argument("--simplex-wait-s", type=float, default=8.0, help="simplex: give up if no response starts within this")
    p.add_argument("--simplex-max-s", type=float, default=120.0)
    p.add_argument("--idle-timeout-s", type=float, default=30.0, help="close the upstream session if the backend goes quiet and no longer holds the service lock")
    p.add_argument("--idle-cap-s", type=float, default=300.0, help="close after this much backend silence even if the lock is still held")
    p.add_argument("--gate-status-url", default="", help="e.g. http://127.0.0.1:18299/status: refuse init while the public gate is busy")
    p.add_argument("--livekit-url", default="", help="e.g. http://127.0.0.1:7880: hold the init reply until the page is ACTIVE in the room")
    p.add_argument("--livekit-key", default="devkey")
    p.add_argument("--livekit-secret", default="secretsecretsecretsecretsecretsecret")
    p.add_argument("--page-ready-timeout-s", type=float, default=20.0)
    p.add_argument("--page-ready-margin-s", type=float, default=0.5)
    p.add_argument("--init-delay-s", type=float, default=5.0, help="without --livekit-url (or on timeout): hold the init reply this long instead")
    p.add_argument("--dump-dir", default="", help="debug: write each session's input timeline, output audio and prefill arrivals here")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    args.upstream_http = args.upstream.rstrip("/")
    args.upstream_ws = args.upstream_http.replace("http://", "ws://", 1).replace("https://", "wss://", 1) + "/v1/realtime"
    args.register_url = args.register_url.rstrip("/")
    return args


async def run(args: argparse.Namespace) -> None:
    bridge = Bridge(args)
    runners = []
    for app, port in ((bridge.main_app(), args.port), (bridge.control_app(), args.port + 1)):
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        await web.TCPSite(runner, args.host, port).start()
        runners.append(runner)
    await bridge.start_background()
    logging.getLogger("bridge").info(
        "bridge on %s:%d (+%d control) -> %s, register=%s", args.host, args.port, args.port + 1, args.upstream_ws, args.register_url or "off"
    )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()
    await bridge.shutdown()
    for runner in runners:
        await runner.cleanup()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
