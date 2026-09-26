# Host settings for deploy/start.sh and deploy/stop.sh. Every value can also be set in the environment;
# a host-specific file deploy/env.local.sh (not shipped; plain VAR=value lines) is sourced first if present.
: "${DEMO_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # holds bridge/, deploy/, *-venv/, logs/
[ -f "$DEMO_ROOT/deploy/env.local.sh" ] && . "$DEMO_ROOT/deploy/env.local.sh"
: "${WEBRTC_DEMO_DIR:=$(cd "$DEMO_ROOT/.." && pwd)}"   # the WebRTC_Demo checkout (omini_backend_code/, o45-frontend/)
: "${CTR:=docker}"                     # container CLI for LiveKit: docker | podman
: "${LK_NET:=publish}"                 # publish: bridge network, ports published on 127.0.0.1 (docker-proxy / rootlessport relays media)
                                       # host: --network host, signalling on 127.0.0.1, ICE on lo only. EXPERIMENTAL: Chromium offers no loopback ICE candidates, calls did not connect in our tests
: "${LK_NAME:=webrtc-demo-livekit-$( [ "$LK_NET" = host ] && echo host || echo lo )}"   # one container per mode; recreate it after changing ports
: "${LK_IMAGE:=docker.io/livekit/livekit-server:v1.5.3}"
: "${LK_PORT:=7880}"                   # signalling (HTTP/WS)
: "${LK_TCP_PORT:=7881}"               # ICE-TCP (what an ssh -L tunnel carries); 0 = off
: "${LK_UDP_START:=50000}"
: "${LK_UDP_END:=50100}"
: "${LK_KEY:=devkey}"
: "${LK_SECRET:=secretsecretsecretsecretsecretsecret}"
: "${BACKEND_PORT:=8021}"
: "${PAGE_PORT:=8088}"                 # duplex build (video tab)
: "${PAGE_SIMPLEX_PORT:=8089}"         # simplex build (voice tab), if dist-simplex exists
: "${BRIDGE_PORT:=18270}"              # control port is BRIDGE_PORT+1 (backend contract)
: "${UPSTREAM:=http://127.0.0.1:18260}"   # sglang-omni server (not the public gate)
: "${GATE_STATUS_URL:=http://127.0.0.1:18299/status}"   # empty = no gate check
: "${BACKEND_PY:=$DEMO_ROOT/backend-venv/bin/python}"
: "${BRIDGE_PY:=$DEMO_ROOT/bridge-venv/bin/python}"
: "${NODE:=node}"
: "${OMNI_PREFILL_CHUNK_MS:=80}"       # patched backend: prefill chunk length (upstream 1000)
: "${OMNI_DUPLEX_KEEP_OPEN:=1}"        # patched backend: one kept-open duplex generate (0 = upstream per-chunk generates)
: "${LOOPWATCH_OUT:=}"                 # set to a file to record backend event-loop stalls (probe/loopwatch)
export OMNI_PREFILL_CHUNK_MS OMNI_DUPLEX_KEEP_OPEN
