#!/bin/bash
# Start LiveKit (container, loopback only), the demo backend (latency-patched), the page servers and the bridge.
# Leaves the sglang-omni server, the public gate and our own page alone. Settings: deploy/env.sh (+ env / env.local.sh).
# Extra arguments go to the bridge.
set -u
. "$(dirname "$0")/env.sh"
cd "$DEMO_ROOT"; mkdir -p logs run
up() { curl -s -o /dev/null --max-time 2 "$1"; }

# LiveKit config rendered from the variables above
cat > run/livekit.yaml <<YAML
port: $LK_PORT
bind_addresses: ["$( [ "$LK_NET" = host ] && echo 127.0.0.1 || echo 0.0.0.0 )"]
log_level: info
rtc:
  tcp_port: $LK_TCP_PORT
  port_range_start: $LK_UDP_START
  port_range_end: $LK_UDP_END
  use_external_ip: false
  node_ip: 127.0.0.1
$( [ "$LK_NET" = host ] && printf '  enable_loopback_candidate: true\n  interfaces:\n    includes: ["lo"]\n  ips:\n    includes: ["127.0.0.1/32"]\n' )
room:
  max_participants: 10
  empty_timeout: 300
keys:
  $LK_KEY: $LK_SECRET
turn:
  enabled: false
YAML
if ! $CTR ps --format '{{.Names}}' | grep -qx "$LK_NAME"; then
  if $CTR container inspect "$LK_NAME" >/dev/null 2>&1; then
    $CTR start "$LK_NAME" >/dev/null
  elif [ "$LK_NET" = host ]; then
    $CTR run -d --name "$LK_NAME" --restart unless-stopped --network host \
      -v "$DEMO_ROOT/run/livekit.yaml:/livekit.yaml:ro" "$LK_IMAGE" --config /livekit.yaml >/dev/null
  else
    tcp=(); [ "$LK_TCP_PORT" != 0 ] && tcp=(-p "127.0.0.1:$LK_TCP_PORT:$LK_TCP_PORT")
    $CTR run -d --name "$LK_NAME" --restart unless-stopped \
      -p "127.0.0.1:$LK_PORT:$LK_PORT" "${tcp[@]}" -p "127.0.0.1:$LK_UDP_START-$LK_UDP_END:$LK_UDP_START-$LK_UDP_END/udp" \
      -v "$DEMO_ROOT/run/livekit.yaml:/livekit.yaml:ro" "$LK_IMAGE" --config /livekit.yaml >/dev/null
  fi
fi
for i in $(seq 20); do up "http://127.0.0.1:$LK_PORT" && break; sleep 0.5; done
echo "livekit ($CTR, $LK_NET): $(curl -s -o /dev/null -w %{http_code} "http://127.0.0.1:$LK_PORT")"

if ! up "http://127.0.0.1:$BACKEND_PORT/health"; then
  # CUDA hidden: livekit 1.1.18 aborts in the NVIDIA video decoder when the page's camera track is subscribed.
  ( cd $WEBRTC_DEMO_DIR/omini_backend_code/code && exec env APP_ENV=local NUMBA_CACHE_DIR=/tmp/numba_cache WORKERS=1 PYTHONFAULTHANDLER=1 CUDA_VISIBLE_DEVICES="" \
      SERVER__HOST=127.0.0.1 SERVER__PORT="$BACKEND_PORT" LIVEKIT__URL="ws://127.0.0.1:$LK_PORT" LIVEKIT__API_KEY="$LK_KEY" LIVEKIT__API_SECRET="$LK_SECRET" \
      ${LOOPWATCH_OUT:+PYTHONPATH=$DEMO_ROOT/probe/loopwatch LOOPWATCH_OUT=$LOOPWATCH_OUT} \
      setsid nohup "$BACKEND_PY" main.py > "$DEMO_ROOT/logs/backend.log" 2>&1 < /dev/null ) &
  echo $! > run/backend.pid
  for i in $(seq 60); do up "http://127.0.0.1:$BACKEND_PORT/health" && break; sleep 1; done
fi
echo "backend: $(curl -s "http://127.0.0.1:$BACKEND_PORT/health")"

page() {  # port dist logname
  [ -d "$2" ] || return 0
  if ! up "http://127.0.0.1:$1/"; then
    setsid nohup "$NODE" deploy/serve_http.mjs --dist "$2" --port "$1" --backend "$BACKEND_PORT" --livekit "$LK_PORT" > "logs/$3.log" 2>&1 < /dev/null &
    echo $! > "run/$3.pid"; sleep 1
  fi
  echo "$3 (:$1): $(curl -s -o /dev/null -w %{http_code} "http://127.0.0.1:$1/")"
}
page "$PAGE_PORT" $WEBRTC_DEMO_DIR/o45-frontend/dist frontend
page "$PAGE_SIMPLEX_PORT" $WEBRTC_DEMO_DIR/o45-frontend/dist-simplex frontend-simplex

if ! up "http://127.0.0.1:$((BRIDGE_PORT + 1))/health"; then
  setsid nohup "$BRIDGE_PY" -m bridge --port "$BRIDGE_PORT" --upstream "$UPSTREAM" --register-url "http://127.0.0.1:$BACKEND_PORT" \
    ${GATE_STATUS_URL:+--gate-status-url "$GATE_STATUS_URL"} --livekit-url "http://127.0.0.1:$LK_PORT" \
    --livekit-key "$LK_KEY" --livekit-secret "$LK_SECRET" "$@" > logs/bridge.log 2>&1 < /dev/null &
  echo $! > run/bridge.pid
  sleep 3
fi
echo "bridge: $(curl -s "http://127.0.0.1:$((BRIDGE_PORT + 1))/health")"
echo "services: $(curl -s "http://127.0.0.1:$BACKEND_PORT/api/inference/services")"
