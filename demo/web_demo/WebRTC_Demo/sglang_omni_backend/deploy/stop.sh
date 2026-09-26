#!/bin/bash
# Stop the bridged demo stack (not the sglang-omni server, gate or our page). Settings: deploy/env.sh.
. "$(dirname "$0")/env.sh"
cd "$DEMO_ROOT"
for name in bridge frontend frontend-simplex backend; do
  [ -f run/$name.pid ] && kill "$(cat run/$name.pid)" 2>/dev/null && echo "stopped $name"
  rm -f run/$name.pid
done
# processes started by an older start.sh whose pid file was wrong
pkill -f "^$BACKEND_PY main.py" && echo "stopped stray backend"
pkill -f "^$BRIDGE_PY -m bridge --port $BRIDGE_PORT " && echo "stopped stray bridge"
$CTR stop "$LK_NAME" >/dev/null 2>&1 && echo "stopped livekit ($LK_NAME)"
