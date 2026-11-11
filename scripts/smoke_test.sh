#!/usr/bin/env bash
# End-to-end test of a built image: start it the way production would, check that it answers,
# and check that it shuts down cleanly. Used by CI and by `make smoke`.
#
#   ./scripts/smoke_test.sh ssv2:local
set -euo pipefail

IMAGE="${1:?usage: smoke_test.sh <image>}"
# Not 8000 on purpose: proves the container honours $PORT, which many hosts inject.
CONTAINER_PORT=9123
NAME="ssv2-smoke-$$"

cleanup() {
  echo "--- container logs ---"
  docker logs "$NAME" 2>&1 || true
  docker rm -f "$NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Run hardened, as production should: read-only filesystem (writes go to /tmp only), no Linux
# capabilities, no privilege escalation. If the app ever starts writing to its own filesystem,
# this test fails.
docker run -d --name "$NAME" \
  -e PORT="$CONTAINER_PORT" \
  -p "127.0.0.1::$CONTAINER_PORT" \
  --read-only --tmpfs /tmp \
  --cap-drop ALL --security-opt no-new-privileges \
  "$IMAGE" >/dev/null

# Docker picked a free host port (so this never clashes with anything already running).
HOST_PORT=$(docker port "$NAME" "$CONTAINER_PORT/tcp" | head -n1 | awk -F: '{print $NF}')
BASE="http://127.0.0.1:$HOST_PORT"

echo "Waiting for $BASE/ready ..."
for _ in $(seq 1 60); do
  if curl -fsS "$BASE/ready" >/dev/null 2>&1; then break; fi
  if [ "$(docker inspect -f '{{.State.Running}}' "$NAME")" != "true" ]; then
    echo "Container exited during startup" >&2
    exit 1
  fi
  sleep 1
done

check() { # check <path> <expected JSON field status>
  local body
  body=$(curl -fsS "$BASE$1")
  echo "GET $1 -> $body"
  python3 -c 'import json, sys; b = json.loads(sys.argv[1]); assert b["status"] == sys.argv[2], b' \
    "$body" "$2"
}
check /health ok
check /ready ready

user=$(docker exec "$NAME" id -u)
echo "Runs as uid $user"
[ "$user" != "0" ] || { echo "Container runs as root" >&2; exit 1; }

# `docker stop` sends SIGTERM and waits; exit code 0 within the timeout means uvicorn is PID 1,
# received the signal and shut down gracefully (exit 137 would mean it had to be killed).
docker stop -t 10 "$NAME" >/dev/null
code=$(docker inspect -f '{{.State.ExitCode}}' "$NAME")
echo "Exit code after SIGTERM: $code"
[ "$code" = "0" ] || { echo "Did not shut down cleanly on SIGTERM" >&2; exit 1; }

echo "Smoke test passed."
