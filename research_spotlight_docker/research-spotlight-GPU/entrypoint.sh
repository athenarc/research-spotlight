#!/bin/sh
set -e

APP_URL="${APP_URL:-http://localhost:8080}"

# Start the server in the background so this script can watch model loading.
"$@" &
SERVER_PID=$!

trap 'kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null' TERM INT

echo "Loading models. This can take a few minutes on first start..."

while true; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Server stopped during startup."
    exit 1
  fi

  STATUS=$(curl -s -m 5 http://127.0.0.1:8000/api/models || true)

  if echo "$STATUS" | grep -q '"status": *"ready"'; then
    break
  fi

  if echo "$STATUS" | grep -q '"status": *"error"'; then
    echo "Model loading FAILED: $STATUS"
    echo "The server is still running but runs are disabled. See the [startup] lines above."
    # Stay up instead of exiting: exiting would make 'restart: unless-stopped' reload everything in a loop.
    wait "$SERVER_PID"
    exit $?
  fi

  sleep 2
done

echo "All models are loaded. Go to ${APP_URL}"

wait "$SERVER_PID"
