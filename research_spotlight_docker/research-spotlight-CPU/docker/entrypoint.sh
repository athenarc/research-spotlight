#!/usr/bin/env bash
set -Eeuo pipefail

LLAMA_HOST="${LLAMA_HOST:-127.0.0.1}"
LLAMA_PORT="${LLAMA_PORT:-8080}"
LLAMA_HF_MODEL="${LLAMA_HF_MODEL:-ggml-org/GLM-OCR-GGUF:Q8_0}"

mkdir -p \
    /app/data/runs \
    /models \
    /cache/huggingface \
    /root/.cache


echo "[research-spotlight] Starting llama-server..."

/opt/llama/llama-server \
    --hf-repo "${LLAMA_HF_MODEL}" \
    --host "${LLAMA_HOST}" \
    --port "${LLAMA_PORT}" \
	-c 8192 \
	--cache-ram 0 \
	--no-cache-idle-slots \
	--no-cache-prompt \
	--parallel 1 \
    > /tmp/llama-server.log 2>&1 &

LLAMA_PID=$!


cleanup() {
    echo "[research-spotlight] Shutting down..."

    kill "${LLAMA_PID}" 2>/dev/null || true
    kill "${APP_PID:-}" 2>/dev/null || true
}

trap cleanup EXIT INT TERM


echo "[research-spotlight] Waiting for llama-server..."

READY=0

for _ in $(seq 1 180); do

    if curl -fsS \
        "http://${LLAMA_HOST}:${LLAMA_PORT}/health" \
        >/dev/null 2>&1
    then
        READY=1
        break
    fi

    if ! kill -0 "${LLAMA_PID}" 2>/dev/null; then
        echo "[research-spotlight] llama-server exited."

        cat /tmp/llama-server.log || true

        exit 1
    fi

    sleep 1

done


if [[ "${READY}" != "1" ]]; then

    echo "[research-spotlight] llama-server did not become ready."

    cat /tmp/llama-server.log || true

    exit 1

fi


echo "[research-spotlight] llama-server ready."
echo "[research-spotlight] Starting FastAPI..."


uvicorn \
    app.main:app \
    --host 0.0.0.0 \
    --port 8000 &

APP_PID=$!


set +e

wait -n \
    "${LLAMA_PID}" \
    "${APP_PID}"

STATUS=$?

set -e


if ! kill -0 "${LLAMA_PID}" 2>/dev/null; then

    echo "[research-spotlight] llama-server stopped."

    cat /tmp/llama-server.log || true

fi


if ! kill -0 "${APP_PID}" 2>/dev/null; then

    echo "[research-spotlight] FastAPI stopped."

fi


exit "${STATUS}"