#!/usr/bin/env bash
set -euo pipefail

PORT="${RWKV_VLLM_PORT:-8120}"
RUNTIME_DIR="${RWKV_VLLM_RUNTIME_DIR:-${XDG_RUNTIME_DIR:-/tmp}/rwkv-vllm}"
PID_FILE="${RWKV_VLLM_PID_FILE:-${RUNTIME_DIR}/server-${PORT}.pid}"

if [[ ! -s "${PID_FILE}" ]]; then
  echo "RWKV vLLM PID file not found: ${PID_FILE}"
  exit 0
fi
pid="$(cat "${PID_FILE}")"
if [[ ! "${pid}" =~ ^[0-9]+$ ]]; then
  echo "Invalid RWKV vLLM PID file: ${PID_FILE}" >&2
  exit 1
fi
if ! kill -0 "${pid}" 2>/dev/null; then
  rm -f "${PID_FILE}"
  echo "RWKV vLLM is not running"
  exit 0
fi
command_line="$(ps -o command= -p "${pid}")"
if [[ "${command_line}" != *"vllm.entrypoints.openai.api_server"* ]]; then
  echo "PID ${pid} is not a vLLM API server; refusing to signal it" >&2
  exit 1
fi

kill -TERM -- "-${pid}"
for _ in {1..100}; do
  if ! kill -0 "${pid}" 2>/dev/null; then
    rm -f "${PID_FILE}"
    echo "RWKV vLLM stopped"
    exit 0
  fi
  sleep 0.1
done
kill -KILL -- "-${pid}" 2>/dev/null || true
rm -f "${PID_FILE}"
echo "RWKV vLLM force-stopped after timeout"

