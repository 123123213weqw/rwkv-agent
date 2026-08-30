#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${RWKV_VLLM_PYTHON:-python}"
MODEL_PATH="${RWKV_VLLM_MODEL_PATH:?set RWKV_VLLM_MODEL_PATH}"
HOST="${RWKV_VLLM_HOST:-127.0.0.1}"
PORT="${RWKV_VLLM_PORT:-8120}"
MODEL_NAME="${RWKV_VLLM_MODEL_NAME:-rwkv7-vllm}"
MAX_MODEL_LEN="${RWKV_VLLM_MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${RWKV_VLLM_MAX_NUM_SEQS:-8}"
MAX_BATCHED_TOKENS="${RWKV_VLLM_MAX_BATCHED_TOKENS:-4096}"
GPU_MEMORY_UTILIZATION="${RWKV_VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
CHAT_TEMPLATE="${RWKV_VLLM_CHAT_TEMPLATE:-${ROOT_DIR}/integrations/vllm_rwkv7/src/vllm_rwkv7/rwkv_chat.jinja}"
RUNTIME_DIR="${RWKV_VLLM_RUNTIME_DIR:-${XDG_RUNTIME_DIR:-/tmp}/rwkv-vllm}"
PID_FILE="${RWKV_VLLM_PID_FILE:-${RUNTIME_DIR}/server-${PORT}.pid}"

export VLLM_USE_V1=0
export VLLM_PLUGINS=rwkv7

mkdir -p "${RUNTIME_DIR}"
if [[ -s "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}")"
  if [[ "${old_pid}" =~ ^[0-9]+$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "RWKV vLLM is already running with PID ${old_pid}" >&2
    exit 1
  fi
fi

SERVER_PID=""
cleanup() {
  trap - EXIT HUP INT TERM
  if [[ "${SERVER_PID}" =~ ^[0-9]+$ ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
    for _ in {1..50}; do
      kill -0 "${SERVER_PID}" 2>/dev/null || break
      sleep 0.1
    done
    kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
  fi
  rm -f "${PID_FILE}"
}
trap cleanup EXIT HUP INT TERM

setsid "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --trust-remote-code \
  --enforce-eager \
  --dtype half \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --disable-log-requests \
  --chat-template "${CHAT_TEMPLATE}" \
  "$@" &
SERVER_PID="$!"
printf '%s\n' "${SERVER_PID}" >"${PID_FILE}"
wait "${SERVER_PID}"

