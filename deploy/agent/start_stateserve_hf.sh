#!/usr/bin/env bash
set -euo pipefail

: "${G1I_HF_MODEL_PATH:?set G1I_HF_MODEL_PATH to a local RWKV HF model directory}"
: "${G1I_MODEL_ID:?set G1I_MODEL_ID}"
: "${G1I_MODEL_REVISION:?set G1I_MODEL_REVISION to an immutable revision}"

export G1I_BACKEND=hf_recurrent
export G1I_TOKENIZER_ID="${G1I_TOKENIZER_ID:-rwkv_vocab_v20230424}"
export G1I_CONTEXT="${G1I_CONTEXT:-8192}"
export G1I_STATE_CAPACITY="${G1I_STATE_CAPACITY:-16}"
export G1I_MAX_BATCH_SIZE="${G1I_MAX_BATCH_SIZE:-8}"
export G1I_PREFILL_CHUNK_SIZE="${G1I_PREFILL_CHUNK_SIZE:-64}"

exec rwkv-stateserve \
  --host "${RWKV_STATESERVE_HOST:-127.0.0.1}" \
  --port "${RWKV_STATESERVE_PORT:-18118}"
