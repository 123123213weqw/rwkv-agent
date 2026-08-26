#!/usr/bin/env bash
set -euo pipefail
role="${1:?source or target}"
ROOT=/home/wzu/codex-build/rwkv-agent
RUN=/home/wzu/codex-run/statepool-4080-worker-kill-20260827
PY=/home/wzu/anaconda3/bin/python
MODEL=/home/wzu/wangyue/models/rwkv7-g1i-20260805/rwkv7-g1i-1.5b-20260805-ctx16384.pth
RUNTIME=/home/wzu/codex-build/albatross-fullmodel-ab-20260825/candidate
exec env \
  CUDA_VISIBLE_DEVICES=0 \
  CUDA_HOME="$RUN/cuda14" \
  TORCH_CUDA_ARCH_LIST=8.9 \
  PATH=/home/wzu/anaconda3/bin:"$RUN/cuda14/bin":/usr/local/bin:/usr/bin:/bin \
  LD_LIBRARY_PATH=/home/wzu/anaconda3/lib/python3.12/site-packages/nvidia/cu13/lib:/home/wzu/anaconda3/lib \
  PYTHONPATH="$ROOT/src:$ROOT/vendor-python" \
  G1I_MODEL_PATH="$MODEL" \
  G1I_RUNTIME_DIR="$RUNTIME" \
  G1I_MODEL_ID=rwkv7-g1i-20260805-1.5b \
  G1I_MODEL_REVISION=g1i-1.5b-20260805-ctx16384 \
  G1I_TOKENIZER_ID=rwkv_vocab_v20230424 \
  G1I_STATE_ABI=rwkv7-albatross-fp32io16-state-v1 \
  G1I_CONTEXT=16384 \
  G1I_STATE_CAPACITY=4 \
  G1I_MAX_BATCH_SIZE=1 \
  G1I_PERSISTENT_STATE_CAPACITY=2 \
  G1I_MAX_SNAPSHOT_BYTES=536870912 \
  RWKV_STATEPOOL_URL=http://127.0.0.1:18131 \
  RWKV_WORKER_ID="worker-4080-${role}" \
  RWKV_WORKER_ENDPOINT=http://127.0.0.1:18218 \
  RWKV_WORKER_ZONE=cloud \
  RWKV_WORKER_DEVICE_VENDOR=nvidia \
  RWKV_WORKER_DEVICE_MODEL=RTX_4080 \
  RWKV_WORKER_DEVICE_RUNTIME=cuda13.0 \
  RWKV_WORKER_DEVICE_MEMORY_BYTES=17171480576 \
  RWKV_WORKER_PRICE_CURRENCY=CNY \
  RWKV_WORKER_PRICE_PER_GPU_HOUR=2.0 \
  "$PY" -m rwkv_agent.sidecar --host 127.0.0.1 --port 18218
