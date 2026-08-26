# GPU environment probe — 2026-08-26

This is an environment-availability record, **not** a StatePool benchmark.

## WZU_4080

- host: `wzu-System-4080GPU`
- GPU: NVIDIA GeForce RTX 4080, 16,376 MiB
- driver: 595.84
- no RWKV `.pth`/`.safetensors` checkpoint was found in the probed model paths

## WZU_Server

- host: `wzu-SYS-4029GP-TRT`
- GPUs: 2 × Tesla V100-PCIE-32GB, driver 580.173.02
- exact candidate checkpoint:
  `/home/data/wangyue/models/rwkv7-g1i-preview3260/rwkv7-g1i_preview3260-7.2b-20260716-ctx12288.pth`
  (14,400,007,869 bytes)
- compatible Albatross runtime candidate:
  `/home/data/wangyue/repos/home-top-level/Albatross-ee3308f-v100/faster3a_2607/`
- both GPUs reported 99% utilization during the probe, so no competing model
  load or inference benchmark was started

The next evidence gate is a scheduled idle-window run of two compatible Worker
processes, the live lifecycle driver, and raw command/GPU telemetry capture.
