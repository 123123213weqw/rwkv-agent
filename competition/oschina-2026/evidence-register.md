# 证据台账

| ID | 能力/结论 | 类型 | 当前状态 | 证据 |
|---|---|---|---|---|
| E01 | 插件关闭时不构造 HTTP Client并走原路径 | 自动测试 | 已验证 | `cloud_plugin::tests::disabled_plugin_never_builds_http_and_returns_original_local_path` |
| E02 | 完整 Workspace Rust 检查 | 远程命令 | 已验证 | WZU_Server `cargo check --workspace --all-targets`，2026-08-26 |
| E03 | Runtime 回归 65 + mock full path 13 | 远程测试 | 已验证 | `/tmp/rwkv-statepool-remote-test-2.log`（本机临时日志） |
| E04 | StatePool 8 项测试和 wire 3 项测试 | 远程测试 | 已验证 | 同 E03 |
| E05 | Snapshot/Restore bytes、checksum、CAS、stale fence | 自动测试 | 已验证 | `snapshot_restore_round_trip_uses_cas_checksum_and_fencing` |
| E06 | 两个容器目标 release build、live/metrics/help smoke | 远程容器 | 已验证 | `evidence/statepool/remote-container-smoke-2026-08-26.md` |
| E07 | JSON Schema/OpenAPI fixtures | 静态验证 | 已验证 | `scripts/check_statepool_contracts.py` |
| E08 | Compose/Helm/KEDA/Dashboard 结构 | 静态验证 | 已验证 | `scripts/check_statepool_deploy.py`、Helm 3.19 lint/template |
| E09 | 既有 RWKV AMD/V100/100-State 结果 | 历史实测 | 可复用，非云结果 | 主 README 与原 `bench/` evidence |
| E10 | live RWKV State export/import | GPU 实测 | 待完成 | — |
| E11 | Worker kill→compatible restore→continue | GPU 故障实验 | 待完成 | — |
| E12 | PostgreSQL distributed Lease/CAS | 双连接真实 17.6 容器集成测试 | 已验证 | `evidence/statepool/durable-adapters-2026-08-27.md` |
| E13 | S3/MinIO immutable Cold State | 真实 MinIO 容器集成测试 | 已验证 | `evidence/statepool/durable-adapters-2026-08-27.md` |
| E14 | KEDA 0→1→N→0 safe drain | Kubernetes/GPU 实测 | 待完成 | — |
| E15 | A/B/C 成本和利用率 | 对照实验 | 待完成 | — |
| E16 | Albatross CPU snapshot→release→restore→continue | 自动测试 | 已验证 | `tests/test_state_runtime.py::test_safe_snapshot_release_restore_continue_roundtrip`、`crates/state-runtime/tests/rwkv_http.rs` |
| E17 | V100/4080 远程环境与模型可用性探测 | 环境清单，非性能证据 | 已记录 | `evidence/statepool/gpu-environment-probe-2026-08-26.md` |
| E18 | Cloud Lite PostgreSQL+MinIO 跨插件重启恢复 | 真实 Compose 服务闭环 | 已验证（通用 State bytes，非 GPU） | `evidence/statepool/cloud-lite-compose-restart-2026-08-27.txt` |

规则：临时 `/tmp` 日志不算发布证据；正式提交前需要把去敏后的完整输出放入
`bench/artifacts/`，填写 commit 和 checksum。
