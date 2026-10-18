# Contributing

1. 为问题添加最小可复现用例。
2. 检索质量改动必须在同一 `bench/realtime_web_retrieval.jsonl`、同一配置和同一指标上与冻结基线比较。
3. 不把模型回答质量与 URL Discovery、抓取或 Evidence 指标混在一个数字里。
4. 不提交模型权重、网页正文、调试 Token Trace、服务器地址、密钥或本地绝对路径。
5. 提交前运行：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src:. python -m pytest -q
ruff check src bench benchmarks tests scripts
python scripts/check_public_release.py

cargo fmt --check
cargo test --locked
cargo clippy --all-targets -- -D warnings
```

StatePool 相关变更还必须运行：

```bash
uv run --with jsonschema python scripts/check_statepool_contracts.py
uv run --with pyyaml python scripts/check_statepool_deploy.py
docker compose -f deploy/statepool/compose.yaml config --quiet
helm lint deploy/statepool/helm/statepool
helm template demo deploy/statepool/helm/statepool >/dev/null
```

协议字段变更需要同步 Rust wire type、JSON Schema、OpenAPI、example、ADR/兼容
说明。插件关闭回归是合并硬门槛。集成 Kubernetes、KEDA、HAMi、AIBrix、
PostgreSQL、S3、Prometheus/Grafana 时优先提交 Adapter、values、manifest 或
dashboard，不复制上游源码、不建立无必要 Fork。

性能 PR 必须明确区分 `measured` 与 `estimated`，包含 commit、硬件、软件版本、
命令和原始日志；不得把“已有配置”写成“运行验证”。

新的来源策略应基于通用页面/来源特征，不应添加行业专用路由表。

面向用户的改动还必须更新对应Quickstart、配置模板、CHANGELOG和Known Issues。不得把内部服务器生命周期、
本机绝对路径或实验密钥做成公开默认值。
