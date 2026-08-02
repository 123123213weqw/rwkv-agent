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

新的来源策略应基于通用页面/来源特征，不应添加行业专用路由表。

面向用户的改动还必须更新对应Quickstart、配置模板、CHANGELOG和Known Issues。不得把内部服务器生命周期、
本机绝对路径或实验密钥做成公开默认值。
