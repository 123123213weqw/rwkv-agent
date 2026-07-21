# Contributing

1. 为问题添加最小可复现用例。
2. 检索质量改动必须在同一 `bench/realtime_web_retrieval.jsonl`、同一配置和同一指标上与冻结基线比较。
3. 不把模型回答质量与 URL Discovery、抓取或 Evidence 指标混在一个数字里。
4. 不提交模型权重、网页正文、调试 Token Trace、服务器地址、密钥或本地绝对路径。
5. 提交前运行：

```bash
PYTHONPATH=src python -m unittest discover -s tests
ruff check src bench tests
```

新的来源策略应基于通用页面/来源特征，不应添加行业专用路由表。
