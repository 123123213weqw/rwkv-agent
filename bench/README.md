# Benchmark

本目录只保留可公开、可复现的小型测试集、Runner、指标实现和审核后的摘要。模型权重、网页正文、完整 Token Trace、运行日志和机器配置不进入 Git。

## 文件

| 路径 | 作用 |
|---|---|
| `realtime_web_retrieval.jsonl` | 50条中英文实时网页检索问题 |
| `retrieval_schema.py` | Schema、枚举、唯一性和字段校验 |
| `retrieval_metrics.py` | Domain/Target Recall、垃圾率、抓取率和延迟 |
| `run_realtime_retrieval_bench.py` | 完整 Discovery + Fetch Runner |
| `generate_p4_queries.py` | 用 G1I/P4 greedy 生成严格 Tool Call |
| `query_formation.py` | 原问题、规则与模型查询的比较逻辑 |
| `run_query_formation_bench.py` | Query Formation Runner |
| `run_candidate_admission_bench.py` | 固定候选离线重放 |
| `baselines/` | 小型冻结摘要，不含网页正文 |

## 快速运行

```bash
PYTHONPATH=src python -m unittest -v tests.test_realtime_retrieval_bench_data

PYTHONPATH=src python bench/run_realtime_retrieval_bench.py \
  --config configs/benchmark.json \
  --case-id retrieval-zh-001 \
  --case-id retrieval-en-006
```

全量运行：

```bash
PYTHONPATH=src python bench/run_realtime_retrieval_bench.py \
  --config configs/benchmark.json \
  --output bench/runs/retrieval.jsonl \
  --summary bench/runs/retrieval_summary.json
```

`bench/runs/` 被 Git 忽略。冻结摘要前必须记录数据 SHA、配置、代码版本、网络出口和上游搜索引擎，并人工检查失败样本。

完整方法见 [`docs/BENCHMARK.md`](../docs/BENCHMARK.md)。
