# Benchmark

本目录只保留可公开、可复现的小型测试集、Runner、指标实现和审核后的摘要。模型权重、网页正文、完整 Token Trace、运行日志和机器配置不进入 Git。

## 文件

| 路径 | 作用 |
|---|---|
| `realtime_web_retrieval.jsonl` | 冻结的50条中英文历史回归集，不再修改 |
| `realtime_web_retrieval_dev_v2.jsonl` | 新增100条、50个中英文配对主题的真实风格开发集 |
| `realtime_web_retrieval_dev_v2_manifest.json` | v2数量、分布、SHA-256和数据来源声明 |
| `build_retrieval_dev_v2.py` | 从人工审核主题表确定性重建v2 JSONL与Manifest |
| `retrieval_schema.py` | Schema、枚举、唯一性和字段校验 |
| `retrieval_metrics.py` | Domain/Target Recall、垃圾率、抓取率和延迟 |
| `run_realtime_retrieval_bench.py` | 完整 Discovery + Fetch Runner |
| `retrieval_failure_attribution.py` | 沿Discovery、抓取、抽取和结果阶段归因失败 |
| `generate_p4_queries.py` | 用 G1I/P4 greedy 生成严格 Tool Call |
| `query_formation.py` | 原问题、规则与模型查询的比较逻辑 |
| `run_query_formation_bench.py` | Query Formation Runner |
| `run_candidate_admission_bench.py` | 固定候选离线重放 |
| `web_extraction_cases.jsonl` | 30条真实官方网页静态抽取用例 |
| `web_extraction_schema.py` | 抽取用例Schema和一致性校验 |
| `web_extraction.py` | 固定快照、失败分类、抽取器适配和指标 |
| `run_web_extraction_bench.py` | Live Capture与固定快照A/B Runner |
| `searxng_engine_bench.py` | 逐引擎稳定性、跨轮命中和公开脱敏汇总 |
| `run_searxng_engine_bench.py` | SearXNG逐引擎隔离Runner，不修改服务配置 |
| `candidate_rerank.py` | 候选级Admission、Cross-Encoder融合与离线排序指标 |
| `run_candidate_rerank_bench.py` | 在冻结搜索候选上运行语义Rerank A/B |
| `external/miracl-v1/` | MIRACL中英文dev人工qrels、上游哈希和许可记录 |
| `long_knowledge_compat_v1/` | 48条项目长期知识兼容集及明确本地缺失探针 |
| `long_knowledge_schema.py` | page-level qrels、query type和local-missing Schema |
| `long_knowledge_metrics.py` | Hit/Recall、MRR、nDCG、覆盖率、缺失准确率和延迟指标 |
| `run_long_knowledge_bench.py` | FineWiki长期知识页级检索Runner |
| `baselines/` | 小型冻结摘要，不含网页正文 |

## 快速运行

```bash
PYTHONPATH=src python -m unittest -v tests.test_realtime_retrieval_bench_data

# 确定性重建100条v2开发集及Manifest
PYTHONPATH=src python bench/build_retrieval_dev_v2.py

PYTHONPATH=src python bench/run_realtime_retrieval_bench.py \
  --config configs/benchmark.json \
  --case-id retrieval-zh-001 \
  --case-id retrieval-en-006
```

`realtime_web_retrieval_dev_v2.jsonl`是人工策划的真实搜索风格开发集，不是用户日志，也不是私有盲测集。
它包含口语、短查询、少量噪声输入、安全公告、标准规范、公共实时信息、公司原始文件、技术文档和社区讨论。
中英文按同一50个检索目标成对构建，可直接比较跨语言召回差异。`query_style`、`task_family`、
`gold_ttl_days`、期望域和路径都只能用于运行结束后的分组和评分，禁止进入模型或搜索查询。

运行v2开发集：

```bash
PYTHONPATH=src python bench/run_realtime_retrieval_bench.py \
  --bench bench/realtime_web_retrieval_dev_v2.jsonl \
  --config configs/benchmark.json \
  --output bench/runs/retrieval_dev_v2.jsonl \
  --summary bench/runs/retrieval_dev_v2_summary.json
```

### SearXNG逐引擎稳定性

Runner从运行中实例的`/config`读取已启用引擎，只用冻结P4查询请求单个引擎。期望域名和URL模式仅在结果返回后用于打分，不加入搜索词。稳定性运行应顺序限速，避免把并发限流误判为引擎质量：

```bash
PYTHONPATH=src python bench/run_searxng_engine_bench.py \
  --endpoint http://127.0.0.1:8888 \
  --model-queries bench/runs/query_formation_p4_queries_v1.jsonl \
  --engine mwmbl \
  --repetitions 2 \
  --concurrency 1 \
  --request-delay 0.5
```

完整URL只写入被Git忽略的`bench/runs/`；公开基线位于：

- `baselines/realtime_retrieval/searxng-engine-stability-v1/`：现有四引擎稳定性；
- `baselines/realtime_retrieval/searxng-candidate-engines-v1/`：Google、Bing、DuckDuckGo、Brave、Startpage和Qwant隔离试用。

### 冻结候选语义Rerank

Rerank Runner只读取已经冻结的Discovery JSONL，不重新请求搜索引擎。Cross-Encoder只接收P4查询和候选的标题、规范化URL来源及摘要；期望域名、目标路径和禁止类型只在排序完成后参与评估：

```bash
PYTHONPATH=src python bench/run_candidate_rerank_bench.py \
  --input bench/runs/searxng_bing_cn_paced_v1.jsonl \
  --model BAAI/bge-reranker-v2-m3 \
  --device cuda --batch-size 16 --max-length 512 --fp16 \
  --output bench/runs/candidate_rerank.jsonl \
  --summary bench/runs/candidate_rerank_summary.json \
  --public-summary bench/runs/candidate_rerank_public.json
```

固定比较四个阶段：Bing原始顺序、通用Admission、纯语义Rerank、Admission与语义等权融合。指标包含Domain Recall@5/10、Target Recall@10/20、Domain/Target MRR、Top8垃圾率、硬过滤误删和P95延迟。公开冻结结果位于`baselines/realtime_retrieval/candidate-rerank-bge-m3-v1/`，完整URL、摘要和逐候选分数只保留在被Git忽略的`bench/runs/`。

全量运行：

```bash
PYTHONPATH=src python bench/run_realtime_retrieval_bench.py \
  --config configs/benchmark.json \
  --output bench/runs/retrieval.jsonl \
  --summary bench/runs/retrieval_summary.json
```

`bench/runs/` 被 Git 忽略。冻结摘要前必须记录数据 SHA、配置、代码版本、网络出口和上游搜索引擎，并人工检查失败样本。

两次独立Live运行完成后，可以生成跨阶段失败归因：

```bash
PYTHONPATH=src python bench/retrieval_failure_attribution.py \
  bench/runs/run1.jsonl bench/runs/run2.jsonl \
  --output bench/runs/attribution.json \
  --case-matrix bench/runs/attribution_cases.json
```

它只使用Runner已经记录的可观察阶段，不把Benchmark期望域名或目标路径反馈给运行时搜索。

## 固定网页抽取

先安装只用于Benchmark的抽取器：

```bash
pip install -e '.[extraction-bench,dev]'
```

第一次采集真实页面。正文只会进入被Git忽略的`data/`：

```bash
PYTHONPATH=src python bench/run_web_extraction_bench.py \
  --capture --capture-only
```

之后所有抽取器读取完全相同的本地字节快照：

```bash
PYTHONPATH=src python bench/run_web_extraction_bench.py \
  --repeat 3 \
  --extractors current,hybrid_fast,trafilatura,justext,readability,resiliparse
```

`hybrid_fast`直接调用`src/rwkv_search/realtime/hybrid_extractor.py`，因此固定快照验证的是实时链路使用的同一套实现，而不是一份Benchmark复制代码。它使用Resiliparse快速正文、Trafilatura轻量元数据，并只在通用低质量信号命中时运行完整Trafilatura兜底。公开冻结结果见：

- `baselines/web_extraction/static-extractors-v1/`：4B-1五种静态抽取器基线；
- `baselines/web_extraction/hybrid-fast-v1/`：4B-2混合快速实现的接入前冻结基线。

公开文件不包含网页正文、响应头或完整运行日志。

完整方法见 [`docs/BENCHMARK.md`](../docs/BENCHMARK.md)。
