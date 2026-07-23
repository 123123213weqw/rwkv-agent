# RWKV Search

面向 RWKV 的本地优先联网搜索与证据问答系统。项目将**是否搜索、搜索词生成、URL 发现、网页抓取、证据选择和回答生成**拆成可单独评测的阶段，避免只凭最终回答主观判断搜索质量。

> 当前状态：研究预览版。截至 2026-07-23，Tool Call、查询形成、实时 URL Discovery、候选准入与
> Rerank、低资源网页抽取、条件反馈搜索以及 FineWiki 长期知识 Benchmark 已经形成可复现链路；优化路径
> 默认关闭，尚未自动替换生产聊天链路。

## 核心设计

```mermaid
flowchart LR
    U[用户问题] --> G[Search Gate]
    G -->|普通聊天| C[RWKV 直接回答]
    G -->|需要搜索| P[G1I / P4 Tool Call]
    P --> R[SearchRequest 约束合并]
    R --> S[SearXNG / HTML fallback]
    S --> A[候选准入与排序]
    A --> D[官网域名 Pivot]
    D --> H[同站一跳链接]
    H --> F[并发抓取与正文抽取]
    F --> E[Evidence 与引用]
    E --> W[RWKV 回答]
```

- **普通聊天优先**：只有 Search Gate 或用户主动搜索才进入联网链路。
- **RWKV 原生 Tool Call**：G1I 模型以 greedy 解码输出单个严格 `web_search` 调用。
- **确定性约束合并**：模型只负责形成检索表达；时间、来源、`site:` 等硬约束由程序合并。
- **不按行业硬编码**：来源通道表达仓库、论文等显式来源形态，不建立股票、软件、政策分类路由。
- **低资源抓取**：`aiohttp`抓取，Resiliparse快速正文 + Trafilatura轻量元数据；仅在通用低质量信号命中时运行完整Trafilatura兜底，并保留缺少可选依赖时的安全降级。
- **可审计 Benchmark**：同时保存首轮候选、官网 Pivot、一跳扩展、抓取结果、阶段事件和延迟。

## 当前效果

### 已完成的可复现能力

- G1I/P4 greedy 解码和严格单个 `web_search(query)` Tool Call，并保留格式、语义和真实搜索基线。
- 冻结50条历史实时检索集、100条中英文配对开发集、30条网页抽取集、1,192条MIRACL人工qrels，
  以及48条项目兼容集；测试标签只在返回后评分，不进入运行时查询。
- SearXNG/HTML Discovery、通用Candidate Admission、官网Domain Pivot、同站一跳链接和可选BGE-M3
  Rerank；没有按股票、软件、政策等领域硬编码检索路由。
- Resiliparse正文快速路径 + Trafilatura元数据与有限完整兜底；固定网页快照上混合抽取通过率由
  78.57%升至85.71%，平均耗时从197.83ms降至70.14ms。
- FineWiki中文全量长期知识索引和英文全量构建工具；英文语料37.72GB、6,614,655行，独立索引仍在
  构建，不覆盖中文索引。

### 实时网页检索

同一组 50 条中英文实时检索问题、同一实验主机和冻结 P4 查询的配对结果：

| 指标 | 基线 | 精确发现 | 变化 |
|---|---:|---:|---:|
| Candidate Domain Recall@10 | 42% | **48%** | +6pp |
| Candidate Target Page Recall@20 | 6% | **12%** | +6pp |
| Result Domain Recall@10 | 20% | **32%** | +12pp |
| 非空结果率 | 56% | **68%** | +12pp |
| 垃圾结果率 | 1.11% | **0.97%** | -0.14pp |
| 抓取成功率 | 44.01% | **53.95%** | +9.94pp |
| 平均耗时 | 4263 ms | 5067 ms | +803 ms |
| P95 | 8010 ms | 8010 ms | 基本不变 |

搜索上游会波动，因此跨运行差值只作为观测；Pivot 与一跳扩展的算法贡献以同一次运行的 `initial → post_pivot → final` 阶段指标为准。详细方法见 [Benchmark 文档](docs/BENCHMARK.md)。

在新增100条开发集上，条件反馈C相对单次P4的Candidate Domain Recall@10由36%升至39%，
Target Page Recall@20由2%升至4%，没有已命中case回退；代价是平均查询数0.97→1.36，平均耗时
1598ms→3067ms。它仍是隔离实验，绝对精确页面召回尚未达到生产目标。

### FineWiki长期知识检索

| 测试集 | 页面覆盖 | Hit@10 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| MIRACL中文dev，393条 | 86.85% | 43.51% | 28.31% | 30.36% | 24.16% |
| 项目中文兼容集，21条正例 | 100% | 76.19% | 73.81% | 64.88% | 65.78% |

项目兼容集另有3条中文本地缺失探针，目前expected-missing准确率只有33.33%，说明Candidate Index
仍缺少可靠的准入/拒答阈值。英文MIRACL和双语汇总将在英文全量索引完成后冻结。

## 快速开始

要求 Python 3.10+。

```bash
git clone https://github.com/123123213weqw/rwkv-search.git
cd rwkv-search
python -m venv .venv
source .venv/bin/activate
pip install -e '.[realtime,dev]'

# 初始化本地索引并启动 Web UI；不加载模型时使用抽取式降级回答
rwkv-search --config configs/default.json init
rwkv-search --config configs/default.json serve --host 127.0.0.1 --port 8765 --no-model
```

打开 <http://127.0.0.1:8765>。

接入 Hugging Face 格式 RWKV：

```bash
rwkv-search --config configs/default.json serve \
  --host 127.0.0.1 --port 8765 \
  --model /path/to/rwkv-hf \
  --model-label RWKV \
  --device cuda:0 --dtype fp16
```

完整安装、SearXNG 和 G1I/P4 用法见 [使用指南](docs/USAGE.md)。

## 运行 Benchmark

```bash
# Schema、指标与离线测试
PYTHONPATH=src python -m unittest discover -s tests

# 先跑少量联网 Smoke
PYTHONPATH=src python bench/run_realtime_retrieval_bench.py \
  --config configs/benchmark.json \
  --case-id retrieval-zh-001 \
  --case-id retrieval-en-006

# 全部 50 条
PYTHONPATH=src python bench/run_realtime_retrieval_bench.py \
  --config configs/benchmark.json

# 本地长期知识检索（FineWiki × MIRACL，中英文人工 qrels）
PYTHONPATH=src:. python -m bench.run_long_knowledge_bench \
  --cases bench/external/miracl-v1/miracl_long_knowledge_dev_v1.jsonl \
  --index rwkv-finewiki-zh-full-v1 --language zh \
  --output bench/runs/long-knowledge-zh.jsonl \
  --summary bench/runs/long-knowledge-zh-summary.json
```

较小的项目回归集可将`--cases`替换为`bench/long_knowledge_compat_v1/cases.jsonl`，并选择对应的
语言和索引。

运行产物写入被 Git 忽略的 `bench/runs/`。只有经过人工复核、记录环境和配置的汇总才进入 `bench/baselines/`。

## 仓库结构

```text
src/rwkv_search/    聊天、RWKV、实时检索和本地检索实现
bench/              公开测试集、Runner、指标与冻结摘要
configs/            通用配置和 Benchmark 配置
contracts/          前后端事件与来源契约
deploy/searxng/     本地 SearXNG 示例
docs/               架构、使用、Benchmark 和里程碑
tests/              单元与回归测试
```

## 文档

- [架构](docs/ARCHITECTURE.md)
- [使用指南](docs/USAGE.md)
- [Benchmark 方法](docs/BENCHMARK.md)
- [长期知识语料与Benchmark](docs/LONG_KNOWLEDGE_BENCHMARK.md)
- [优化里程碑](docs/MILESTONES.md)
- [贡献指南](CONTRIBUTING.md)

## 已知限制

- 当前实时搜索的最大瓶颈是搜索引擎没有发现精确目标URL；Tool Call序列化和静态正文抽取已不是首要瓶颈。
- 长期知识检索尚未加入Dense Retriever，且缺少可靠的本地知识准入/拒答阈值。
- SearXNG 的结果质量取决于可用搜索引擎和网络出口。
- `configs/default.json` 中候选增强能力默认关闭；请先使用 Benchmark 或 Shadow 验证再启用。
- 仓库不包含 RWKV 权重、第三方搜索 API 密钥、抓取正文、私有服务器配置或完整调试 Trace。

## License

[MIT](LICENSE)
