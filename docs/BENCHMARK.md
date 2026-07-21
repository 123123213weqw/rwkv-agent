# 实时网页检索 Benchmark

## 目标

评测从问题到网页证据的检索链路，不混入最终答案生成。这样可以区分：

- 模型是否形成了正确搜索词；
- Discovery 是否找到正确官方域名；
- 是否找到真正需要的目标页面；
- 页面是否抓取和抽取成功；
- 最终候选是否被垃圾页或排序问题污染。

## 数据集

`bench/realtime_web_retrieval.jsonl` 包含 50 条人工可审核查询，中文和英文各 25 条，覆盖：

- 软件版本和官方文档；
- GitHub Release/Commit；
- 政府政策与统计数据；
- 公司财报、公告和监管文件；
- 官方 Newsroom；
- 论文；
- 台风、地震等实时公共信息。

每条记录包含：

| 字段 | 含义 |
|---|---|
| `query` | 原始用户问题 |
| `language` / `category` | 分组维度 |
| `freshness` | stable/latest/realtime |
| `source_policy` | official/original/primary 来源要求 |
| `expected_domains_any` | 至少应召回的官方域名 |
| `target_url_patterns_any` | 可接受的目标页面路径 |
| `forbidden_result_types` | 搜索首页、字典、错误页、验证码、空正文等 |

运行时绝不能把期望域名或目标路径作为搜索提示注入系统。

## 三阶段候选

1. `initial_candidates`：P4 主查询及显式来源通道的首轮结果。
2. `post_pivot_candidates`：从高置信组织域执行一次有限 `site:` 查询后的结果。
3. `candidates`：同组织一跳链接扩展后的最终候选。

每个候选保留来源通道、Discovery 阶段、父页面、查询、引擎和原始位置。

## 指标

- Candidate/Result Domain Recall@5/10/20
- Candidate/Result Target Page Recall@10/20
- 非空结果率
- 垃圾结果率
- 抓取成功/失败/取消数与成功率
- 平均和 P95 延迟
- 语言、类别、来源策略分组

Domain Recall 只能说明找到了组织网站，Target Page Recall 才说明找到了发布、财报、政策正文等具体页面。

## 正确的优化流程

1. 冻结数据集、模型查询、配置和指标定义。
2. 先运行关闭新功能的基线。
3. 只改变一个可解释阶段。
4. 固定候选时做离线重放，消除搜索引擎波动。
5. 再运行相同 Live Benchmark。
6. 同时报告跨运行观测和同一次运行的阶段增益。
7. 保存失败样本，不只展示平均分。

## 当前冻结结果

公开摘要位于：

- `bench/baselines/query_formation/g1i-p4-v1/`
- `bench/baselines/realtime_retrieval/precision-discovery-v1/`
- `bench/baselines/g1i/`

完整网页正文、模型 Token Trace 和私有机器配置不进入 Git。

## 已知标签问题

两条 llama.cpp 用例仍保留历史仓库路径 `/ggerganov/llama.cpp`，当前官方仓库为 `/ggml-org/llama.cpp`。为了保持与冻结历史基线可比，本版本未静默修改标签，因此当前 Target Recall 会低估这两条的实际召回。
