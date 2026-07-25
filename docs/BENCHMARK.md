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

### v2开发集扩充

历史50条继续作为冻结回归集，文件和旧指标均不修改。新增
`bench/realtime_web_retrieval_dev_v2.jsonl`作为独立开发集，包含100条问题：

- 50个检索目标，每个目标各有1条中文和1条英文查询，用于直接比较跨语言差异；
- 50条中文、50条英文；标准问法、口语问法、短查询和少量噪声查询均有覆盖；
- 新增安全公告、标准规范、产品支持、社区讨论等来源形态；
- 每条均有人工作者指定的可审核来源域、目标路径和标签有效期；
- `gold_ttl_days=1`的实时标签需要每日重新确认，稳定规范最长可使用730天。

扩展元数据：

| 字段 | 含义 |
|---|---|
| `query_style` | canonical/conversational/terse/noisy，仅用于分组 |
| `task_family` | 更细的离线失败分析维度，不参与路由 |
| `gold_ttl_days` | 来源和目标页面标签重新审核周期 |
| `annotation_status` | 当前只表示来源策略和页面形态已人工审核 |
| `origin` | 固定为`manually_curated_realistic` |

该开发集不是从真实用户日志抽样，也不是私有盲测集，不能据此声称真实流量表现。它的作用是扩大开发覆盖面、
发现中英文和输入风格差异。真实日志集需要合法的数据来源、隐私清洗和独立标注；私有盲测必须与开发集隔离。
确定性构建脚本和分布Manifest分别为`bench/build_retrieval_dev_v2.py`和
`bench/realtime_web_retrieval_dev_v2_manifest.json`。

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
- 候选级 Domain/Target MRR、Top8垃圾率与硬过滤误删数
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

`bench/retrieval_failure_attribution.py`把每条问题归入互斥生命周期结果：未发现官方域、已发现域但无精确页、目标未进入抓取、抓取/抽取失败、抓取后拒绝、最终排序淘汰或成功命中。它同时保留`initial → post_pivot → one_hop`阶段增益及失败类型，避免把网络超时误判为正文抽取问题。

## 当前冻结结果

公开摘要位于：

- `bench/baselines/query_formation/g1i-p4-v1/`
- `bench/baselines/realtime_retrieval/precision-discovery-v1/`
- `bench/baselines/realtime_retrieval/hybrid-live-audit-v1/`
- `bench/baselines/realtime_retrieval/agent-web-shadow-v1/`
- `bench/baselines/g1i/`

完整网页正文、模型 Token Trace 和私有机器配置不进入 Git。

### Agent Web Shadow v1

桌面Agent保持原`web_search(query)`、Legacy可见`W1..W5`和回答Prompt不变，将增强路径放入默认关闭
的单worker、最多两个等待任务的异步Shadow。每个A/B Arm记录完整候选阶段、抓取、最终结果、垃圾类型、
告警、回退原因和延迟；队列满或异常只丢弃Shadow，不得影响聊天请求。

50条冻结集在V100隔离环境的结果是：

| 指标 | Legacy | Enhanced |
|---|---:|---:|
| Candidate Domain Recall@10 | 32% | 26% |
| Candidate Target Page Recall@20 | 2% | 2% |
| Result Domain Recall@10 | 24% | 20% |
| Result Target Page Recall@20 | 0% | 0% |
| 非空结果率 | 86% | 58% |
| 垃圾结果率 | 17.56% | **1.08%** |
| 抓取成功率 | 72.99% | 58.77% |
| 平均/P95延迟 | 2820.5/7223.5ms | 2680.7/7072.9ms |

通用候选准入明显降低垃圾，但当前组合牺牲了有效召回和非空，因此Shadow安全门槛通过、默认切换门槛
失败。两臂交替先后顺序各25条，仍观察到Live上游/执行顺序干扰；实验机本地SearXNG不可用，两臂
实际都使用Bing HTML fallback。因此这组结果不能包装成SearXNG或多搜索引擎的质量结论。

### Agent Web Recall Repair 5H

5H使用同一50条、同一指标修复5G暴露的召回与非空回退。增强Arm增加：

1. 只对长英文自然语言请求做主题前置的通用Query Compaction，不按领域或金标站点路由；
2. 候选重排可改变已准入首屏内部顺序，但后部候选不能挤掉原Top-10集合；
3. Legacy/Enhanced发出相同查询时共享有TTL和字节上限的进程内Discovery缓存；
4. Trace区分Discovery、Admission、Fetch、Post-fetch和Final-ranking失败；
5. Enhanced无公开Evidence时记录Legacy Evidence安全回退；仍不改变用户可见输出。

| 指标 | 配对Legacy | Enhanced |
|---|---:|---:|
| Candidate Domain Recall@10 | 30% | **50%** |
| Result Domain Recall@10 | 24% | **40%** |
| Candidate Target Page Recall@10 | 8% | **14%** |
| Result Target Page Recall@10 | 6% | **10%** |
| 非空结果率 | 72% | **88%** |
| Evidence非空率 | 96% | **98%** |
| 垃圾结果率 | 9.15% | **1.12%** |
| 抓取成功率 | 56.52% | **71.35%** |
| 平均/P95延迟 | 3291.3/8008.4ms | 3301.5/**6901.9ms** |

Candidate Domain Recall@10、Result Domain Recall@10和非空结果逐例分别为Enhanced胜10/8/8、
Legacy胜0/0/0。英文Candidate Domain Recall@10从0%提升到40%，中文两臂保持60%。

隔离SearXNG的Bing/DuckDuckGo等公开引擎在持续运行中出现超时、连接重置、CAPTCHA或限流；
失败诊断没有被包装成健康多引擎结果。最终通过门槛的运行使用已有Bing HTML fallback，证明的是
Query/Admission/Fetch修复而非持久SearXNG能力。公开摘要位于
`bench/baselines/realtime_retrieval/agent-web-recall-5h-v1/`。

## 静态网页抽取 Benchmark

`bench/web_extraction_cases.jsonl`与检索问题集分离，专门回答“给定同一个网页响应，哪个低资源抽取器能稳定保留可回答内容”。30条真实官方页面覆盖：

- 中英文文章、Release、文档、仓库和政府页面；
- 作者、发布日期、表格与代码块；
- JSON、Markdown、PDF和JavaScript空壳；
- 页头、导航、页脚等污染文本。

执行分为两个阶段：

1. `--capture`使用有并发、超时、重定向和响应大小上限的`aiohttp`客户端抓取一次，将原始字节与Manifest放入被Git忽略的`data/web-extraction-bench/`。
2. 固定快照Runner让实时链路、`hybrid_fast`、Trafilatura、jusText、Readability和可选Resiliparse读取完全相同的字节；`hybrid_fast`直接调用实时链路的生产实现，避免候选与接入代码漂移；每个抽取重复3次并记录中位耗时。

失败被分为DNS、连接、TLS、连接/请求超时、HTTP 403/429/4xx/5xx、重定向、响应类型、响应大小、空响应、截止取消、解码、JS空壳、抽取器不可用、抽取异常、空正文和低质量，不再把所有失败压成“没有内容”。

抽取指标包括严格通过率、可用正文、标题、正文标记、禁止导航泄漏、作者、发布日期、表格、代码、输出长度、平均和P95耗时，并按语言和页面类型分组。运行记录只保存摘要、长度和SHA-256，不保存抽取正文。

冻结结果位于：

- `bench/baselines/web_extraction/static-extractors-v1/`：原始五种抽取器基线；
- `bench/baselines/web_extraction/hybrid-fast-v1/`：Resiliparse正文 + Trafilatura元数据/有限完整兜底的接入前冻结基线。

4B-2候选不读取用例标签、域名或页面类型，只按空/短正文、主内容比例和短正文尾部通用导航污染触发完整兜底。同一轮结果为：严格通过率85.71%（current为78.57%），可用正文86.96%、正文标记95.65%、导航泄漏8.70%均不退化，作者与日期命中均100%；完整兜底触发4/28、实际替换1/28。平均耗时70.1ms，对比同轮current的197.8ms降低64.55%。

这些结果只用于选择静态快速路径，不代表搜索召回或最终回答质量。4B-3已将同一实现接入源码实时抽取路径，但没有部署或重启常驻服务；浏览器后备和Shadow切换仍未授权。

## 候选级语义Rerank Benchmark

4C-5使用4C-3已经冻结的两轮Bing候选做纯离线重放，从而把搜索引擎波动与排序收益分离。每条候选只向Cross-Encoder提供P4查询、标题、规范化URL来源和搜索摘要，不提供Benchmark期望域名、目标路径、类别或来源标签。

固定消融包括：

1. `raw`：Bing原始顺序；
2. `admission`：确定性垃圾过滤、元数据评分和域名多样化；
3. `semantic`：仅Cross-Encoder相关性；
4. `hybrid`：垃圾过滤后，将语义分数与元数据分数等权融合，再执行域名多样化。

冻结候选为50条查询、两轮共100条记录。`BAAI/bge-reranker-v2-m3`以FP16、batch 16、最大512 tokens在V100运行。Hybrid相对raw将Domain Recall@5从48%提高到56%，Domain Recall@10保持56%，Target Recall@20保持8%，Domain MRR从0.372413提高到0.411333，Target MRR从0.026857提高到0.05，Top8垃圾率从17.38%降到0；未误删可用官方域或目标页。模型常驻后的平均/P95候选批次耗时为23.963/24.649ms，峰值显存约1.12GB。纯语义排序会降低Domain MRR，因此没有选择“只上Reranker”的方案。

这些结果只批准离线候选，不代表已经接入在线Shadow或生产。

## 已知标签问题

两条 llama.cpp 用例仍保留历史仓库路径 `/ggerganov/llama.cpp`，当前官方仓库为 `/ggml-org/llama.cpp`。为了保持与冻结历史基线可比，本版本未静默修改标签，因此当前 Target Recall 会低估这两条的实际召回。
