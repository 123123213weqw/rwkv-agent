# G1I P4 Query Formation Baseline v1

本基线只比较“搜索词怎么形成”，不抓取网页正文、不生成答案，也没有接入生产聊天链路。

## 固定输入与运行环境

- 50条真实检索问题：中文25、英文25
- P4模型：RWKV7 G1I preview3260 7.2B
- 解码：greedy
- P4生成：RTX 4080
- URL Discovery：V100直连Bing HTML fallback
- 对比策略：原始问题、当前`QueryAnalyzer`规则、P4严格Tool Call中的单条`query`
- 相同查询共享同一组候选，避免重复请求引入差异

## 核心结果

| 策略 | Domain Recall@5 | Domain Recall@10/20 | Target Page Recall@20 |
|---|---:|---:|---:|
| 原始问题 | 28% | 34% | 2% |
| 当前规则 | 32% | 38% | 2% |
| G1I/P4 | **50%** | **58%** | **4%** |

P4相对原始问题提升24个百分点：新增12条官方域名命中，0条原有命中丢失。P4严格Tool Call为50/50。
中文P4 Domain Recall@10为76%，英文为40%；原始英文问题在当前Bing出口为0%。

## 结论

G1I/P4是本轮明确胜出的查询形成方式，适合进入后续Shadow比较，但尚未批准生产接入。
目标页面Recall@20仍只有4%，说明下一瓶颈是精确页面发现、候选预排序和来源约束，不应把域名召回提升误认为搜索已经完成。

公开仓库只保留聚合摘要和逐例指标对比；完整Discovery正文、原始Token Trace和机器配置不进入Git。
