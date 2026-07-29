# Discovery Profile A/B/C/D（隔离实验）

本实验在统一回归集的 WebWalkerQA 40 条和 FRAMES 40 条上，比较历史 Legacy、P0 Legacy、
Balanced 与 Enhanced Discovery。模型、两轮四分支查询协调器、并发和预算保持一致；C/D 运行时
V100 本机 SearXNG 不可用，因此都使用 Bing HTML 后备。

## 配置

| 组 | 证据策略 | Discovery 功能 |
|---|---|---|
| A | P0 前历史逻辑 | Legacy |
| B | P0 严格范围、页面质量、Evidence 不足拒答 | Legacy |
| C | 同 B | Candidate Admission + Query Compaction + Domain Pivot |
| D | 同 B | C + Source Channels + One-hop Link Expansion |

## WebWalkerQA 40

| 组 | F1 | Gold 域 Recall | 精确页 Recall | 非空 Evidence | 回答数 | P95 |
|---|---:|---:|---:|---:|---:|---:|
| A | 22.71% | 22.5% | 2.5% | 100% | 40 | 19.38s |
| B | 1.73% | 20.0% | 0% | 20.0% | 0 | 18.74s |
| C | 1.59% | 22.5% | 2.5% | 22.5% | 0 | 13.47s |
| D | 6.43% | 80.0% | 2.5% | 80.0% | 14 | 29.19s |

D 的 80% 域召回不能当成普通搜索提升：WebWalkerQA 会向 Agent 提供 `root_url`，Enhanced 会把该
根页直接作为 Seed 再做一跳扩展。D 的 127 条 Evidence 中 94.49% 位于给定 Root Site；精确目标页仍
只有同一个 Case 命中。它恢复了“站内有内容”，但没有解决“找到正确历史页面”。

## FRAMES 40

| 组 | F1 | Gold 域 Recall | 精确页 Recall | 非空 Evidence | 回答数 | P95 |
|---|---:|---:|---:|---:|---:|---:|
| A | 7.29% | 0% | 0% | 100% | 40 | 20.85s |
| B | 1.00% | 0% | 0% | 72.5% | 3 | 24.11s |
| C | 1.23% | 0% | 0% | 92.5% | 3 | 25.75s |
| D | 1.53% | 0% | 0% | 92.5% | 5 | 27.33s |

FRAMES 不提供 Root Scope，因此更接近普通未知网页检索。C/D 都没有召回任何 Gold 域或精确页；
Enhanced 只多回答两条并增加延迟，没有形成可接受的检索收益。

## 结论

1. A 的高 F1 是“全部作答”带来的假象：按当前页面分类器重放，Web 320 条 Evidence 中 96 条、
   FRAMES 320 条中 175 条应被拒绝；Unsupported Claim 分别为 96.38% 和 99.06%。
2. P0 不应回退。它正确暴露了 Discovery 召回不足，而不是制造了召回问题。
3. Balanced 可以保留为下一轮实验起点，但本次尚未恢复可回答覆盖。
4. Enhanced 不应全局启用。它在有 Root Oracle 的站内遍历任务有效，但精确页召回不变、延迟翻倍附近，
   且会从相关域内的错误年份或导航页生成看似有引用、实际答非所问的答案。
5. 下一步应把主指标固定为精确目标页 Recall，并分别优化“搜索结果内深页定位”和“根站点内受限遍历”；
   不能用根域命中代替目标页命中，也不能为了覆盖率放松 Evidence Gate。

机器可读完整结果见 `discovery-profile-abcd-v1-summary.json`。A/B 为历史 Live 结果，C/D 为本次顺序
Live 结果，Bing 上游波动没有完全消除，因此算法归因应优先看同次运行内的阶段Trace和精确页变化。
