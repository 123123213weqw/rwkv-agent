# 优化里程碑

| 阶段 | 主要改动 | 结果/状态 |
|---|---|---|
| M1 Benchmark | 50条真实查询、Schema、阶段Trace和指标 | 完成 |
| M2 SearXNG | 自托管 JSON Discovery，保留 HTML fallback | 完成；单独使用未优于基线 |
| M2.2 Query Formation | 原问题/规则/P4同候选比较 | P4 Domain@10 34% → 58% |
| M3 Candidate Admission | 标题、URL、摘要、RRF、来源约束、垃圾过滤 | 固定候选无召回回退，垃圾率显著下降 |
| M3.1 Discovery Correctness | P4唯一主查询、禁止原问题回灌、保留查询溯源 | 完成 |
| M4A Precision Discovery | 来源通道、官网Pivot、同站一跳 | Candidate Domain@10 42% → 48%；Target@20 6% → 12% |
| M4B Extraction | 静态正文抽取和有限JS后备 | 待开始 |
| M5 Evidence | 段落选择、证据多样性、Claim-Citation | 待开始 |
| M6 Shadow | Search Gate/P4接入现有聊天Shadow | 待开始 |
| M7 Deep Research | 有限多轮搜索和证据缺口判断 | 待开始 |
| M8 Frontend | 搜索进度、引用、来源卡片 | 待开始 |

## 关键结论

- Tool Call 序列化已经稳定，不是当前主要瓶颈。
- 搜索词形成对召回影响很大，P4 比规则清洗更有效。
- 候选过滤不能补回 Discovery 从未返回的 URL。
- 官网 Pivot 与一跳链接能提高精确页面候选，但最终页面抓取仍需继续优化。
- 所有实验优化默认关闭，必须经过 Shadow 和质量门槛后才能进入生产。
