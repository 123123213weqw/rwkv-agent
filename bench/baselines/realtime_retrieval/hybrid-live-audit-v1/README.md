# Hybrid Live Retrieval Audit v1

状态：公开脱敏摘要；未修改生产配置、召回算法或服务。

## 方法

- 使用与Precision Discovery v1相同的50条中英文查询、冻结P4查询、RTX 4080实验主机和有界Precision配置。
- Hybrid实时抽取接入后独立运行两次Live Benchmark；完整候选、URL和抓取Trace继续只保存在被Git忽略的`bench/runs/`。
- `retrieval_failure_attribution.py`沿`initial → post_pivot → final candidate → fetch → result`生命周期归因，不读取Benchmark期望值参与运行时检索。

## 两次Live结果

| 指标 | 冻结4A参考 | Run 1 | Run 2 | 两次范围 |
|---|---:|---:|---:|---:|
| Candidate Domain Recall@10 | 48% | 44% | 48% | 44–48% |
| Candidate Target Recall@20 | 12% | 20% | 12% | 12–20% |
| Result Domain Recall@10 | 32% | 16% | 22% | 16–22% |
| Result Target Recall@20 | 0% | 4% | 2% | 2–4% |
| 非空结果率 | 68% | 76% | 70% | 70–76% |
| 垃圾率 | 0.97% | 0% | 0.91% | 0–0.91% |
| 抓取成功率 | 53.95% | 44.16% | 45.20% | 44.16–45.20% |
| 平均耗时 | 5067ms | 5304ms | 4991ms | 4991–5304ms |

Live上游波动明显，所以不能把与历史4A的单次差值直接归因给Hybrid。两次共同出现的失败桶更可靠。

## 失败归因

| 失败桶 | Run 1 | Run 2 |
|---|---:|---:|
| 首轮没有发现期望官方域名 | 28 | 26 |
| 已发现官方域，但Precision后仍无精确目标页 | 11 | 18 |
| 官方域页面进入抓取但抓取/抽取失败 | 13 | 11 |
| 精确目标页没有进入8页抓取预算 | 2 | 0 |
| 精确目标页成功抓取但最终未保留 | 2 | 1 |
| 最终命中精确目标页 | 2 | 1 |

官方域抓取失败绝大多数是连接、HTTP状态、整体截止或取消；Hybrid自身`ExtractionError`仅3/4条。两次精确目标页抓取失败里都没有`ExtractionError`，说明下一优先级不是继续调正文抽取。

## 结论

1. 第一优先级：提高首轮URL Discovery的官方域名召回，同时控制请求数与上游波动。
2. 第二优先级：官方域已知后，提高精确页面发现，而不是继续增加领域硬编码。
3. 第三优先级：对进入队列的官方页面改善有界网络抓取可靠性。
4. 抓取名额和最终排序确有少量损失，但当前不是最大失败桶。

本阶段没有修改查询形成、Discovery、排序、抓取预算、Evidence、答案、Router、前端或生产配置。
