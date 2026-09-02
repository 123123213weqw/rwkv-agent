# RWKV Agent S-Level Benchmark

`RWKV-Agent-S-Level-v1`把生产目标和理想的S级目标固定成独立硬门槛。它不计算总分，任何核心门槛失败都不能
被其他容易指标抵消。

## 两档门槛

- `production`：允许进入生产候选的最低质量线。
- `s_level`：接近成熟商业联网问答产品的目标线。

完整数值位于`benchmarks/s_level_targets.v1.json`。主要S级目标包括：

- Search Gate准确率不低于99.5%，误触发不高于0.5%，漏触发不高于0.3%。
- 严格Tool Call成功率不低于99.9%。
- Domain Recall@10不低于99%，Exact Page Recall@20不低于95%。
- Final Evidence Exact Recall不低于90%，引用精确页Recall不低于90%。
- Citation Validity不低于99.5%，Unsupported Claim不高于1%。
- 普通搜索P50不高于5秒、P95不高于10秒；State Leak必须为0。

## 数据要求

- 至少500条隐藏测试题。
- 中文、英文必须分别报告；需要分组门槛的指标，语言子组至少达到总目标的90%。
- Live Benchmark至少独立运行三次。
- 正式数据应覆盖普通聊天、实时单跳、官方精确页、多跳、口语/错别字、证据冲突与系统异常。
- Token F1只是诊断指标，不能冒充事实正确率或完整率。

## 运行

评估完整标准化Measurement：

```bash
python benchmarks/evaluate_s_level_bench.py \
  --measurements /path/to/measurements.json \
  --profile s_level \
  --output /path/to/s-level-report.json
```

Runner只接受版本化Measurement，不再把历史模型专用结果转换成当前发布判断。没有可靠测量的指标保持缺失并
判定失败，不用Token F1猜测事实正确率，也不把单次Live运行伪装成稳定性结果。
