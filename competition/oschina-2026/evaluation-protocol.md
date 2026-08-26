# A/B/C 可复现实验协议

## 固定变量

- 同一模型权重 digest、revision、Tokenizer、State ABI、精度与推理参数；
- 同一 100 个 Session 请求到达时间序列、Prompt/Transcript 和输出 token 上限；
- 同一硬件池、驱动、容器镜像、网络区域、对象大小和观测窗口；
- 每方案至少 3 次独立运行；报告 median run，并保留所有 raw logs；
- correctness 失败、上下文串扰、双写或未标注数据丢失使该轮无效。

## Baseline A：Sticky Worker

Session 首次落到 Worker 后保持亲和；即使 Session 空闲也不释放其 Worker/State。

预期：最佳继续延迟和 Hot 命中；GPU idle minutes 高。

## Baseline B：Stateless re-prefill

每轮可选任意兼容 Worker；请求结束释放推理 State；下一轮从完整 transcript
re-prefill。

预期：容易 scale-to-zero；重复 Prefill tokens 和恢复延迟高。

## Scheme C：StatePool

请求结束按策略把 State 从 Hot 降到 Warm/Cold；Lease + CAS 提交后才允许
Worker Drain；下一轮在 exact-compatible Worker restore 并 continue。

## 必测场景

1. 100 Session，活跃 60 秒、空闲 5 分钟、再活跃 60 秒；
2. burst：0→1→32 并发→0；
3. 在已提交 State 后强制删除 Worker Pod；
4. 在 Snapshot 过程中杀 Worker，验证旧 Lease不能晚提交；
5. 注入错误 revision/tokenizer/state_abi，必须 re-prefill/reject；
6. local_only/hybrid/cloud_allowed 各 20 个请求；
7. 对象校验和破坏，restore 必须 fail closed；
8. KEDA/Prometheus 暂时不可用，不得触发双执行。

## 指标定义

| 指标 | 计算 |
|---|---|
| GPU average utilization | 观测窗内 DCGM/供应商 busy 百分比平均 |
| GPU idle minutes | 分配 GPU 且 busy 低于固定阈值的分钟数 |
| GPU-hours / 100 Sessions | 所有 Worker 分配 GPU 秒数 / 3600，标准化到 100 Session |
| restore P50/P95 | restore 接收至 Worker 确认可 continue |
| scale-from-zero | demand metric 首次 >0 至 Worker ready |
| scale-to-zero | 最后请求完成至最后 Worker Pod 终止 |
| fault recovery | Pod delete 至相同 Session 产生正确下一 token |
| State hit rate | `(Hot + Warm + Cold exact restore) / eligible continues` |
| bytes migrated | Snapshot + Restore 的 payload bytes |
| Prefill tokens avoided | B 的实际 Prefill tokens − 方案实际 Prefill tokens |
| estimated cost | `GPU seconds × pinned GPU-hour price / 3600 + storage/egress estimate` |
| deployment time | 从干净集群执行文档命令到 smoke pass |

估算费用必须标记 `estimated` 并记录价格来源/日期/币种；不得跨币种直接求和。

## 正确性断言

- 每个输出绑定 request/session/owner/State version/fencing token；
- deterministic 配置下，故障恢复后文本/token IDs 与未故障控制组比较；
- 任意时刻同一 Session 最多一个有效 writer；
- 当前 State version 单调递增且没有两个同 version 不同 checksum；
- 所有 release/drain 后 Worker busy/dirty/State slot 计数归零；
- local_only 原始 State 不出本机/允许区域。

## 原始产物布局

```text
bench/artifacts/statepool-<date>-<hardware>/
  manifest.json
  commands.txt
  images.txt
  events.jsonl
  prometheus/
  logs/
  results.csv
  summary.json
  README.md
```

`manifest.json` 必须含 Git commit、dirty 状态、硬件、驱动、Kubernetes/KEDA、
镜像 digest、模型 digest、区域、币种和 measured/estimated 字段。

## 当前可用的仿真回放

[`bench/artifacts/statepool-abc-replay-20260827/`](../../bench/artifacts/statepool-abc-replay-20260827/README.md)
把 100 Session、两段 60 秒活跃和 300 秒空闲固定为同一 trace，并引用真实
KEDA 时间、RTX 4080 State 大小/单次本地 snapshot/restore 和 100 Session
contract correctness。它输出 `simulation_replay`，不能替代本协议要求的
same-topology 三次 live GPU 实验；其作用是让比赛叙事中的成本公式先可审计、
可改参数、可复算，而不是填造尚未采集的 GPU utilization 或 restore P95。
