# 12 页答辩 PPT 结构

## 1. 封面：助手常驻，GPU 不必常驻

- 项目：RWKV State Agent + StatePool Cloud Plugin
- 一句话：每个人长期拥有一个 AI 助手，而不需要长期占用一张 GPU
- 只放一个矛盾：Sticky 浪费 GPU，Stateless 重复 Prefill

## 2. 为什么是现在

- 智算集群碎片化、请求短而 Session 长
- 个人助手流量稀疏但计算历史有价值
- 上游已经有 Kubernetes/KEDA/HAMi/AIBrix；缺的是 Agent State 生命周期

## 3. 为什么是 RWKV

- recurrent State 随 token 更新
- State 大小不随完整历史线性增长
- 同 Session continue、多 Session 隔离、本地/边缘可运行
- 不宣称无限记忆或跨模型 State 迁移

## 4. 三种方案

| A Sticky | B Stateless | C StatePool |
|---|---|---|
| 保留 State 和 GPU | 释放 GPU、重做 Prefill | 保存 State、释放 GPU、恢复继续 |

页面结论：C 交换的是 State I/O，避免的是 GPU idle 和重复 Prefill。

## 5. 架构：本地是信任根，云是弹性 Worker

- Controller 默认走原本地路径
- plugin-v1 只传 Placement metadata
- PostgreSQL 管 Lease/CAS，S3 管 Cold State
- KEDA/Prometheus 管弹性，HAMi/AIBrix 为可选 Adapter

## 6. 技术创新不是 YAML

依次出现五个框：

1. exact `model/revision/tokenizer/state_abi`；
2. owner-scoped single writer Lease；
3. monotonic fencing token；
4. immutable checksum Snapshot + version CAS；
5. Hot/Warm/Cold + safe Drain + State-aware FinOps。

## 7. 默认不破坏

- `enabled=false`
- 不构造插件 Client
- 原 API/CLI/UI/Sidecar/SQLite/工具调用不变
- 插件故障只在远程操作开始前允许安全本地降级
- 已持有 Lease/提交不确定时禁止双执行

## 8. 真 GPU 数据面证据

- RTX 4080、RWKV-7 G1I 1.5B
- 12,911,277 bytes、SHA-256、PostgreSQL/S3
- source PID dead → fresh PID → new State ID → seen 42→66
- snapshot 81.266 ms、restore 104.246 ms、safe drain
- 页脚边界：同一物理 GPU sequential process；S3 隧道时间不是 SLO

## 9. 真 KEDA 控制面证据

- kind/K8s 1.34 + KEDA 2.20.1 + Prometheus 3.11.3
- replica sequence：0→1→3→0
- first desired 3.234 s；3 Ready 14.380 s；last Pod gone 62.260 s
- 3/3 preStop safe
- 页脚边界：非 GPU Worker 仿真，不讲模型吞吐

## 10. A/B/C 可审计回放

- 展示 `results.csv`，页眉红字 `SIMULATION REPLAY`
- C vs A：modeled GPU-hours -45.151%
- C vs B：avoided Prefill 51,200 tokens
- 代价：3.873 GB State transfer
- 明说没有采集 GPU utilization、production restore P95、B Prefill GPU time

## 11. 为什么不是重复造轮子

| 上游 | 它负责 | 我们补充 |
|---|---|---|
| HAMi | GPU sharing | State dirty/drain/restore cost |
| KEDA | 副本数 | pending/decode State metrics |
| AIBrix/KServe | gateway/serving | recurrent State ownership |
| vLLM | 广泛模型推理 | affinity-only Worker 契约；不谎称 KV 迁移 |
| PostgreSQL/S3 | 事务/对象 | fencing/CAS/ABI rules |

页脚：无 vendoring，无必须维护的上游 Fork。

## 12. 开源与路线图

- MIT、SBOM、第三方 license、ADR、Schema/OpenAPI、Issue/PR、兼容矩阵
- 已完成：真实 GPU process recovery + KEDA control plane + Cloud Lite
- 已完成：OpenAI-compatible Adapter 协议闭环，可接现成 vLLM
- 下一门：固定 vLLM/模型 revision 做真实 GPU 认证，并完成一个拓扑内的 GPU Pod 和 live A/B/C
- 收尾：**State 长期属于用户，GPU 只在需要时出现。**

## 视觉规则

- 所有数字旁标 `measured`、`simulation` 或 `estimated`；
- GPU 与 KEDA 两页使用不同底色，避免观众误以为同一次实验；
- 不出现“大幅提升”“生产可用”等无证据词；
- 演示失败时直接回放带 checksum 的原始产物，不用剪辑伪造曲线。
