# “智算云”赛道材料索引

项目名：**RWKV State Agent + StatePool Cloud Plugin**

统一叙事：

> 助手常驻，但 GPU 不必常驻；每个人长期拥有一个 AI 助手，而不需要长期占用一张 GPU。

本目录中的“已实现/已验证/待验证”必须和
[`docs/RELATED_WORK.md`](../../docs/RELATED_WORK.md) 的 Claim Matrix 一致。

- [`submission-summary.md`](submission-summary.md)：报名摘要与评分映射；
- [`architecture.md`](architecture.md)：一页架构与创新边界；
- [`demo-script.md`](demo-script.md)：答辩演示脚本；
- [`evaluation-protocol.md`](evaluation-protocol.md)：A/B/C 可复现实验；
- [`evidence-register.md`](evidence-register.md)：实测、估算、待测证据台账。

当前代码可以演示：默认关闭兼容、插件握手、Worker 注册/Placement、
Lease/fencing、LocalFS Snapshot/Restore、旧 Lease 拒绝、Drain admission、
Prometheus 指标与部署配置。跨真实 RWKV Worker 的 kill/restore、PostgreSQL、
S3 和 KEDA 集群闭环仍是进入正式答辩前的硬门槛。
