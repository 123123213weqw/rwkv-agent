# “智算云”赛道材料索引

项目名：**RWKV State Agent + StatePool Cloud Plugin**

统一叙事：

> 助手常驻，但 GPU 不必常驻；每个人长期拥有一个 AI 助手，而不需要长期占用一张 GPU。

本目录中的“已实现/已验证/待验证”必须和
[`docs/RELATED_WORK.md`](../../docs/RELATED_WORK.md) 的 Claim Matrix 一致。

- [`submission-summary.md`](submission-summary.md)：报名摘要与评分映射；
- [`technical-whitepaper.md`](technical-whitepaper.md)：技术白皮书与证据边界；
- [`architecture.md`](architecture.md)：一页架构与创新边界；
- [`demo-script.md`](demo-script.md)：答辩演示脚本；
- [`presentation-outline.md`](presentation-outline.md)：12 页答辩 PPT 结构；
- [`evaluation-protocol.md`](evaluation-protocol.md)：A/B/C 可复现实验；
- [`evidence-register.md`](evidence-register.md)：实测、估算、待测证据台账。

当前代码和证据可以演示：默认关闭兼容、插件握手、Worker
注册/Placement、Lease/fencing、PostgreSQL/S3 Snapshot/Restore、旧 Lease
拒绝、RTX 4080 强杀进程后恢复续写、Drain admission、Prometheus 指标，以及
kind/KEDA 0→1→3→0。GPU 数据面与非 GPU Kubernetes 控制面是两个独立证据
切片；A/B/C 已有明确标注的 100 Session 仿真回放，same-topology live GPU
对照和真实 GPU Pod 闭环仍是后续证据门槛。可选 OpenAI-compatible Worker
Adapter 已完成协议/OCI/Helm 审计，可接现成 vLLM；具体 vLLM + 模型 + GPU
尚未实测，材料中只能称 Adapter 已验证，不能称模型已认证。
