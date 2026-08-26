# 报名摘要与评分映射

## 200 字摘要

RWKV State Agent 是一个本地优先、State 原生的个人助手。StatePool Cloud
Plugin 在不改变原默认行为的前提下，把长期 Agent recurrent State 变成可选
的云原生调度对象：本地电脑作为隐私根和边缘 Worker，云 GPU 作为弹性
Worker；控制面依据隐私、State 位置、模型/Tokenizer/State ABI、队列、SLO、
恢复代价和 GPU 价格做可解释 Placement。系统通过单写 Lease、单调 fencing
token、不可变 Snapshot、版本 CAS 和安全 Drain 防止双写；通过 KEDA、
Prometheus/Grafana、可选 HAMi/AIBrix 适配实现弹性与 FinOps。核心主张是：
**助手常驻，但 GPU 不必常驻。**

## 痛点

长期个人 Agent 的请求稀疏但上下文有价值。Sticky Worker 让 GPU 在用户离开
时空转；Stateless re-prefill 虽可释放 Worker，却在用户回来时重复处理历史。
通用 GPU 调度器看见 Pod/GPU，通用推理平台看见请求/模型/KV cache，但通常
没有看见 owner-scoped、可版本化、可恢复的 RWKV recurrent State。

## 技术方案

```text
Local/Edge/Cloud Worker
    ↕ WorkerCapability + heartbeat + drain
StatePool Cloud Plugin
    ├─ explainable placement
    ├─ Session Lease + fencing + version CAS
    ├─ Hot / Warm / Cold State lifecycle
    ├─ State-aware demand and FinOps metrics
    └─ adapters: PostgreSQL / S3 / KEDA / AIBrix / HAMi
    ↕ versioned HTTP contract
rwkv-agent Controller (plugin disabled by default)
```

## 相对已有方案的创新（30%）

不是重做 Kubernetes、GPU 虚拟化或 LLM Gateway，而是补上长期 Agent State
这一层：

1. `model_id + revision + tokenizer + state_abi` 全等才迁移原始 State；
2. owner-scoped 单写 Lease + 单调 fencing token + version CAS；
3. Hot/Warm/Cold 与 restore cost 进入 Placement；
4. Drain 的安全条件包含在途请求为 0 且脏 State 明确为 0；
5. FinOps 统计 GPU-seconds、Prefill tokens avoided、State I/O、命中层级和
   分币种估算费用。

## 场景落地（30%）

现有主仓库已有真实 RWKV Agent、Sidecar、多 Session State pool 和 AMD/V100
实验。本插件不是纸面新仓库：协议、服务、LocalFS State round trip、部署镜像、
Compose、Helm、KEDA、ServiceMonitor、Dashboard 已进入同一仓库并有远程测试。

正式提交必须补齐三组同协议实验：Sticky、Stateless re-prefill、StatePool，
以及一次兼容 Worker 强杀/恢复闭环。结果必须标记实测或估算。

## 开源治理（20%）

- 主仓库 MIT；第三方组件不 vendoring、不 Fork；
- `third_party/COMPONENTS.yaml` 记录许可证、版本、接入形式和状态；
- JSON Schema/OpenAPI 固化协议，兼容窗口写入 `COMPATIBILITY.md`；
- ADR、贡献指南、安全边界、治理规则、Issue/PR 模板和发布清单齐全；
- 上游只通过协议、镜像、Chart values、Adapter 接入。

## 长期发展（20%）

维护对象限定为 State/session contract、Placement、Lifecycle、FinOps 和适配器
接口，而非维护 Kubernetes/KEDA/HAMi/AIBrix Fork。Roadmap 以证据门槛推进：
LocalFS 单进程 → PostgreSQL/S3 Cloud Lite → 真实 KEDA GPU 闭环 → 可选
AIBrix/HAMi 集成。

## 当前诚实边界

已经实现不等于已经完成比赛最终目标。Albatross live RWKV 的 CPU
Snapshot/Restore 协议边界和确定性续写测试已经完成，但尚未完成
PostgreSQL distributed Lease、S3 adapter、真实 KEDA 集群 0→1→N→0、以及
Worker kill/restore GPU 实测。答辩材料不得把 CPU conformance 或配置存在
写成跨 Worker 云端运行结果。
