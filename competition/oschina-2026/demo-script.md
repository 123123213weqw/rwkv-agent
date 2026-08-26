# 8 分钟答辩演示脚本

## 0:00–0:45：一句话和矛盾

打开现有本地个人助手，连续对话两轮，展示同 Session State reuse。

讲述：个人助手需要“长期在”，但个人请求大多数时间不在；把每个助手固定在
GPU 上浪费资源，把 State 丢掉又会重复 Prefill。

屏幕大字：**助手常驻，但 GPU 不必常驻。**

## 0:45–1:30：默认不破坏

展示不带 `--cloud-plugin` 的启动和 readiness：`cloud_plugin.status=disabled`。
强调无插件 HTTP Client、无云依赖，原 API/UI/Sidecar/SQLite 行为不变。

## 1:30–2:30：Worker 能力和 Placement

注册两个完全相同模型身份的 Worker：本地/云，不同队列和价格；依次发送
`local_only`、`cloud_allowed`、Hot affinity 请求。展示 `reason_code`、Worker、
预计排队/恢复/成本。把 revision 改一位，展示被过滤而不是“强行迁移”。

## 2:30–4:00：一致性主秀

1. holder A 对 State v0 获取 Lease，得到 fencing token 1；
2. holder B 同时获取，返回 `lease_held`；
3. A 上传 State，服务重算 SHA-256，原子发布并 CAS 到 v1；
4. 释放 A，B 对 v1 获取 token 2；
5. B restore 得到相同 checksum/bytes；
6. 模拟 A 复活提交，返回 `stale_fencing_token`。

一句解释：版本 CAS 防并发覆盖，fencing 防“过期进程复活”。

## 4:00–5:00：安全缩容

Worker heartbeat 报 `running_requests=1` 或不报告 dirty State，Drain 只能返回
`draining`；只有 `running_requests=0` 且 `unpersisted_state_slots=0` 才返回
`safe_to_stop`。展示 Helm preStop 与 180 秒 grace；说明 KEDA 不负责 State
正确性，StatePool 提供缩容前置条件。

## 5:00–6:00：0→1→N→0 与 Dashboard

在最终集群版演示：无云 Worker 时发 cloud-allowed 请求，
`statepool_pending_requests` 从 0→1，KEDA 拉起第一个 Worker；负载增加后
`estimated_decode_seconds` 驱动 1→N；停止负载并完成 Drain 后 N→0。

如果真实集群闭环尚未完成，答辩时只能展示 YAML/静态检查并明确说“待测”，
不能播放伪造曲线。

Grafana 展示 GPU seconds、Local/Cloud、Hot/Warm/Cold、Restore、避免 Prefill、
State I/O、Lease conflict、分币种 estimated cost。

## 6:00–7:15：A/B/C 结果

同一批 Session、相同模型、相同输出预算：

- A Sticky：最低恢复延迟、最高 GPU idle；
- B Re-prefill：可缩零、重复 Prefill 最大；
- C StatePool：恢复有代价，但避免重复 Prefill并释放空闲 GPU。

只展示 `evaluation-protocol.md` 生成的实测表，并在每列标注 hardware、date、
measured/estimated。

## 7:15–8:00：为什么不是拼装

HAMi 管 GPU，KEDA 管副本，AIBrix/KServe 管通用 serving，S3/PostgreSQL 管
存储；本项目不 Fork 它们。本项目贡献的是长期 Agent State 的 ABI、Lease、
Lifecycle、Placement 和 benchmark。

收尾：**未来不是每人常驻一张卡，而是每人拥有一个可恢复、可迁移、可计费的
AI State。**
