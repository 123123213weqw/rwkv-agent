# Security

请不要在公开 Issue 中提交密钥、服务器地址、私有模型路径或完整网页抓取内容。

实时抓取默认阻止私网和环回目标，限制重定向、压缩前后响应体大小、全局并发和单域并发。部署到公网前仍应增加身份验证、请求限流、出站网络策略和独立运行账户。

`v0.3.0-beta.2`的Agent HTTP服务没有内置身份验证、TLS或可信多租户授权。请求中的`owner_id`只用于
本地隔离，不是身份认证。官方示例只监听`127.0.0.1`。远程访问应使用SSH/VPN或具有身份验证、限流和TLS的
反向代理，不能直接公开8120或8122端口。

Debug Trace与其API默认关闭。显式启用`full`会保存Prompt、模型输出、Tool参数与结果以及Command输出；这类
目录必须保持本机Owner-only权限，不得提交、发布、用于训练或暴露到非Loopback接口。

发现安全问题时，请提供最小化复现，不要附带真实API Key、完整Session数据库、模型私有路径、网页正文或
未脱敏Token Trace。

StatePool Cloud Plugin 同样没有内置公网身份认证或租户认证。`owner_id` 是协议
隔离字段，不是可信身份；8130 端口只能监听 loopback、受限容器网络或 mTLS/
VPN 后的私网。LocalFS State 可能包含可恢复的模型上下文，应按私密数据处理，
限制目录权限、备份和日志，禁止提交。

生产远程 State 在实现加密、可信 Worker 身份、PostgreSQL/S3 权限、密钥轮换和
审计前不得启用。发现 Lease 绕过、fencing/CAS 失效、owner 越权、State URI
逃逸、checksum 绕过或 Drain 数据丢失时，请使用 GitHub Private Security
Advisory，不要公开真实 State payload。
