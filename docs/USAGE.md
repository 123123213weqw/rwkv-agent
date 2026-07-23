# 使用指南

## 1. 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[realtime,dev]'
```

可选依赖：

```bash
pip install -e '.[model]'      # Hugging Face RWKV
pip install -e '.[browser]'    # 少量 JS 页面后备
pip install -e '.[extraction-bench,dev]'  # 静态抽取器固定快照A/B
```

## 2. 本地聊天界面

```bash
rwkv-search --config configs/default.json init
rwkv-search --config configs/default.json serve --host 127.0.0.1 --port 8765 --no-model
```

加载本地 HF RWKV：

```bash
rwkv-search --config configs/default.json serve \
  --host 127.0.0.1 --port 8765 \
  --model /path/to/rwkv-hf --device cuda:0 --dtype fp16
```

## 3. SearXNG

示例只监听本机：

```bash
cd deploy/searxng
# 将 settings.yml 中的示例 secret_key 替换为随机值
# python -c 'import secrets; print(secrets.token_hex(32))'

docker compose up -d
curl 'http://127.0.0.1:8888/search?q=python&format=json'
```

主程序通过 `realtime_search.searxng_url` 连接它。SearXNG 负责聚合 Discovery，不负责网页正文抽取或答案生成。

## 4. 精确发现配置

`configs/benchmark.json` 展示了完整实验开关：

```json
{
  "candidate_admission_enabled": true,
  "source_channels_enabled": true,
  "domain_pivot_enabled": true,
  "domain_pivot_max_domains": 2,
  "one_hop_link_expansion_enabled": true,
  "one_hop_max_links": 8
}
```

请先跑 Benchmark。不要仅因为功能可用就直接覆盖生产配置。

## 5. G1I/P4 查询生成

原生适配器需要 RWKV G1I 权重及对应本地运行时：

```bash
PYTHONPATH=src python bench/generate_p4_queries.py \
  --model /path/to/model.pth \
  --runtime-dir /path/to/rwkv-runtime \
  --output bench/runs/p4_queries.jsonl \
  --summary bench/runs/p4_queries_summary.json
```

生成格式：

```xml
<tool_call>{"name":"web_search","arguments":{"query":"Python latest stable version"}}</tool_call>
```

运行时使用 greedy 解码，并在完整 `</tool_call>` 处停止。解析失败时不猜测或执行半截工具调用。

## 6. 联网 Benchmark

```bash
PYTHONPATH=src python bench/run_realtime_retrieval_bench.py \
  --config configs/benchmark.json \
  --bench bench/realtime_web_retrieval.jsonl \
  --output bench/runs/retrieval.jsonl \
  --summary bench/runs/retrieval_summary.json
```

使用已生成的模型查询：

```bash
PYTHONPATH=src python bench/run_realtime_retrieval_bench.py \
  --config configs/benchmark.json \
  --model-queries bench/runs/p4_queries.jsonl
```

## 7. 固定网页抽取 Benchmark

首次抓取原始快照（不会写入Git）：

```bash
PYTHONPATH=src python bench/run_web_extraction_bench.py \
  --capture --capture-only
```

离线比较同一批字节：

```bash
PYTHONPATH=src python bench/run_web_extraction_bench.py \
  --repeat 3 \
  --extractors current,hybrid_fast,trafilatura,justext,readability,resiliparse
```

`data/web-extraction-bench/`保存网页快照，`bench/runs/`保存本地逐例结果，两者均被Git忽略。可公开摘要必须只包含指标、失败类型、长度和哈希。

`hybrid_fast`直接复用实时链路的`src/rwkv_search/realtime/hybrid_extractor.py`。安装`realtime`可选依赖后，HTML正文默认使用Resiliparse快速抽取和Trafilatura轻量元数据；仅在通用质量信号命中时运行完整Trafilatura兜底。缺少可选依赖时会安全降级到原抽取实现。接入前冻结结果见`bench/baselines/web_extraction/hybrid-fast-v1/`。
