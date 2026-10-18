# Local knowledge service setup

`knowledge_search` uses an external Elasticsearch-compatible index. The Agent
Data Plane connects to the configured endpoint; the Rust Server and Provider
processes do not download the corpus, build indexes or start Elasticsearch.

The public repository currently publishes the index builders and benchmark
summaries, but **does not publish a prebuilt FineWiki index snapshot**. Download
the source corpus and build the indexes locally as described below.

## Downloads

| Component | Required | Official source | Approximate download |
|---|---:|---|---:|
| Hugging Face CLI | yes | [CLI documentation](https://huggingface.co/docs/huggingface_hub/guides/cli) | small installer |
| FineWiki Chinese (`zhwiki`) | for Chinese | [dataset](https://huggingface.co/datasets/HuggingFaceFW/finewiki), [files](https://huggingface.co/datasets/HuggingFaceFW/finewiki/tree/main/data/zhwiki) | 5 shards, about 5.5 GB |
| FineWiki English (`enwiki`) | for English | [dataset](https://huggingface.co/datasets/HuggingFaceFW/finewiki), [files](https://huggingface.co/datasets/HuggingFaceFW/finewiki/tree/main/data/enwiki) | 15 shards, about 37.7 GB |
| Elasticsearch 8.17.4 | yes | [Docker image](https://www.docker.elastic.co/r/elasticsearch/elasticsearch%3A8.17.4), [official Docker guide](https://www.elastic.co/guide/en/elasticsearch/reference/8.17/docker.html) | container image plus index storage |
| Multilingual E5 Small | Hybrid Shadow only | [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) | about 493 MB for selected Transformers files |
| BGE Reranker v2 M3 | Hybrid Shadow only | [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3) | about 2.3 GB |
| MIRACL zh/en | benchmark only | [project and downloads](https://github.com/project-miracl/miracl) | depends on selected files |

FineWiki uses August 2025 Wikipedia snapshots and is licensed under CC BY-SA
4.0/GFDL. Review each upstream model and dataset license before redistribution.

Observed storage for the full indexes used by this project is approximately:

- Chinese lexical index: 12.1 GB;
- English lexical index: 56.1 GB;
- Chinese page-vector index: 5.7 GB;
- English page-vector index: 20.7 GB.

Source Parquet, alias/revision sidecars, checkpoints and temporary index data
are additional. Allow at least 40 GB for a Chinese lexical-only build and about
200 GB of free disk for a bilingual lexical plus dense build.

## 1. Install the download and indexing tools

From the repository root:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
python -m pip install -e '.[indexing,agent,retrieval]'
```

An unauthenticated FineWiki download works, but an `HF_TOKEN` increases Hub
rate limits. Do not put the token in the repository.

## 2. Download FineWiki

Choose a data disk and download only the languages you need:

```bash
export KNOWLEDGE_ROOT=/data/rwkv-knowledge
mkdir -p "$KNOWLEDGE_ROOT"

# Chinese: data lands under $KNOWLEDGE_ROOT/finewiki/data/zhwiki
hf download HuggingFaceFW/finewiki \
  --repo-type dataset \
  --include 'data/zhwiki/*.parquet' \
  --local-dir "$KNOWLEDGE_ROOT/finewiki"

# Optional English: data lands under $KNOWLEDGE_ROOT/finewiki/data/enwiki
hf download HuggingFaceFW/finewiki \
  --repo-type dataset \
  --include 'data/enwiki/*.parquet' \
  --local-dir "$KNOWLEDGE_ROOT/finewiki"
```

Preview the transfer without downloading:

```bash
hf download HuggingFaceFW/finewiki \
  --repo-type dataset \
  --include 'data/zhwiki/*.parquet' \
  --local-dir "$KNOWLEDGE_ROOT/finewiki" \
  --dry-run
```

## 3. Start a local Elasticsearch node

The following loopback-only node is suitable for local development. It disables
Elasticsearch authentication and must not be exposed to another host:

```bash
docker volume create rwkv-knowledge-data
docker run -d \
  --name rwkv-knowledge \
  --restart unless-stopped \
  -m 8GB \
  -p 127.0.0.1:19220:9200 \
  -e discovery.type=single-node \
  -e xpack.security.enabled=false \
  -v rwkv-knowledge-data:/usr/share/elasticsearch/data \
  docker.elastic.co/elasticsearch/elasticsearch:8.17.4

curl -fsS http://127.0.0.1:19220/
```

For production, retain authentication and TLS and follow Elastic's official
Docker deployment guidance instead of using the development command above.

## 4. Build the lexical indexes

Build Chinese metadata, aliases and the lexical index:

```bash
mkdir -p "$KNOWLEDGE_ROOT"/{metadata,aliases,reports}

PYTHONPATH=src python scripts/build_finewiki_revision_map.py \
  --language zh \
  --data-root "$KNOWLEDGE_ROOT/finewiki/data/zhwiki" \
  --output "$KNOWLEDGE_ROOT/metadata/duplicate-latest-zh.parquet" \
  --report "$KNOWLEDGE_ROOT/reports/revisions-zh.json"

PYTHONPATH=src python scripts/build_finewiki_aliases.py \
  --language zh \
  --data-root "$KNOWLEDGE_ROOT/finewiki/data/zhwiki" \
  --output-root "$KNOWLEDGE_ROOT/aliases/zh" \
  --report "$KNOWLEDGE_ROOT/reports/aliases-zh.json"

PYTHONPATH=src python scripts/index_finewiki_candidate.py \
  --language zh \
  --wikiname zhwiki \
  --data-root "$KNOWLEDGE_ROOT/finewiki/data/zhwiki" \
  --aliases-root "$KNOWLEDGE_ROOT/aliases/zh" \
  --revision-map "$KNOWLEDGE_ROOT/metadata/duplicate-latest-zh.parquet" \
  --endpoint http://127.0.0.1:19220 \
  --index rwkv-finewiki-zh-full-v1 \
  --limit 0 --recreate
```

For English, substitute `zh` with `en`, `zhwiki` with `enwiki`, and use index
`rwkv-finewiki-en-full-v1`. `--recreate` deletes an existing index with the same
name; omit it unless a fresh rebuild is intended.

Verify the completed indexes:

```bash
curl -fsS \
  'http://127.0.0.1:19220/_cat/indices?h=health,status,index,docs.count,store.size&s=index'
```

## 5. Connect the Agent

Set the endpoint in `~/.config/rwkv-agent/rwkv-agent.env`:

```bash
RWKV_AGENT_KNOWLEDGE_ENDPOINT=http://127.0.0.1:19220
```

Restart the local Agent after changing its environment, then test:

```bash
rwkv tool knowledge-search "Python是什么"
```

The lexical `K1..K5` path is the current visible knowledge result. Realtime Web
pages are not silently written into this index.

## 6. Optional Hybrid Shadow downloads

The dense E5 plus Cross-Encoder path remains an experimental Shadow and does
not replace visible lexical results. Download only the Transformers files used
by the current implementation:

```bash
mkdir -p "$KNOWLEDGE_ROOT/models"

hf download intfloat/multilingual-e5-small \
  --include '*.json' --include '*.safetensors' --include '*.model' \
  --exclude 'onnx/*' --exclude 'openvino/*' \
  --local-dir "$KNOWLEDGE_ROOT/models/multilingual-e5-small"

hf download BAAI/bge-reranker-v2-m3 \
  --include '*.json' --include '*.safetensors' --include '*.model' \
  --local-dir "$KNOWLEDGE_ROOT/models/bge-reranker-v2-m3"
```

Create the page-vector index after the lexical source index is complete:

```bash
mkdir -p "$KNOWLEDGE_ROOT/checkpoints"

PYTHONPATH=src python scripts/index_finewiki_page_embeddings.py \
  --endpoint http://127.0.0.1:19220 \
  --source-index rwkv-finewiki-zh-full-v1 \
  --target-index rwkv-finewiki-page-e5-small-zh-v1 \
  --model "$KNOWLEDGE_ROOT/models/multilingual-e5-small" \
  --device cuda:0 --fp16 \
  --checkpoint "$KNOWLEDGE_ROOT/checkpoints/e5-zh.json" \
  --recreate
```

Enable Shadow mode only after both the lexical and dense indexes exist:

```bash
RWKV_AGENT_KNOWLEDGE_SHADOW=1
RWKV_AGENT_EMBEDDING_MODEL=/data/rwkv-knowledge/models/multilingual-e5-small
RWKV_AGENT_RERANKER_MODEL=/data/rwkv-knowledge/models/bge-reranker-v2-m3
RWKV_AGENT_RETRIEVAL_DEVICE=cuda:0
RWKV_AGENT_KNOWLEDGE_SHADOW_LOG=/data/rwkv-knowledge/knowledge-shadow.jsonl
```

See [Long-term knowledge benchmark](LONG_KNOWLEDGE_BENCHMARK.md) for the frozen
quality results, MIRACL preparation and benchmark commands.
