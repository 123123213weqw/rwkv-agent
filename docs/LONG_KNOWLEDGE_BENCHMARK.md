# Long-term knowledge corpus and retrieval benchmark

This benchmark evaluates page-level retrieval from the isolated FineWiki indexes. It does not run web
discovery, page fetching, answer generation, or the production chat router.

## Corpora

- [`HuggingFaceFW/finewiki`](https://huggingface.co/datasets/HuggingFaceFW/finewiki),
  [`zhwiki` files](https://huggingface.co/datasets/HuggingFaceFW/finewiki/tree/main/data/zhwiki),
  August 2025 snapshot: the frozen Chinese baseline.
- [`HuggingFaceFW/finewiki`](https://huggingface.co/datasets/HuggingFaceFW/finewiki),
  [`enwiki` files](https://huggingface.co/datasets/HuggingFaceFW/finewiki/tree/main/data/enwiki),
  August 2025 snapshot: the English expansion built by milestone 5A.
- License: Wikipedia content is CC BY-SA 4.0 / GFDL; the processed FineWiki release is CC BY-SA 4.0.

FineWiki is preferred over a generic web crawl because it renders templates, retains headings, tables,
math and infobox metadata, and excludes redirects, disambiguation pages and reference-like tails. Chinese
and English are kept in separate physical indexes so identical Wikipedia page IDs cannot collide.

Download commands, storage estimates, Elasticsearch setup and optional E5/BGE
model links are in [Local knowledge service setup](KNOWLEDGE_SETUP.md).

## External test set

Milestone 5A uses the public MIRACL development topics and human relevance judgments for Chinese and
English. MIRACL contains native-speaker relevance annotations over Wikipedia passages. The preparation
script collapses positive passage qrels from `page_id#passage_id` to stable Wikipedia `page_id` values.
It never adds benchmark queries or relevance labels to the search index.

Source: <https://github.com/project-miracl/miracl>
MIRACL repository license: Apache-2.0.

Prepare the fixed development set on a data disk:

```bash
PYTHONPATH=src:. python scripts/prepare_miracl_long_knowledge.py \
  --output-root /data/rwkv-search-bench/long-knowledge/miracl-v1
```

When Hugging Face is available through a mirror:

```bash
PYTHONPATH=src:. python scripts/prepare_miracl_long_knowledge.py \
  --base-url https://hf-mirror.com/datasets/miracl/miracl/resolve/main \
  --output-root /data/rwkv-search-bench/long-knowledge/miracl-v1
```

The output manifest records upstream file hashes, output hash, license, split and per-language counts.

## Project compatibility set

`bench/long_knowledge_compat_v1/cases.jsonl` adds 48 manually reviewed product-regression cases: 24
Chinese and 24 English. It covers entities, aliases, natural questions, descriptions, comparisons,
ambiguity, noisy spacing, cross-language expressions and explicit local misses. This small set diagnoses
known product failure modes; it does not replace MIRACL and is not described as production user logs.

Positive labels use Wikipedia page IDs. Seven `expectation=missing` probes have no qrels. They are excluded
from Hit/Recall/MRR/nDCG and reported separately as expected-missing accuracy, so a missing probe cannot
artificially improve retrieval quality. The manifest freezes the dataset hash and records its CC0-1.0
query/qrel license.

## Index preparation

Install the optional indexing dependencies:

```bash
python -m pip install -e '.[indexing]'
```

For each language, first build latest-revision metadata and row-group-aligned alias sidecars. Then build a
new isolated index; never recreate the frozen Chinese index when testing English.

```bash
PYTHONPATH=src python scripts/build_finewiki_revision_map.py \
  --language en \
  --data-root /data/finewiki/enwiki \
  --output /data/finewiki/revisions-en-v1/duplicate-latest.parquet \
  --report /data/reports/finewiki-en-revisions-v1.json

PYTHONPATH=src python scripts/build_finewiki_aliases.py \
  --language en \
  --data-root /data/finewiki/enwiki \
  --output-root /data/finewiki/aliases-en-v1 \
  --report /data/reports/finewiki-en-aliases-v1.json

PYTHONPATH=src python scripts/index_finewiki_candidate.py \
  --language en \
  --wikiname enwiki \
  --data-root /data/finewiki/enwiki \
  --aliases-root /data/finewiki/aliases-en-v1 \
  --revision-map /data/finewiki/revisions-en-v1/duplicate-latest.parquet \
  --index rwkv-finewiki-en-full-v1 \
  --limit 0 --recreate
```

## Run

```bash
PYTHONPATH=src:. python -m bench.run_long_knowledge_bench \
  --cases /data/rwkv-search-bench/long-knowledge/miracl-v1/miracl_long_knowledge_dev_v1.jsonl \
  --endpoint http://127.0.0.1:19220 \
  --index rwkv-finewiki-zh-full-v1 \
  --language zh \
  --channel-size 100 --limit 10 \
  --output /data/runs/finewiki-zh-miracl-dev.jsonl \
  --summary /data/runs/finewiki-zh-miracl-dev-summary.json
```

The summary reports:

- Hit@1/5/10
- macro Recall@1/5/10
- MRR@10
- nDCG@10
- empty-result rate
- P50/P95 latency
- qrel page coverage in the target index
- retrieval metrics conditional on at least one relevant page existing in the target index
- expected-missing accuracy for compatibility probes, separately from retrieval metrics

The conditional view is diagnostic only. The unconditional score remains the primary end-to-end measure,
because missing relevant pages are a real corpus-coverage failure.

## Frozen milestone 5A results

The English build completed in the isolated `rwkv-finewiki-en-full-v1` index with 6,498,759 effective
latest-revision pages and 18,898,488 chunks. The source contained 6,614,655 rows; older duplicate
revisions were not indexed. The final primary store size was 60,249,901,240 bytes.

| Benchmark | Language | Cases scored | Qrel page coverage | Hit@10 | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| MIRACL dev | zh | 393 | 86.85% | 43.51% | 28.31% | 30.36% | 24.16% |
| MIRACL dev | en | 799 | 97.23% | 53.69% | 37.26% | 39.93% | 32.84% |
| Project compatibility v1 | zh | 21 | 100% | 76.19% | 73.81% | 64.88% | 65.78% |
| Project compatibility v1 | en | 20 | 100% | 45.00% | 40.00% | 38.75% | 34.75% |

The compatibility set also has three Chinese and four English expected-missing probes. Accuracy was
33.33% and 0% respectively because the lexical candidate index has no calibrated abstention threshold.
The English query-type groups contain at most five positive cases each; descriptive and cross-language
failures are useful regression targets, not statistically broad claims.

Frozen public-safe artifacts:

- `bench/baselines/long_knowledge/finewiki-zh-miracl-dev-v1/`
- `bench/baselines/long_knowledge/finewiki-zh-compat-v1/`
- `bench/baselines/long_knowledge/finewiki-en-miracl-dev-v1/`
- `bench/baselines/long_knowledge/finewiki-en-compat-v1/`
- `bench/baselines/long_knowledge/finewiki-bilingual-v1/`

The bilingual report keeps MIRACL and the project compatibility set separate. It does not average
different language splits, treat the small compatibility subsets as paired equivalents, or compare
latency as a hardware-neutral language metric.

## Frozen milestone 5B hybrid experiment

Milestone 5B first expands the lexical candidate depth to 100 and separates failures into corpus miss,
candidate-recall miss, ranking miss and Top-10 hit. It then compares two independent improvements:

1. A Cross-Encoder reranks only the first 50 lexical candidates.
2. A separate page-level dense index embeds title, aliases, section headings and the lead passage.

The dense index deliberately has one vector per page rather than one vector per chunk. Elasticsearch uses
384-dimensional cosine `int8_hnsw` vectors. Lexical and dense Top-100 results are combined with reciprocal
rank fusion (`k=60`, equal weights), then the Cross-Encoder reranks the first 50 fused pages.

| Benchmark | Language | Lexical Hit@10 | Lexical rerank | Dense | Equal RRF | Equal RRF + rerank |
|---|---|---:|---:|---:|---:|---:|
| MIRACL dev | zh | 43.51% | 50.13% | 61.58% | 65.14% | **67.18%** |
| MIRACL dev | en | 53.69% | 60.45% | 82.23% | 77.97% | **85.11%** |
| Project compatibility v1 | zh | 76.19% | 80.95% | 90.48% | 85.71% | **90.48%** |
| Project compatibility v1 | en | 45.00% | 50.00% | **85.00%** | 75.00% | 80.00% |

For MIRACL Chinese, final Recall@10 rises from 28.31% to 47.78%; for English it rises from 37.26% to
62.43%. Candidate-recall misses fall from 98 to 58 in Chinese and from 258 to 60 in English. Corpus misses
remain unchanged, which correctly distinguishes corpus coverage from retrieval quality.

The Chinese page vector index contains 1,256,291 pages and occupies 5.7 GB. The English index contains
6,498,758 pages and occupies 20.7 GB. The measured embedding plus Cross-Encoder process peaked at 3,292
MiB of GPU memory. Chinese fusion latency is measured in one sequential run. English fusion latency is a
composition of frozen lexical timing and current dense/rerank timing, so it is explicitly an estimate;
no unmeasured parallel latency is claimed.

A maximum Cross-Encoder score threshold of zero rejects six of seven compatibility missing probes. The
same threshold loses 0.51 percentage points of Chinese MIRACL Hit@10 and 2.88 points of English Hit@10.
Seven missing probes are insufficient for production calibration, so the threshold remains disabled.

Public-safe results are frozen in:

- `bench/baselines/long_knowledge/finewiki-hybrid-v1/`

Raw candidates, queries, scores, host paths and model caches are excluded. No production Candidate Index
default, chat router, answer path, service or frontend was changed.

## Frozen milestone 5F Agent Shadow integration

The desktop RWKV Agent now has the 5B retriever and 5C passage hydration behind
an optional `knowledge_search` Shadow. The visible tool response remains the
existing lexical `K1..K5`; the Shadow has one worker, an eight-request bounded
queue, lazy model loading, per-stage telemetry and exception isolation.

The Agent-specific frozen set contains 24 compatibility queries, split evenly
between Chinese and English. No answer model was called:

| Metric | Legacy | Hybrid Shadow |
|---|---:|---:|
| Hit@1 | 15/24 (62.5%) | 15/24 (62.5%) |
| Hit@5 | 18/24 (75.0%) | 21/24 (87.5%) |
| Mean latency | 851.9 ms | 1704.8 ms |
| P95 latency | 2179.9 ms | 3088.7 ms |
| Empty / failure | 0 | 0 |

Chinese Hit@5 improves from 9/12 to 12/12; English remains 9/12. Hybrid wins
six and loses six Top-1 cases, so the integration is accepted only for Shadow
observation. It is not approved as the default Agent retriever.

Public-safe results are frozen in:

- `bench/baselines/long_knowledge/agent-hybrid-shadow-v1/`
