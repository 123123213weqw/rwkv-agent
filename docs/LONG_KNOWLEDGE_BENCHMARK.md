# Long-term knowledge corpus and retrieval benchmark

This benchmark evaluates page-level retrieval from the isolated FineWiki indexes. It does not run web
discovery, page fetching, answer generation, or the production chat router.

## Corpora

- `HuggingFaceFW/finewiki`, `zhwiki`, August 2025 snapshot: the frozen Chinese baseline.
- `HuggingFaceFW/finewiki`, `enwiki`, August 2025 snapshot: the English expansion built by milestone 5A.
- License: Wikipedia content is CC BY-SA 4.0 / GFDL; the processed FineWiki release is CC BY-SA 4.0.

FineWiki is preferred over a generic web crawl because it renders templates, retains headings, tables,
math and infobox metadata, and excludes redirects, disambiguation pages and reference-like tails. Chinese
and English are kept in separate physical indexes so identical Wikipedia page IDs cannot collide.

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
