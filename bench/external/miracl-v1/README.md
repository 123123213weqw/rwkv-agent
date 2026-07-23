# MIRACL zh/en development qrels

This directory contains the public Chinese and English MIRACL development topics and qrels, plus a
deterministic page-level conversion for RWKV Search's FineWiki benchmark.

- Upstream: https://github.com/project-miracl/miracl
- Paper: https://doi.org/10.1162/tacl_a_00595
- Upstream repository license: Apache-2.0
- Conversion: positive `page_id#passage_id` qrels are collapsed to Wikipedia `page_id`; negative judgments
  remain in the raw qrels but are not copied into the positive page-level case schema.

The files are evaluation data only. They are never indexed and are not available to query planning or
retrieval at runtime. See `manifest.json` for source and output hashes.
