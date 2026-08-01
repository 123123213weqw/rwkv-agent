# No-Tavily self-hosted retrieval candidate

This history starts with Dogpile on a local SearXNG instance and finishes with
independent Dogpile and Naver lanes merged locally. It does not call Tavily and
does not use 360 Search.

## Result

On the 50-case bilingual realtime retrieval set, a two-repetition engine test
gave 94% stable Domain Recall@10 and 68% stable Target Page Recall@20. A
simultaneous end-to-end retrieval A/B then compared the same P4+Raw queries,
structured source adapters, local FineWiki discovery, admission logic and page
fetcher. Only the primary general discovery source changed:

| Metric | Direct Bing | SearXNG Dogpile |
| --- | ---: | ---: |
| Candidate Domain Recall@10 | 52% | **98%** |
| Candidate Target Page Recall@20 | 10% | **72%** |
| Result Domain Recall@10 | 46% | **68%** |
| Result Target Page Recall@20 | 6% | **36%** |
| Non-empty results | 98% | **100%** |
| Garbage result rate | 0.42% | **0%** |
| Mean total latency | **4.98 s** | 8.25 s |
| P95 total latency | **12.21 s** | 14.52 s |

The paired exact-page result was 16 wins, 1 loss (`p=0.00027466`). Candidate
quality is no longer the main realtime bottleneck. Fetch success dropped from
73.64% to 54.50%, so the next optimization must improve scheduling, extraction
and Evidence conversion rather than adding more search engines.

## Blocked-origin resilience

Many official sites were already present in the candidate set but returned a
timeout or HTTP 403 from the experiment host. A second paired test therefore
kept Discovery candidates identical and changed only the failure path:

- a search-result excerpt is accepted only after an access/network failure;
- the excerpt must pass candidate-score, entity-coverage and length gates;
- Chinese excerpts use a character-density-aware threshold;
- missing URLs, CAPTCHA/login text and other garbage page shapes are rejected;
- the Evidence prompt labels the excerpt as limited evidence because the
  origin page was not fetched;
- full-page fetch success and excerpt fallback remain separate metrics.

| Metric | Full-page only | + labeled excerpt fallback |
| --- | ---: | ---: |
| Candidate Domain Recall@10 | 94% | 94% |
| Candidate Target Page Recall@20 | 70% | 70% |
| Result Domain Recall@10 | 70% | **90%** |
| Result Target Page Recall@20 | 38% | **46%** |
| Non-empty results | 98% | **100%** |
| Garbage result rate | 0% | 0% |

Result-domain recall produced 10 paired wins and no losses
(`p=0.00195312`). Exact-page recall improved by 8 points in the final filtered
run, but that 50-case comparison was not significant (`p=0.125`). A preceding
replication produced 8 exact-page wins and no losses (`p=0.0078125`), while
also revealing one CAPTCHA-bearing excerpt; the final generic filter removed
that failure.

The remaining gap was made observable: strong official pages could be present
in Discovery but fall outside the eight-page fetch budget, and a restrictive
same-domain cap could then discard a fetched exact page. Two
same-candidate A/B tests addressed those stages without changing Discovery:

| Accepted change | Candidate Domain@10 | Candidate Target@20 | Result Domain@10 | Result Target@20 |
| --- | ---: | ---: | ---: | ---: |
| Display-order fetch schedule | 94% | 76% | 90% | 56% |
| Candidate-confidence schedule | 94% | 76% | **96%** | **64%** |
| Same-domain limit 2 | 94% | 76% | 96% | 66% |
| Same-domain limit 3 | 94% | 76% | 96% | **72%** |

The scheduling comparison kept all 50 candidate lists identical and produced
four exact-page wins with no losses under the corrected metric. The domain-limit
comparison also kept all candidates identical and produced three wins with no
losses. Both accepted arms retained 100% non-empty results and 0% garbage.
Sequential run latency is not used as a causal claim because the later arm could
benefit from shared Discovery caches.

A later full 50-case paired run tested the remaining final-ranking loss by
changing only the same-domain limit from three to four. Candidate lists were
again identical for all cases. Result Target Page Recall@20 improved from 72%
to 76% with two wins and no losses, while Result Domain Recall@10 stayed at
98%, non-empty results stayed at 100%, and garbage stayed at 0%. The recovered
pages were the PostgreSQL release announcement and Rust's official latest-release
page. This small win count is not claimed as statistically significant; the
change is accepted because it adds no Discovery request or page GET and removes
two generic source-diversity truncations. A separate URL-specificity score prior
produced zero wins and zero losses on all 50 cases, so that code was reverted.

Target Page Recall now treats slash-delimited benchmark patterns as path-segment
sequences after optional locale prefixes. Thus `/newsroom/` matches
`/cn/newsroom/...`, but not `/newsroom-malware`. Re-scoring the immutable traces
with this corrected semantics raised Candidate Target Page Recall@20 from the
previously reported 70% to 76%; it did not alter runtime results.

In the latest accepted paired run, 41 of 50 cases reached a matching exact page:
nine misses were not discovered and no exact page was lost in final ranking.
Discovery is live and therefore varies between runs; the individual paired A/B blocks
above remain the causal evidence for each accepted change. No exact-page miss
was attributed to the fetch schedule. A broader domain-pivot plus one-hop experiment
increased requests from 1.8 to 3.13 per case without a final exact-page gain, so
it was rejected rather than enabled by default. The next bottleneck is precise
Discovery for the unresolved cases, not Tool Call formatting, fetch scheduling,
or a missing Tavily key.

## Gold-label audit

The original 50-case file remains immutable, but dynamic official sites do not.
Primary-source re-verification found nine outdated or incomplete target labels:
llama.cpp moved to `ggml-org`; ChinaMoney changed its LPR routes; national
infectious-disease summaries moved to China CDC; the NMC typhoon interface,
Moutai issuer filings, Tesla IR assets and the NHC no-active-storm outlook were
not represented by the frozen paths; the official USGS Significant Earthquakes
page was also missing as a valid exact answer surface. These are label
migrations or incomplete answer sets, not retrieval algorithm failures.

`bench/realtime_web_retrieval_audited_v2.jsonl` keeps every query and policy
field unchanged and revises only expected official domains or target paths for
those nine cases. Its manifest records old/new labels, reasons, verification
URLs and both dataset hashes. Re-scoring the same accepted runtime output gives:

| Metric | Frozen v1 labels | Audited v2 labels |
| --- | ---: | ---: |
| Candidate Domain@10 | 100% | 100% |
| Candidate Target@20 | 82% | **98%** |
| Result Domain@10 | 100% | 100% |
| Result Target@20 | 82% | **96%** |
| Non-empty results | 100% | 100% |
| Garbage result rate | 0% | 0% |

No query, candidate, fetched page or result order changed. Under audited v2,
only the State Council AI-policy case is still a true Discovery miss. The China
CDC page is discovered but is the single remaining candidate-to-Evidence drop.
Future live comparisons should report both frozen-v1 continuity and audited-v2
current validity rather than fitting the retriever to obsolete paths.

## Structured API request budget

GitHub Discovery previously called the profile, owner-repository list, latest
release and latest-commit endpoints after every repository search, regardless of
what the question requested. The source adapter now selects at most three of
those endpoint capabilities from the query and caches repository search
separately. This is a declarative source contract, not topic routing.

A four-case live experiment covered release, commit activity, founder plus
projects plus activity, and repository-root requests. Calls fell from 20 to 9
(55%) while all four requested evidence stages remained present. The control
recorded one unrelated endpoint timeout and the candidate recorded none. The
arms were sequential, so latency and the transient timeout are not treated as
causal evidence; the deterministic request count and stage preservation are.

Provider activation was narrowed at the same time: a generic software-release
question no longer calls GitHub REST merely because the capability description
contained “release”. GitHub, repository, code-hosting or equivalent source
shape must be present; general SearXNG Discovery still handles the question.
Across the frozen 50 cases and both P4 and raw query lanes, GitHub activation
fell from 21/100 lanes to 9/100. Planned detail calls fell from an upper bound of
84 to 14 before cache reuse. The nine retained lanes are exactly the five
GitHub/repository cases in Chinese and English.

## Planner/runtime parity and extraction fallback

Frozen P4 plans are now executed with the same constraint semantics as the
runtime. A plan's `effective_query` takes precedence; legacy rows that only
contain `model_query` are revalidated. Four old 13.3B rows had invented
calendar years that were absent from the user request. Falling back to the raw
question reduced their average Discovery requests from 2 to 1 without reducing
Candidate Domain@10 or Candidate Target@20 in the four-case diagnostic. This
small diagnostic is not used to replace the frozen 50-case quality score.

The full current 13.3B plan set exposed a weakness in that all-or-nothing
fallback: 20 of 50 queries contained an absolute date, fiscal year, quarter or
numeric version that was absent from the user's request. Throwing away the
whole query also discarded useful English translation, entity focus and source
wording. The accepted guard now deletes only those introduced absolute terms;
site-operator expansion, excessive length, empty output or another unresolved
violation still falls back to the original question.

| Metric | Whole-query raw fallback | Deletion-only repair |
| --- | ---: | ---: |
| Candidate Domain@10 | 100% | 100% |
| Candidate Target@20 | 72% | **82%** |
| Result Domain@10 | 100% | 100% |
| Result Target@20 | 72% | **82%** |
| Non-empty results | 100% | 100% |
| Garbage result rate | 0% | 0% |

The paired run produced five exact-page wins and no losses (`p=0.0625`),
recovering Node.js, PyTorch, PostgreSQL, Apple Newsroom and Microsoft investor
relations cases. It adds no model call, Discovery request or page GET. Latency
is not treated as causal because the repaired arm intentionally issued
different query text. The normal Agent path now passes both the original user
question and the model-authored Tool query into this guard; direct tool calls
that have no original question remain compatible. A two-case live 13.3B Agent
smoke returned correct official Node.js and Microsoft evidence. Production
services were not switched or restarted.

A fresh 50-case run then reproduced the next live bottleneck: 76% Candidate
Target@20 became only 58% Result Target@20 because many fetched shell/JS pages
yielded no extractable body. A same-candidate A/B shared both Discovery
candidates and successful page-response bytes, then changed only whether a
validated SERP excerpt could be used after extraction failure:

| Metric | Network/access fallback only | + extraction-failure fallback |
| --- | ---: | ---: |
| Candidate Domain@10 | 98% | 98% |
| Candidate Target@20 | 76% | 76% |
| Result Domain@10 | 90% | **94%** |
| Result Target@20 | 58% | **64%** |
| Non-empty results | 100% | 100% |
| Garbage result rate | 0% | 0% |

All 50 candidate lists were identical. The candidate produced three exact-page
wins and no losses, recovering Rust, Go and NVIDIA investor-relations pages.
The same length, candidate-confidence, entity-coverage and garbage-page gates
remain in force; the evidence stays labeled as a limited search-result excerpt.
No additional search request or page GET is introduced. Sequential latency and
origin fetch-success counts are not treated as causal because failed network
requests are not cached between arms.

The remaining fallback misses showed a second generic failure mode: a precise
official page could have a high composite admission score while literal entity
overlap was only one third. A second 50-case A/B therefore allowed a composite
score of at least `0.42` to satisfy the entity gate, without changing any other
length, quality or garbage filter. All candidate lists again remained identical:

| Metric | Strict entity overlap | + composite-confidence bypass |
| --- | ---: | ---: |
| Result Domain@10 | 96% | **98%** |
| Result Target@20 | 70% | **74%** |
| Non-empty results | 100% | 100% |
| Garbage result rate | 0% | 0% |

This recovered the Chinese Go release page and English Rust release page, for
two exact-page wins and no losses. The win count is too small for a statistical
significance claim; it is accepted as a zero-request, zero-garbage generic
confidence fusion rather than a domain-specific exception.

## Independent engine pool and Evidence conversion

Sending Dogpile and Naver in one SearXNG request was rejected because one slow
upstream could stall the entire response. The accepted implementation sends one
bounded request per engine, lets either lane fail independently, caches each
lane separately and merges canonical URLs with local reciprocal-rank fusion.
On the audited 50-case set, the paired Discovery comparison was:

| Metric | Dogpile | Dogpile + Naver fanout |
| --- | ---: | ---: |
| Candidate Domain@10 | 100% | 100% |
| Candidate Target@20 | 92% | **100%** |
| Non-empty results | 100% | 100% |
| Garbage result rate | 0% | 0% |
| Mean latency | **5.07 s** | 5.38 s |
| P95 latency | 10.08 s | **9.34 s** |

Exact-page candidate recall produced four wins and no losses. The paired run's
final exact-page score stayed flat because the unchanged fetch budget could
still be consumed by commentary pages. A generic primary-source quota now
reserves half of that same budget for regulator, filing, paper, official-doc and
repository page shapes whenever the request explicitly requires primary or
official sources; it does not add a page GET.

GitHub, Crossref and MediaWiki already return bounded authoritative text in
their structured responses. The final change reuses that text as explicitly
labeled `structured_api` Evidence instead of making answer quality depend on a
second scrape of the same URL. The complete accepted-stack verification reused
14 structured records while keeping network attempts at 388.

| Current audited-v2 verification | Result |
| --- | ---: |
| Candidate Domain@10 | **100%** |
| Candidate Target@20 | **100%** |
| Result Domain@10 | **100%** |
| Result Target@20 | **100%** |
| Non-empty results | **100%** |
| Garbage result rate | **0%** |
| Mean / P95 latency | 5.66 s / 8.89 s |

This is one complete live verification, not a claim that live network recall is
permanently deterministic. The paired blocks above remain the causal evidence;
the next quality gate must use unseen Fresh-Web cases rather than tuning further
on this saturated 50-case set.

## Decision

- Use independent Dogpile and Naver lanes through self-hosted SearXNG as the
  default configured general engine pool.
- Keep direct Bing HTML only as a bounded fallback.
- Keep GitHub, MediaWiki and Crossref as independent structured adapters.
- Reuse bounded structured-adapter text as `structured_api` Evidence instead
  of requiring a redundant origin-page GET.
- Use the labeled excerpt fallback for origins that cannot be fetched or yield
  no extractable body, while keeping its lower evidence level visible to RWKV.
- Schedule the bounded network fetch budget by generic candidate confidence,
  while keeping cached Evidence outside that budget.
- Keep up to four final pages from one domain so official indexes, release
  lists, current documentation and an exact article can coexist.
- Delete model-introduced absolute dates, quarters and versions when that is the
  only query violation; otherwise fall back to the original user question.
- Do not require Tavily and do not enable 360 Search.
- This changes repository defaults only. It did not deploy or restart the
  production RWKV Agent service.

Engine availability remains network- and region-dependent. Operators should
rerun `bench/run_searxng_engine_bench.py` on their deployment host before a
production rollout.

For stable multi-hop knowledge, a separate deterministic 83-case FRAMES A/B
removed standalone calendar years from local-index queries while preserving
embedded identifiers. Candidate Target Page Recall@20 improved from 43.37% to
56.63% (11 wins, 0 losses), and Result Target Page Recall@20 improved from
31.33% to 43.37% (10 wins, 0 losses), with a 17 ms mean latency change.
