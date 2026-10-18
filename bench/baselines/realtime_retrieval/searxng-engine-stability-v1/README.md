# SearXNG Engine Stability v1

This baseline isolates each engine exposed by the RTX 4080 SearXNG instance. It does not change the running SearXNG settings and does not fetch result pages or generate answers.

## Method

- General engines (`360search`, `mwmbl`): the same 50 Chinese/English cases, frozen P4 query only, two independent repetitions.
- Repository engine (`github`): the five repository-release cases, two repetitions.
- Science engine (`arxiv`): the academic-paper case, two clean repetitions.
- Requests are sequential and paced. Expected domains and URL patterns are used only after retrieval to score results; they are never injected into the query.
- Full result URLs remain in ignored `bench/runs/` files. Public JSON retains metrics and per-case booleans only.

## Result

| Channel | Engine | Stable non-empty | Stable Domain@10 | Stable Target@20 | Decision |
|---|---:|---:|---:|---:|---|
| General zh | mwmbl | 20% | 16% | 4% | fallback only |
| General en | mwmbl | 92% | 20% | 4% | provisional |
| General zh/en | 360search | 0% | 0% | 0% | reject |
| Repositories (5 cases) | github | 20% | 20% | 0% | reject for current query path |
| Science (1 case) | arxiv | 100% | 100% | 100% | provisional, explicit science only |

The current four-engine pool is not sufficient for production-quality general search. `mwmbl` is operational for English but has low official-source recall; no enabled engine passes a useful Chinese general-search gate. A separate, explicitly authorized experiment should evaluate Bing, Google, DuckDuckGo, Brave, Startpage and Qwant before changing SearXNG configuration.

No production or SearXNG configuration was changed.
