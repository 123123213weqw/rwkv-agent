# SearXNG Candidate Engines v1

Six general-search candidates were tested on a temporary SearXNG instance bound to port 8889. The existing port-8888 instance was not edited or restarted.

## Smoke

Google, DuckDuckGo, Brave, Startpage and Qwant all failed with connection timeouts from the RTX 4080 host. They did not enter the full benchmark. Bing was reachable, but SearXNG's default `www.bing.com` endpoint returned an intermediate redirect body and no parsed results. The documented China-host override, `cn.bing.com`, returned valid results.

## Bing full run

The same 50 frozen P4 queries were run twice, sequentially and paced.

| Metric | Bing (`cn.bing.com`) | Mwmbl baseline |
|---|---:|---:|
| Stable non-empty | 100% | 56% |
| Stable Domain Recall@10 | 56% | 18% |
| Stable Target Page Recall@20 | 8% | 4% |
| Average latency | 185.8 ms | 414.8 ms |
| Candidate garbage rate | 16.22% | 0% |

Bing is the clear discovery winner on this host, including 68% Chinese and 44% English stable Domain Recall@10. It is not yet approved for a configuration switch because its unfiltered candidate garbage rate must be measured after the existing candidate-admission layer.

Full URLs remain only in ignored remote `bench/runs/` files. Public JSON contains aggregate metrics and per-case booleans. No production configuration changed.

## V100 existing-proxy trial

A temporary SSH reverse tunnel reused the already-running V100 Clash Verge proxy without changing its node or configuration. DuckDuckGo initially performed well on six smoke cases (5/6 Domain@10 and 3/6 Target@20), but the subsequent paced 50-case run received CAPTCHA/403 responses from the first case and produced 0/100 successful requests across two repetitions. Brave returned 429, Qwant returned CAPTCHA/403, Startpage had connection/parsing errors, and Google produced connection errors or empty parsed results. None passed the stability gate.

The proxy was not reconfigured or rotated. The tunnel and temporary SearXNG instance were removed after the trial. Bing through the direct China endpoint therefore remains the only stable winner in this milestone.
