# SearXNG V100 Direct Trial v1

This frozen public summary checks whether running SearXNG directly on the V100 proxy host removes instability caused by the earlier temporary SSH proxy tunnel.

- Six frozen P4 queries: three Chinese and three English.
- Sequential requests with a one-second delay.
- Existing local proxy reused without node, rule, or service changes.
- Google, DuckDuckGo, Brave, Startpage, and Qwant were tested.
- No candidate passed Smoke, so the 50-case run was intentionally not started.
- The temporary SearXNG container was removed and the proxy remained healthy.

The result shows that the tunnel was not the primary cause. Upstream blocking, rate limiting, parsing incompatibility, and proxy-exit reputation remain the limiting factors.
