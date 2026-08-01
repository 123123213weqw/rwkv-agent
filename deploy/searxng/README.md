# Local SearXNG

This example binds SearXNG to `127.0.0.1:8888` and enables JSON output.

```bash
# Replace the example secret_key in settings.yml first.
# python -c 'import secrets; print(secrets.token_hex(32))'
docker compose up -d
curl 'http://127.0.0.1:8888/search?q=python&format=json'
```

The checked-in profile enables Dogpile and Naver as independent general-web
lanes; GitHub and arXiv remain available as specialized engines. The Agent
fans the general lanes out separately, merges them with reciprocal-rank fusion,
and keeps a slow lane from blocking a healthy one. It still uses bounded
structured-source adapters and direct Bing HTML only as fallback.

Engine availability depends on network egress. Before production use, run the
same frozen retrieval benchmark against the exact host and engine set instead
of assuming that an engine working in one region will work in another.
