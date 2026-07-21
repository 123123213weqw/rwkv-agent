# Local SearXNG

This example binds SearXNG to `127.0.0.1:8888` and enables JSON output.

```bash
# Replace the example secret_key in settings.yml first.
# python -c 'import secrets; print(secrets.token_hex(32))'
docker compose up -d
curl 'http://127.0.0.1:8888/search?q=python&format=json'
```

Engine availability depends on network egress. Keep the HTML discovery adapter as a bounded fallback and benchmark the exact engine set before production use.
