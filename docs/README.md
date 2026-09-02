# Documentation

## Users

- [QUICKSTART.md](QUICKSTART.md): install, configure, start and chat
- [KNOWLEDGE_SETUP.md](KNOWLEDGE_SETUP.md): corpus, model and Elasticsearch download links plus index setup
- [CONFIGURATION.md](CONFIGURATION.md): environment and Web configuration
- [FRONTEND.md](FRONTEND.md): RWKV browser layout, TypeScript client, Task UI and same-origin boundary
- [HTTP_API.md](HTTP_API.md): canonical frontend endpoints, NDJSON streaming and task lifecycle
- [SERVICE_PIPELINE.md](SERVICE_PIPELINE.md): canonical startup, readiness, lifecycle and failure diagnosis
- [DEBUG_TRACE.md](DEBUG_TRACE.md): opt-in local diagnostic capture
- [KNOWN_ISSUES.md](KNOWN_ISSUES.md): defects, limits and stable-release gates
- [RELEASE.md](RELEASE.md): maintainer audit, clean-install and publish checklist

## Architecture and evaluation

- [ARCHITECTURE.md](ARCHITECTURE.md): system layers and resource boundaries
- [CODEMAP.md](CODEMAP.md): active implementation ownership
- [REPOSITORY_SURFACE.md](REPOSITORY_SURFACE.md): retained and removed executable surface
- [RUST_MIGRATION.md](RUST_MIGRATION.md): incremental provider and tooling migration
- [STATEPOOL_CLOUD_PLUGIN.md](STATEPOOL_CLOUD_PLUGIN.md): optional cloud plugin contract, compatibility and current limits
- [statepool-cloud](https://github.com/123123213weqw/statepool-cloud): independent cloud control plane, deployment, adapters and evidence
- [SCHEDULER.md](SCHEDULER.md): state pool and continuous batching
- [AGENT_BENCHMARK.md](AGENT_BENCHMARK.md): fixed evaluation methodology
- [BENCHMARK.md](BENCHMARK.md): retrieval benchmark discipline

Internal owner TODOs, private traces, model files and restricted benchmark cases
are intentionally not part of the public documentation surface.
