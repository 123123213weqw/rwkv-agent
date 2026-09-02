# Release checklist

This checklist is for maintainers preparing a public beta or stable release. It
does not authorize a production rollout.

## 1. Scope

- Freeze one commit and one version.
- Separate production code from `experiments/`, private traces and runtime state.
- Publish only benchmark manifests and aggregate results whose licenses allow it.
- Record unresolved quality, deployment and security limitations in
  [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).

## 2. Automated checks

From the repository root:

```bash
python scripts/check_public_release.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src:. python -m pytest -q
ruff check src bench benchmarks tests scripts
python -m build

cargo fmt --check
cargo test --locked
cargo clippy --all-targets -- -D warnings
cargo build --release --locked
bash cli/tests/cli_smoke.sh
```

The release audit verifies version alignment, required files, executable release
scripts, safe realtime-search defaults, blank credential templates, narrow
Provider entrypoints, and the absence of secrets, user-specific home paths and
Tailscale addresses on the documented public surface.

## 3. Clean-install smoke

1. Install the built Python wheel into a fresh virtual environment.
2. Confirm `rwkv_search`, `rwkv_agent` and `rwkv7_scheduler` import, then verify
   the two Provider entrypoints are `rwkv-g1i-sidecar` and
   `rwkv-agent-data-plane`.
3. Run `rwkv-agent --version` and the mock Controller smoke test.
4. Verify the removed Python Controller, `rwkv-search` product CLI and service
   wrapper are absent from the wheel and source release.
5. Start an isolated split stack or verify structured fail-closed readiness when
   the external Providers are intentionally unavailable.

## 4. Artifacts

Allowed:

- source code, configuration templates and documentation;
- aggregate benchmark summaries and reproducibility metadata;
- Python source distribution/wheel and Rust CLI binary checksums.

Not allowed:

- model weights, API keys, populated `.env` files or session databases;
- fetched webpage bodies, complete private traces or machine-specific paths;
- restricted benchmark cases, especially ALCE-derived data without a resolved
  redistribution license.

## 5. Publish gate

Before tagging:

- CI is green on the exact release commit;
- `CHANGELOG.md` and versions match the tag;
- a maintainer has reviewed `KNOWN_ISSUES.md` and benchmark claims;
- the default server binds to loopback unless authentication/TLS are provided by
  an explicitly documented gateway;
- no experimental profile is presented as the stable default.

The current version is `0.3.0-beta.2`; the stable `0.3.0` quality gates remain
listed in [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).
