# Software bill of materials

`statepool.cdx.json` is a CycloneDX 1.5 inventory of:

- Cargo packages from `cargo metadata --locked` generated on `WZU_Server`;
- exact Python packages from `uv.lock`;
- versioned images in `deploy/statepool/compose.yaml`.

Regenerate without running Cargo locally:

```bash
ssh WZU_Server 'export PATH="$HOME/.cargo/bin:$PATH"; \
  cd ~/codex-build/rwkv-agent; \
  cargo metadata --locked --format-version 1 > /tmp/rwkv-agent-cargo-metadata.json'
rsync WZU_Server:/tmp/rwkv-agent-cargo-metadata.json /tmp/
uv run --with pyyaml python scripts/generate_statepool_sbom.py \
  /tmp/rwkv-agent-cargo-metadata.json sbom/statepool.cdx.json \
  --revision <git-commit>
```

The inventory is not a vulnerability scan, image signature or runtime
verification. Registry digests replace version tags only at a release gate.
