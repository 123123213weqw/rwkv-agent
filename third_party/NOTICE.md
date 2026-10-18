# Third-party integration notice

The files in `third_party/COMPONENTS.yaml` describe optional, independently
deployed systems. Their source code is not copied into this repository by the
StatePool work.

An integration directory may contain only original `rwkv-agent` adapter code,
configuration, Kubernetes resources, Helm values, dashboards or tests. The
upstream component retains its own copyright and license. Operators are
responsible for reviewing the license and distribution terms of the exact
images and packages they deploy.

AGPL-licensed MinIO and Grafana are treated as separate optional services and
are not linked into the MIT-licensed Rust binaries. Before publishing a release
bundle, generate an SBOM from the actual locked packages and container images
and reconcile it against `COMPONENTS.yaml`.

No compatibility is implied by listing a component. Only versions marked
`verified` in `COMPATIBILITY.md` have passed the repository's integration
tests.
