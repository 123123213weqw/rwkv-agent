# syntax=docker/dockerfile:1.7
FROM rust:1.95-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY web ./web
RUN cargo build --locked --release \
    -p rwkv-statepool-cloud-plugin \
    -p rwkv-agent-server

FROM debian:bookworm-slim AS runtime-base
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 65532 statepool \
    && useradd --system --uid 65532 --gid statepool --home-dir /nonexistent statepool
USER 65532:65532

FROM runtime-base AS statepool-plugin
COPY --from=build /src/target/release/rwkv-statepool-cloud-plugin /usr/local/bin/
ENV RWKV_STATEPOOL_HOST=0.0.0.0 \
    RWKV_STATEPOOL_PORT=8130 \
    RWKV_STATEPOOL_STATE_DIR=/var/lib/statepool/states
EXPOSE 8130
VOLUME ["/var/lib/statepool/states"]
ENTRYPOINT ["/usr/local/bin/rwkv-statepool-cloud-plugin"]

FROM runtime-base AS agent-controller
COPY --from=build /src/target/release/rwkv-agent-server-rs /usr/local/bin/
ENV RWKV_AGENT_HOST=0.0.0.0 \
    RWKV_AGENT_PORT=18120
EXPOSE 18120
ENTRYPOINT ["/usr/local/bin/rwkv-agent-server-rs"]
