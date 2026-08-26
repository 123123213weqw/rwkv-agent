FROM python:3.12.11-slim-bookworm
COPY --chmod=0755 mock_worker.py /usr/local/bin/mock_worker.py
COPY --chmod=0755 drain.py /usr/local/bin/rwkv-statepool-drain
USER 65532:65532
ENTRYPOINT ["/usr/local/bin/mock_worker.py"]
