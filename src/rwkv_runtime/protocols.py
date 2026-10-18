"""Structural contracts shared by serving runtimes and scheduler backends."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol


class TokenizerProtocol(Protocol):
    def encode(self, text: str) -> Sequence[int]: ...

    def decode(self, token_ids: list[int]) -> str: ...


class StatePoolProtocol(Protocol):
    @property
    def free(self) -> int: ...


class SchedulerRequestProtocol(Protocol):
    remaining: int
    seen_tokens: int
    logits: Any


class SchedulerProtocol(Protocol):
    pool: StatePoolProtocol

    def admit(self, request_id: str, token_ids: Sequence[int]) -> Any: ...

    def request(self, request_id: str) -> SchedulerRequestProtocol: ...

    def prefill(self, request_ids: Sequence[str] | None = None) -> Mapping[str, Any]: ...

    def prefill_round(
        self,
        request_ids: Sequence[str] | None = None,
    ) -> Mapping[str, Any]: ...

    def continue_many(
        self,
        rows: Sequence[tuple[str, Sequence[int]]],
    ) -> Mapping[str, Any]: ...

    def install_continuations(
        self,
        rows: Sequence[tuple[str, Sequence[int]]],
    ) -> Sequence[SchedulerRequestProtocol]: ...

    def fork(
        self,
        parent_request_id: str,
        child_request_ids: Sequence[str],
    ) -> Sequence[Any]: ...

    def sample_next(self, request_ids: Sequence[str]) -> Mapping[str, int]: ...

    def advance_tokens(self, tokens_by_request: dict[str, int]) -> None: ...

    def release(self, request_id: str) -> None: ...

    def export_state(
        self,
        request_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]: ...

    def import_state(
        self,
        request_id: str,
        manifest: Mapping[str, Any],
        tensors: Mapping[str, Any],
    ) -> Any: ...

    def metrics(self) -> dict[str, Any]: ...
