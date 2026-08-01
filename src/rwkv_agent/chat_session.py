"""Direct-chat recurrent State lifecycle.

The durable transcript remains owned by ``MemoryStore``.  This service owns
only the optional opaque Sidecar State cache and its rebuild/fallback rules.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .chat_prompts import (
    CHAT_UNSAFE_STOPS,
    render_direct_answer_prompt,
    render_direct_chat_prefix,
    render_direct_chat_turn,
)
from .chat_state import ChatSessionState, ChatStateCache


class DirectChatSession:
    """Manage one Controller's bounded cache of opaque chat states."""

    def __init__(
        self,
        *,
        model_provider: Callable[[], Any],
        enabled_provider: Callable[[], bool],
        cache: ChatStateCache,
    ) -> None:
        self._model_provider = model_provider
        self._enabled_provider = enabled_provider
        self.cache = cache

    @property
    def enabled(self) -> bool:
        return bool(self._enabled_provider())

    def supports_state(self) -> bool:
        model = self._model_provider()
        return all(
            callable(getattr(model, name, None))
            for name in (
                "state_prefill",
                "state_chat_complete",
                "state_release",
            )
        )

    def release_records(self, records: list[ChatSessionState]) -> None:
        model = self._model_provider()
        for record in records:
            try:
                model.state_release(
                    home_url=record.home_url,
                    owner_id=record.owner_id,
                    state_ids=[record.state_id],
                )
            except Exception:
                # TTL expiry and Sidecar restart both invalidate opaque IDs.
                # The local ownership record is already gone, so release is
                # intentionally best effort.
                self.cache.count("release_failures")
            else:
                self.cache.count("released")

    def invalidate(self, session_id: str) -> None:
        record = self.cache.pop(session_id)
        if record is not None:
            self.release_records([record])

    def discard(self, record: ChatSessionState) -> None:
        cached = self.cache.pop(record.session_id)
        self.release_records([cached or record])

    def complete(
        self,
        message: str,
        *,
        session_id: str,
        history: list[Any],
        context: str,
    ) -> tuple[dict[str, Any], ChatSessionState | None, dict[str, Any]]:
        if not self.enabled:
            return self._stateless_completion(
                message,
                context=context,
                fallback_reason="disabled",
            )
        if not self.supports_state():
            return self._stateless_completion(
                message,
                context=context,
                fallback_reason="unsupported_model_client",
            )

        last_message_id = int(history[-1].id) if history else 0
        record, stale = self.cache.get(
            session_id,
            last_message_id=last_message_id,
        )
        if stale is not None:
            self.release_records([stale])

        reused = record is not None
        rebuilt = False
        failures: list[str] = []
        for attempt in range(2):
            created = False
            if record is None:
                try:
                    record = self._prefill(
                        session_id=session_id,
                        last_message_id=last_message_id,
                        context=context,
                    )
                    created = True
                except Exception as exc:
                    failures.append(type(exc).__name__)
                    break
            try:
                completion = self._continue(
                    record,
                    message=message,
                    created=created,
                )
                self.cache.count("continuations")
                if reused and not rebuilt:
                    self.cache.count("reuses")
                return (
                    completion,
                    record,
                    {
                        "enabled": True,
                        "used": True,
                        "reused": reused and not rebuilt,
                        "rebuilt": rebuilt,
                        "cached": False,
                        "fallback_reason": "",
                        "prefill_history_messages": len(history) if created else 0,
                        "seen_tokens": record.seen_tokens,
                    },
                )
            except Exception as exc:
                failures.append(type(exc).__name__)
                cached = self.cache.pop(session_id)
                self.release_records([cached or record])
                record = None
                if reused and attempt == 0:
                    rebuilt = True
                    self.cache.count("rebuilds")
                    continue
                break

        reason = "state_error"
        if failures:
            reason += ":" + ",".join(failures)
        return self._stateless_completion(
            message,
            context=context,
            fallback_reason=reason,
        )

    def store_completed(
        self,
        record: ChatSessionState | None,
        *,
        assistant_message_id: int,
        reasoning_stripped: bool,
        trace: dict[str, Any],
    ) -> None:
        if record is None:
            return
        if reasoning_stripped:
            trace["cache_reject_reason"] = "hidden_reasoning_was_generated"
            self.discard(record)
            return
        if record.stop_reason in CHAT_UNSAFE_STOPS:
            trace["cache_reject_reason"] = "unsafe_stop_boundary"
            self.discard(record)
            return
        record.last_message_id = int(assistant_message_id)
        evicted = self.cache.put(record)
        trace["cached"] = True
        trace["seen_tokens"] = record.seen_tokens
        if evicted:
            self.release_records(evicted)

    def close(self) -> None:
        self.release_records(self.cache.clear())

    def _stateless_completion(
        self,
        message: str,
        *,
        context: str,
        fallback_reason: str,
    ) -> tuple[dict[str, Any], None, dict[str, Any]]:
        completion = self._model_provider().complete(
            render_direct_answer_prompt(message, context=context),
            max_tokens=256,
        )
        if fallback_reason:
            self.cache.count("fallbacks")
        return (
            completion,
            None,
            {
                "enabled": self.enabled,
                "used": False,
                "reused": False,
                "rebuilt": False,
                "cached": False,
                "fallback_reason": fallback_reason,
                "prefill_history_messages": len(context.splitlines()) if context else 0,
                "seen_tokens": 0,
            },
        )

    def _prefill(
        self,
        *,
        session_id: str,
        last_message_id: int,
        context: str,
    ) -> ChatSessionState:
        owner_id = self.cache.owner_id(session_id)
        state = self._model_provider().state_prefill(
            owner_id=owner_id,
            prompt=render_direct_chat_prefix(context=context),
        )
        self.cache.count("prefills")
        return ChatSessionState(
            session_id=session_id,
            owner_id=owner_id,
            state_id=str(state["state_id"]),
            home_url=str(state["home_url"]),
            last_message_id=last_message_id,
            stop_reason="",
            seen_tokens=int(state.get("seen_tokens") or 0),
        )

    def _continue(
        self,
        record: ChatSessionState,
        *,
        message: str,
        created: bool,
    ) -> dict[str, Any]:
        completion = self._model_provider().state_chat_complete(
            home_url=record.home_url,
            owner_id=record.owner_id,
            state_id=record.state_id,
            input_text=render_direct_chat_turn(
                message,
                continuation=not created,
                previous_stop=record.stop_reason,
            ),
            max_tokens=256,
        )
        returned_state_id = str(completion.get("state_id") or record.state_id)
        if returned_state_id != record.state_id:
            raise RuntimeError("Sidecar changed the chat state ID")
        record.stop_reason = str(completion.get("stop") or "")
        record.seen_tokens = int(
            completion.get("seen_tokens") or record.seen_tokens
        )
        return completion
