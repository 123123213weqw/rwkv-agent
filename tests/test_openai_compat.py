from __future__ import annotations

import json

from fastapi.testclient import TestClient

from rwkv_agent.openai_compat import (
    CHAT_STOP_STRINGS,
    completion_usage,
    normalize_stops,
    openai_finish_reason,
    render_chat_prompt,
)
import rwkv_agent.sidecar as sidecar


class _FakeService:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def complete(
        self,
        prompt,
        stops,
        max_tokens,
        prefix_token_ids=(),
        prefill_chunk_size=sidecar.PREFILL_CHUNK_SIZE,
        event_sink=None,
    ):
        self.calls.append(
            {
                "prompt": prompt,
                "stops": list(stops),
                "max_tokens": max_tokens,
                "prefix_token_ids": list(prefix_token_ids),
                "prefill_chunk_size": prefill_chunk_size,
            }
        )
        if event_sink is not None:
            event_sink({"type": "delta", "text": "你", "delta": "你"})
            event_sink({"type": "delta", "text": "你好", "delta": "好"})
        return {
            "text": "你好",
            "stop_reason": "max_tokens",
            "token_ids": [10, 11],
            "input_tokens": 7,
            "batch_mode": "unified",
            "elapsed_ms": 12.5,
            "queue_ms": 1.25,
        }


def _client(monkeypatch):
    fake = _FakeService()
    monkeypatch.setattr(sidecar, "service", fake)
    monkeypatch.setattr(sidecar, "worker_agent", None)
    return TestClient(sidecar.create_app()), fake


def test_render_chat_prompt_uses_rwkv_roles() -> None:
    prompt = render_chat_prompt(
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Who are you?"},
        ]
    )
    assert prompt == (
        "System: Be concise.\n\nUser: Hello\n\nAssistant: Hi\n\n"
        "User: Who are you?\n\nAssistant:"
    )


def test_chat_prompt_rejects_multimodal_and_non_user_final_message() -> None:
    for messages in (
        [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        [{"role": "assistant", "content": "unfinished"}],
        [{"role": "tool", "content": "result"}],
    ):
        try:
            render_chat_prompt(messages)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid chat message list was accepted")


def test_stop_and_usage_mapping() -> None:
    assert normalize_stops("END", chat=True) == ["END", *CHAT_STOP_STRINGS]
    assert openai_finish_reason("max_tokens") == "length"
    assert openai_finish_reason("</s>") == "stop"
    assert completion_usage(
        {"input_tokens": 3, "token_ids": [1, 2]}
    ) == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }


def test_chat_completions_non_stream_is_openai_compatible(monkeypatch) -> None:
    client, fake = _client(monkeypatch)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": sidecar.MODEL_ID,
            "messages": [
                {"role": "system", "content": "直接回答。"},
                {"role": "user", "content": "你好"},
            ],
            "temperature": 0,
            "max_tokens": 2,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-rwkv-")
    assert body["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "你好"},
            "finish_reason": "length",
        }
    ]
    assert body["usage"] == {
        "prompt_tokens": 7,
        "completion_tokens": 2,
        "total_tokens": 9,
    }
    assert fake.calls[0]["prompt"].endswith("User: 你好\n\nAssistant:")
    assert set(CHAT_STOP_STRINGS).issubset(fake.calls[0]["stops"])


def test_chat_completions_stream_uses_sse_and_usage_chunk(monkeypatch) -> None:
    client, _fake = _client(monkeypatch)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": sidecar.MODEL_ID,
            "messages": [{"role": "user", "content": "你好"}],
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0,
            "max_tokens": 2,
        },
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    payloads = [
        line.removeprefix("data: ")
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert payloads[-1] == "[DONE]"
    chunks = [json.loads(value) for value in payloads[:-1]]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert "".join(
        chunk["choices"][0].get("delta", {}).get("content", "")
        for chunk in chunks
        if chunk["choices"]
    ) == "你好"
    assert chunks[-2]["choices"][0]["finish_reason"] == "length"
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"]["total_tokens"] == 9


def test_completions_supports_stream_and_standard_usage(monkeypatch) -> None:
    client, _fake = _client(monkeypatch)
    response = client.post(
        "/v1/completions",
        json={"model": sidecar.MODEL_ID, "prompt": "User: hi\nAssistant:", "max_tokens": 2},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"]["completion_tokens"] == 2


def test_chat_endpoint_rejects_silently_unsupported_sampling(monkeypatch) -> None:
    client, _fake = _client(monkeypatch)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": sidecar.MODEL_ID,
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.7,
        },
    )
    assert response.status_code == 422
    assert "greedy" in response.json()["detail"]
