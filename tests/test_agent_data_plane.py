from __future__ import annotations

from dataclasses import dataclass

from rwkv_agent.data_plane import AgentDataPlane


class FakeWeb:
    def execute(self, query: str, *, original_query: str | None = None):
        return {
            "status": "ok",
            "effective_query": query,
            "original_query": original_query,
            "evidence": [
                {
                    "id": "W1",
                    "title": "Example",
                    "content": "ExampleDB is maintained by Example Foundation.",
                    "uri": "https://example.com/db",
                }
            ],
        }

    def close(self) -> None:
        return None


class FakeKnowledge:
    def execute(self, query: str):
        return {"status": "ok", "evidence": [], "query": query}

    def close(self) -> None:
        return None


class FakeLongText:
    def execute(self, text: str, question: str, *, document_name: str):
        return {
            "status": "ok",
            "evidence": [
                {
                    "id": "L1",
                    "title": document_name,
                    "content": f"{question}: {text[:20]}",
                    "uri": "session-text://current#chunk=1",
                }
            ],
        }


@dataclass
class Pasted:
    name: str
    text: str
    chars: int
    sha256: str


class FakeSessionText:
    def __init__(self) -> None:
        self.values: dict[str, Pasted] = {}

    def put(self, session_id: str, text: str) -> Pasted:
        pasted = Pasted("pasted-text", text, len(text), "abc")
        self.values[session_id] = pasted
        return pasted

    def get(self, session_id: str):
        return self.values.get(session_id)

    def health(self):
        return {"active": len(self.values)}

    def close(self) -> None:
        return None


def make_plane() -> AgentDataPlane:
    return AgentDataPlane(
        web=FakeWeb(),
        knowledge=FakeKnowledge(),
        long_text=FakeLongText(),
        session_text=FakeSessionText(),
    )


def test_data_plane_admits_web_and_preserves_original_query() -> None:
    plane = make_plane()
    result = plane.execute(
        "web_search",
        {"query": "ExampleDB maintainer"},
        session_id="s1",
        original_query="Who maintains ExampleDB?",
    )
    assert result["status"] == "ok"
    assert result["original_query"] == "Who maintains ExampleDB?"
    assert [item["id"] for item in result["evidence"]] == ["W1"]
    assert "evidence_admission" in result


def test_data_plane_capture_and_long_text_are_session_scoped() -> None:
    plane = make_plane()
    captured = plane.capture_text("alpha", "secret source")
    assert captured["status"] == "accepted"
    assert plane.text_status("alpha")["active"] is True
    assert plane.text_status("beta")["active"] is False
    result = plane.execute(
        "long_text_qa",
        {"question": "What is it?"},
        session_id="alpha",
    )
    assert result["status"] == "ok"
    assert result["evidence"][0]["id"] == "L1"


def test_data_plane_answer_validation_rejects_unsupported_number() -> None:
    plane = make_plane()
    validation = plane.validate_answer(
        question="Who maintains ExampleDB?",
        answer="ExampleDB version 9.9 is maintained by Example Foundation [W1].",
        evidence=[
            {
                "id": "W1",
                "title": "Example",
                "content": "ExampleDB is maintained by Example Foundation.",
                "uri": "https://example.com/db",
            }
        ],
    )
    assert validation["valid"] is False
