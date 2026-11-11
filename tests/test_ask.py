import json
from collections.abc import Mapping
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ssv2.api.app import create_app
from ssv2.cli import main
from ssv2.config import Settings
from ssv2.generate import (
    ABSTAIN,
    ClaudeLanguageModel,
    MissingModelError,
    ModelReply,
    UnavailableLanguageModel,
    ask,
    cost_usd,
    parse_stream,
)
from ssv2.judge import ModelJudge, RubricJudge, mean_judgements
from ssv2.retrieval import Hit
from tests.fakes import StubRetriever


class EchoModel:
    def __init__(self, text: str = "See [1].") -> None:
        self.text = text
        self.calls = 0
        self.user = ""

    def complete(self, system: str, user: str) -> ModelReply:
        del system
        self.calls += 1
        self.user = user
        return ModelReply(text=self.text, input_tokens=1_000_000, output_tokens=1_000)


class OneHit:
    def __init__(self, score: float) -> None:
        self.info = StubRetriever().info
        self.score = score

    def search(self, query: str, k: int, filters: Mapping[str, str] | None = None) -> list[Hit]:
        del query, k, filters
        return [Hit(id="doc-1", title="Title", text="Body", score=self.score, metadata={})]


def test_cost_is_the_published_per_million_rate() -> None:
    settings = Settings()
    assert cost_usd(1_000_000, 0, settings) == pytest.approx(3.0)
    assert cost_usd(500, 1_000, settings) == pytest.approx(0.0165)
    with pytest.raises(ValueError, match="token counts"):
        cost_usd(-1, 0, settings)


def test_a_weak_dense_hit_abstains_without_calling_the_model() -> None:
    model = EchoModel()
    answer = ask(OneHit(0.2), model, "why", Settings(), k=3)
    assert answer.abstained is True
    assert answer.answer == ABSTAIN
    assert answer.hits == []
    assert answer.usage.input_tokens == 0
    assert answer.usage.cost_usd == 0
    assert model.calls == 0


def test_a_strong_hit_is_cited_and_billed() -> None:
    model = EchoModel()
    answer = ask(OneHit(0.9), model, "why", Settings(), k=3)
    assert answer.abstained is False
    assert answer.answer == "See [1]."
    assert answer.hits[0].id == "doc-1"
    assert "[1] Title Body" in model.user
    assert answer.usage.input_tokens == 1_000_000
    assert answer.usage.cost_usd == pytest.approx(3.015)
    assert answer.usage.latency_ms >= 0


def test_the_model_can_still_abstain_after_retrieval() -> None:
    answer = ask(OneHit(0.9), EchoModel(ABSTAIN), "why", Settings(), k=1)
    assert answer.abstained is True
    assert answer.hits[0].id == "doc-1"


def test_hybrid_scores_are_not_compared_to_the_cosine_floor() -> None:
    settings = Settings(retriever="hybrid")
    answer = ask(OneHit(0.01), EchoModel(), "why", settings, k=1)
    assert answer.abstained is False


def test_claude_client_sends_the_key_as_a_header_and_parses_usage() -> None:
    captured: dict[str, object] = {}

    def transport(payload: bytes, headers: Mapping[str, str]) -> bytes:
        captured["payload"] = json.loads(payload)
        captured["headers"] = dict(headers)
        return json.dumps(
            {
                "content": [{"type": "text", "text": "Cited [1]."}],
                "usage": {"input_tokens": 4, "output_tokens": 2},
            }
        ).encode()

    reply = ClaudeLanguageModel("test-key", model="claude-test", transport=transport).complete(
        "system", "user"
    )
    assert reply.text == "Cited [1]."
    assert reply.input_tokens == 4
    assert reply.output_tokens == 2
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["x-api-key"] == "test-key"
    assert headers["anthropic-version"] == "2023-06-01"
    with pytest.raises(MissingModelError):
        ClaudeLanguageModel("  ", model="claude-test")
    with pytest.raises(MissingModelError):
        UnavailableLanguageModel().complete("s", "u")


def test_stream_parser_joins_deltas_and_ignores_other_events() -> None:
    reply = parse_stream(
        [
            'data: {"type":"message_start","message":{"usage":{"input_tokens":3}}}',
            "event: ping",
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi"}}',
            'data: {"type":"content_block_delta","delta":{"text":"!"}}',
            'data: {"type":"message_delta","usage":{"output_tokens":2}}',
            "data: [DONE]",
        ]
    )
    assert reply.text == "Hi!"
    assert reply.input_tokens == 3
    assert reply.output_tokens == 2


def test_rubric_boundaries() -> None:
    rubric = RubricJudge()
    passages = ["the body"]
    both = rubric.score("q", ABSTAIN, passages, ABSTAIN)
    assert both.faithfulness == 1
    assert both.correctness == 1
    wrong = rubric.score("q", ABSTAIN, passages, "yes")
    assert wrong.correctness == 0
    assert wrong.faithfulness == 1
    cited = rubric.score("q", "See [1].", passages, "See [1].")
    assert cited.faithfulness == 1
    assert cited.correctness == 1
    assert rubric.score("q", "See [2].", passages, "See [2].").faithfulness == 0
    assert rubric.score("q", "no cite", passages, "no cite").faithfulness == 0
    with pytest.raises(ValueError, match="no judgements"):
        mean_judgements([])


def test_model_judge_rejects_a_non_scorecard() -> None:
    judge = ModelJudge(EchoModel("not json"))
    with pytest.raises(ValueError, match="not a scorecard"):
        judge.score("q", "a", ["p"], "a")
    parsed = ModelJudge(EchoModel('{"faithfulness": 1, "correctness": 0}')).score("q", "a", [], "b")
    assert parsed.faithfulness == 1
    assert parsed.correctness == 0
    with pytest.raises(ValueError, match="not a scorecard"):
        ModelJudge(EchoModel('{"faithfulness": 2, "correctness": 0}')).score("q", "a", [], "b")


def test_ask_endpoint_answers_streams_and_reports_a_missing_model() -> None:
    app = create_app(Settings(), retriever=StubRetriever(), language_model=EchoModel())
    with TestClient(app) as client:
        response = client.post("/v1/ask", json={"query": "vitamin D", "k": 2})
        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "See [1]."
        assert body["abstained"] is False
        assert body["hits"][0]["id"] == "doc-1"
        assert body["usage"]["cost_usd"] == pytest.approx(3.015)
        streamed = client.post("/v1/ask", json={"query": "vitamin D", "stream": True})
        assert streamed.status_code == 200
        assert "text/event-stream" in streamed.headers["content-type"]
        assert "See [1]." in streamed.text
        assert '"done": true' in streamed.text
        assert client.post("/v1/ask", json={"query": "  "}).status_code == 422

    bare = create_app(Settings(), retriever=StubRetriever())
    with TestClient(bare) as client:
        missing = client.post("/v1/ask", json={"query": "vitamin D"})
        assert missing.status_code == 503
        assert "ANTHROPIC_API_KEY" in missing.json()["detail"]


def test_home_page_can_ask() -> None:
    app = create_app(Settings(), retriever=StubRetriever(), language_model=EchoModel())
    with TestClient(app) as client:
        page = client.get("/").text
    assert 'id="ask"' in page
    assert "/v1/ask" in page
    assert 'id="answer"' in page


def test_judge_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "query": "q",
                        "answer": "See [1].",
                        "passages": ["body"],
                        "expected": "See [1].",
                    }
                ),
                json.dumps({"query": "q", "answer": ABSTAIN, "passages": [], "expected": "yes"}),
                "",
            ]
        ),
        encoding="utf-8",
    )
    assert main(["judge", str(cases)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["faithfulness"] == 1
    assert summary["correctness"] == 0.5
