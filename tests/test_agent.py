import json
from collections.abc import Mapping

from ssv2.agent import AgenticRetriever
from ssv2.mcp_server import handle, serve
from ssv2.retrieval import Hit
from tests.fakes import StubRetriever


class Counting:
    def __init__(self, score: float) -> None:
        self.info = StubRetriever().info
        self.calls: list[str] = []
        self.score = score

    def search(self, query: str, k: int, filters: Mapping[str, str] | None = None) -> list[Hit]:
        del k, filters
        self.calls.append(query)
        if self.score < 0:
            return []
        return [Hit(id="a", title="Alpha", text="body", score=self.score, metadata={})]


def test_default_cutoff_follows_up_when_the_top_hit_is_under_0_8() -> None:
    base = Counting(0.7)
    AgenticRetriever(base).search("claim", k=1)
    assert base.calls == ["claim", "claim Alpha"]


def test_a_strong_hit_is_a_single_search() -> None:
    base = Counting(0.9)
    hits = AgenticRetriever(base).search("claim", k=3)
    assert base.calls == ["claim"]
    assert hits[0].id == "a"
    assert hits[0].score == 0.9


def test_a_weak_hit_searches_again_and_fuses() -> None:
    base = Counting(0.2)
    hits = AgenticRetriever(base, followup_below=0.55).search("claim", k=3)
    assert base.calls == ["claim", "claim Alpha"]
    assert hits[0].id == "a"
    # The same document is rank 1 in both lists, so its fused score is 1/61 + 1/61.
    assert hits[0].score == 2 / 61


def test_no_hits_does_not_search_again() -> None:
    base = Counting(-1)
    assert AgenticRetriever(base).search("claim", k=3) == []
    assert base.calls == ["claim"]


def _search(query: str, k: int) -> list[Hit]:
    assert query == "vitamin"
    assert k == 2
    return [Hit(id="doc-1", title="Vitamin", text="Serum", score=0.5, metadata={})]


def test_mcp_initialize_list_and_search() -> None:
    init = handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}, _search)
    assert init is not None
    result = init["result"]
    assert isinstance(result, dict)
    assert result["serverInfo"]["name"] == "ssv2"
    listed = handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, _search)
    assert listed is not None
    tools = listed["result"]
    assert isinstance(tools, dict)
    tool_list = tools["tools"]
    assert isinstance(tool_list, list)
    assert tool_list[0]["name"] == "search"
    called = handle(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "search", "arguments": {"query": "vitamin", "k": 2}},
        },
        _search,
    )
    assert called is not None
    content = called["result"]
    assert isinstance(content, dict)
    blocks = content["content"]
    assert isinstance(blocks, list)
    hits = json.loads(blocks[0]["text"])
    assert hits[0]["id"] == "doc-1"


def test_mcp_rejects_bad_calls_and_skips_notifications() -> None:
    assert handle({"jsonrpc": "2.0", "method": "notifications/initialized"}, _search) is None
    unknown = handle({"jsonrpc": "2.0", "id": 1, "method": "nope"}, _search)
    assert unknown is not None
    assert "error" in unknown
    blank = handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "search", "arguments": {"query": "  "}},
        },
        _search,
    )
    assert blank is not None
    error = blank["error"]
    assert isinstance(error, dict)
    assert error["code"] == -32602
    bad_k = handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "search", "arguments": {"query": "q", "k": 0}},
        },
        _search,
    )
    assert bad_k is not None
    bad_error = bad_k["error"]
    assert isinstance(bad_error, dict)
    assert "k" in str(bad_error["message"])
    other = handle(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "delete"}},
        _search,
    )
    assert other is not None
    other_error = other["error"]
    assert isinstance(other_error, dict)
    assert "unknown tool" in str(other_error["message"])


def test_mcp_serve_skips_blank_lines_and_reports_bad_json() -> None:
    lines = serve(["\n", "not-json", "[]", '{"jsonrpc":"2.0","method":"ping"}'], _search)
    assert len(lines) == 2
    assert json.loads(lines[0])["error"]["code"] == -32700
    assert json.loads(lines[1])["error"]["code"] == -32600
