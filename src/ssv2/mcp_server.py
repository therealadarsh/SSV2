"""A small stdio MCP server with one tool: search.

Speaks JSON-RPC 2.0 for ``initialize``, ``tools/list`` and ``tools/call``. It does not
depend on an MCP SDK. The search function is injected so tests never load a model.
"""

import json
from collections.abc import Callable, Iterable, Mapping

from ssv2 import __version__
from ssv2.retrieval import MAX_K, MIN_K, Hit

Search = Callable[[str, int], list[Hit]]
PROTOCOL = "2024-11-05"


def handle(message: Mapping[str, object], search: Search) -> dict[str, object] | None:
    """Answer one JSON-RPC message. A notification (no id) gets no response."""
    if "id" not in message:
        return None
    msg_id = message["id"]
    method = message.get("method")
    params = message.get("params")
    arguments = params if isinstance(params, dict) else {}
    if method == "initialize":
        return _result(
            msg_id,
            {
                "protocolVersion": PROTOCOL,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "ssv2", "version": __version__},
            },
        )
    if method == "tools/list":
        return _result(msg_id, {"tools": [_search_tool()]})
    if method == "tools/call":
        return _call(msg_id, arguments, search)
    return _error(msg_id, -32601, f"unknown method {method!r}")


def iter_responses(lines: Iterable[str], search: Search) -> Iterable[str]:
    """Yield one JSON response per request, as soon as that request is read."""
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            yield json.dumps(_error(None, -32700, f"line {line_number} is not JSON"))
            continue
        if not isinstance(message, dict):
            yield json.dumps(_error(None, -32600, f"line {line_number} is not an object"))
            continue
        response = handle(message, search)
        if response is not None:
            yield json.dumps(response)


def serve(lines: Iterable[str], search: Search) -> list[str]:
    return list(iter_responses(lines, search))


def _call(msg_id: object, params: Mapping[str, object], search: Search) -> dict[str, object]:
    name = params.get("name")
    if name != "search":
        return _error(msg_id, -32602, f"unknown tool {name!r}")
    raw = params.get("arguments")
    arguments = raw if isinstance(raw, dict) else {}
    query = arguments.get("query", "")
    if not isinstance(query, str) or not query.strip():
        return _error(msg_id, -32602, "query must not be blank")
    k = arguments.get("k", 5)
    if isinstance(k, bool) or not isinstance(k, int) or k < MIN_K or k > MAX_K:
        return _error(msg_id, -32602, f"k must be an integer from {MIN_K} to {MAX_K}")
    hits = search(query.strip(), k)
    payload = [hit.model_dump(mode="json") for hit in hits]
    return _result(
        msg_id,
        {"content": [{"type": "text", "text": json.dumps(payload)}], "isError": False},
    )


def _search_tool() -> dict[str, object]:
    return {
        "name": "search",
        "description": "Search the loaded index and return ranked passages.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "minimum": MIN_K, "maximum": MAX_K},
            },
            "required": ["query"],
        },
    }


def _result(msg_id: object, result: dict[str, object]) -> dict[str, object]:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _error(msg_id: object, code: int, message: str) -> dict[str, object]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}
