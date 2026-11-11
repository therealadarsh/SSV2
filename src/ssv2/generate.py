"""Retrieve, then ask a language model to answer with citations.

When the top hit is below the score floor, the service says "I don't know." and does
not call the model. The Anthropic client is used only when ``ANTHROPIC_API_KEY`` is set.
The key is read from the environment inside the client and is never logged.
"""

import json
import os
import time
import urllib.request
from collections.abc import Callable, Mapping
from typing import Protocol

from pydantic import BaseModel, Field

from ssv2.config import Settings
from ssv2.retrieval import Hit, Retriever

ABSTAIN = "I don't know."
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

Transport = Callable[[bytes, Mapping[str, str]], bytes]


class ModelReply(BaseModel):
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0


class Answer(BaseModel):
    answer: str
    abstained: bool
    hits: list[Hit] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)


class MissingModelError(RuntimeError):
    """No language model is configured for this process."""


class LanguageModel(Protocol):
    def complete(self, system: str, user: str) -> ModelReply: ...


class UnavailableLanguageModel:
    """Used when the process has no API key. ``complete`` explains that, and nothing else."""

    def complete(self, system: str, user: str) -> ModelReply:
        del system, user
        raise MissingModelError("ANTHROPIC_API_KEY is not set")


def cost_usd(input_tokens: int, output_tokens: int, settings: Settings) -> float:
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("token counts must be >= 0")
    million = 1_000_000
    input_cost = input_tokens * settings.input_usd_per_million
    output_cost = output_tokens * settings.output_usd_per_million
    return (input_cost + output_cost) / million


def ask(
    retriever: Retriever,
    model: LanguageModel,
    query: str,
    settings: Settings,
    *,
    k: int,
    filters: Mapping[str, str] | None = None,
) -> Answer:
    started = time.perf_counter()
    hits = retriever.search(query, k, filters)
    # Hybrid and rerank scores are not cosine, so the cosine floor would abstain on every hit.
    floor = settings.abstain_below if settings.retriever == "dense" else 0.0
    relevant = [hit for hit in hits if hit.score >= floor]
    if not relevant:
        return Answer(
            answer=ABSTAIN,
            abstained=True,
            hits=[],
            usage=Usage(latency_ms=_elapsed_ms(started)),
        )
    reply = model.complete(_system_prompt(), _user_prompt(query, relevant))
    text = reply.text.strip() or ABSTAIN
    abstained = text == ABSTAIN
    usage = Usage(
        input_tokens=reply.input_tokens,
        output_tokens=reply.output_tokens,
        cost_usd=cost_usd(reply.input_tokens, reply.output_tokens, settings),
        latency_ms=_elapsed_ms(started),
    )
    return Answer(answer=text, abstained=abstained, hits=relevant, usage=usage)


def language_model_from_env(settings: Settings) -> LanguageModel:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        return UnavailableLanguageModel()
    return ClaudeLanguageModel(key, model=settings.anthropic_model)


class ClaudeLanguageModel:
    """Messages API client. ``transport`` replaces the network in tests."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str,
        transport: Transport | None = None,
        max_tokens: int = 1024,
    ) -> None:
        if not api_key.strip():
            raise MissingModelError("ANTHROPIC_API_KEY is not set")
        self._api_key = api_key
        self._model = model
        self._transport = transport or _post_anthropic
        self._max_tokens = max_tokens

    def complete(self, system: str, user: str) -> ModelReply:
        payload = json.dumps(
            {
                "model": self._model,
                "max_tokens": self._max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }
        ).encode()
        headers = {
            "content-type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }
        body = json.loads(self._transport(payload, headers))
        parts = body.get("content", [])
        text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
        usage = body.get("usage", {})
        return ModelReply(
            text=text,
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
        )


def parse_stream(chunks: list[str]) -> ModelReply:
    """Turn Anthropic SSE data lines into one reply. Unknown events are ignored."""
    text: list[str] = []
    input_tokens = 0
    output_tokens = 0
    for chunk in chunks:
        if not chunk.startswith("data:"):
            continue
        raw = chunk[len("data:") :].strip()
        if raw == "[DONE]" or not raw:
            continue
        event = json.loads(raw)
        kind = event.get("type")
        if kind == "message_start":
            input_tokens = int(event.get("message", {}).get("usage", {}).get("input_tokens", 0))
        elif kind == "content_block_delta":
            text.append(str(event.get("delta", {}).get("text", "")))
        elif kind == "message_delta":
            output_tokens = int(event.get("usage", {}).get("output_tokens", output_tokens))
    return ModelReply(text="".join(text), input_tokens=input_tokens, output_tokens=output_tokens)


def _system_prompt() -> str:
    return (
        "Answer the question using only the numbered passages. "
        f"Cite a passage as [n]. If they do not contain the answer, reply exactly: {ABSTAIN}"
    )


def _user_prompt(query: str, hits: list[Hit]) -> str:
    lines = [f"Question: {query}", "Passages:"]
    for number, hit in enumerate(hits, start=1):
        lines.append(f"[{number}] {hit.title} {hit.text}".strip())
    return "\n".join(lines)


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _post_anthropic(payload: bytes, headers: Mapping[str, str]) -> bytes:
    request = urllib.request.Request(
        ANTHROPIC_URL, data=payload, headers=dict(headers), method="POST"
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return bytes(response.read())
