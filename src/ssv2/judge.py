"""Score an answer for faithfulness and correctness.

The rubric judge is deterministic and is what unit tests and ``ssv2 judge`` use when no
model is configured. ``ModelJudge`` asks a language model for the same two scores.
"""

import json
import re
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, Field

from ssv2.generate import ABSTAIN, LanguageModel

_CITATION = re.compile(r"\[(\d+)\]")


class Judgement(BaseModel):
    faithfulness: float = Field(ge=0, le=1)
    correctness: float = Field(ge=0, le=1)


class Judge(Protocol):
    def score(
        self, query: str, answer: str, passages: Sequence[str], expected: str
    ) -> Judgement: ...


class RubricJudge:
    """Hand rules, no model.

    Correctness is 1 only when the answer text equals the expected text.
    Faithfulness is 0 when a non-abstaining answer cites a passage number that was not
    provided, or cites nothing. Abstaining is faithful.
    """

    def score(self, query: str, answer: str, passages: Sequence[str], expected: str) -> Judgement:
        del query
        text = answer.strip()
        correctness = 1.0 if text == expected.strip() else 0.0
        if text == ABSTAIN:
            return Judgement(faithfulness=1.0, correctness=correctness)
        numbers = [int(match) for match in _CITATION.findall(text)]
        if not numbers or any(number < 1 or number > len(passages) for number in numbers):
            return Judgement(faithfulness=0.0, correctness=correctness)
        return Judgement(faithfulness=1.0, correctness=correctness)


class ModelJudge:
    def __init__(self, model: LanguageModel) -> None:
        self._model = model

    def score(self, query: str, answer: str, passages: Sequence[str], expected: str) -> Judgement:
        reply = self._model.complete(
            'Reply with JSON {"faithfulness": number, "correctness": number}, each 0 to 1.',
            json.dumps(
                {"query": query, "answer": answer, "passages": list(passages), "expected": expected}
            ),
        )
        try:
            payload = json.loads(reply.text)
            return Judgement(
                faithfulness=float(payload["faithfulness"]),
                correctness=float(payload["correctness"]),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"judge reply was not a scorecard: {reply.text!r}") from exc


def mean_judgements(judgements: Sequence[Judgement]) -> Judgement:
    if not judgements:
        raise ValueError("no judgements")
    count = len(judgements)
    return Judgement(
        faithfulness=sum(item.faithfulness for item in judgements) / count,
        correctness=sum(item.correctness for item in judgements) / count,
    )
