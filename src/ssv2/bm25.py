"""Okapi BM25 over the passages already stored in an index.

``idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))`` and
``score = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))``.
"""

import math
import re
from collections import Counter
from collections.abc import Sequence

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    def __init__(self, texts: Sequence[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("k1 must be >= 0 and b must be between 0 and 1")
        self._k1 = k1
        self._b = b
        self._docs = [tokenize(text) for text in texts]
        lengths = [len(tokens) for tokens in self._docs]
        self._avgdl = (sum(lengths) / len(lengths)) if lengths else 0.0
        df: Counter[str] = Counter()
        for tokens in self._docs:
            df.update(set(tokens))
        total = len(self._docs)
        self._idf = {
            term: math.log(1 + (total - count + 0.5) / (count + 0.5)) for term, count in df.items()
        }

    def ranked(self, query: str) -> list[tuple[int, float]]:
        """Document indexes and scores, highest score first. Zero scores are left out."""
        terms = tokenize(query)
        if not terms or not self._docs:
            return []
        wanted = Counter(terms)
        scored: list[tuple[int, float]] = []
        for index, tokens in enumerate(self._docs):
            score = self._score(tokens, wanted)
            if score > 0:
                scored.append((index, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored

    def _score(self, tokens: list[str], wanted: Counter[str]) -> float:
        if not tokens or self._avgdl == 0:
            return 0.0
        counts = Counter(tokens)
        norm = self._k1 * (1 - self._b + self._b * len(tokens) / self._avgdl)
        total = 0.0
        for term, _query_tf in wanted.items():
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            idf = self._idf.get(term, 0.0)
            total += idf * (tf * (self._k1 + 1)) / (tf + norm)
        return total
