"""Identifier-aware lexical channel; no labels, repository IDs or model clients."""
from dataclasses import replace
import re

from ...core.models import RankedMemory
from ...core.retrieval import BM25Retriever


def identifier_text(text):
    # Preserve the intact token alongside its components. Language-independent.
    pieces = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    pieces = re.sub(r"([A-Z])([A-Z][a-z])", r"\1 \2", pieces)
    pieces = re.sub(r"[_./\\:-]+", " ", pieces)
    return text + "\n" + pieces


def code_source(text):
    return bool(re.search(r"(?m)^(?:diff --git |@@ .*@@|(?:async )?def \w+\(|class \w+\(|Traceback \(most recent call last\):)", text))


class CodeLexicalRetriever:
    def __init__(self, memories, *, k1=1.2, b=.75):
        self.memories = tuple(memories)
        self.original = BM25Retriever(self.memories, k1=k1, b=b)
        self.symbols = BM25Retriever([replace(m, text=identifier_text(m.text)) for m in self.memories], k1=k1, b=b)
        self.positions = {m.id: i for i, m in enumerate(self.memories)}

    def search(self, query, top_k=100):
        values = {}
        for ranking in (self.original.search(query, 100), self.symbols.search(identifier_text(query), 100)):
            for item in ranking:
                if item.score > 0:
                    ordinal = self.positions[item.memory.id]
                    values[ordinal] = values.get(ordinal, 0.) + .5 / (20 + item.rank)
        order = sorted(values, key=lambda i: (-values[i], i))[:top_k]
        return [RankedMemory(self.memories[i], values[i], rank) for rank, i in enumerate(order, 1)]
