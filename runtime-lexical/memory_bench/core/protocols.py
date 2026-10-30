"""Adapters receive raw memories, query text and retrieval signals only."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .models import CandidateSet, ContextBudget, Evidence, Memory, RankedMemory


class EmbeddingProvider(Protocol):
    def encode(self, texts: list[str], *, text_type: str, scope_id: str) -> np.ndarray: ...


class SparseRetriever(Protocol):
    def search(self, query: str, top_k: int = 100) -> list[RankedMemory]: ...


class DomainAdapter(Protocol):
    def index(self, scope_id: str, memories: tuple[Memory, ...], source: object) -> object: ...

    def search(self, index: object, query: str, candidates: CandidateSet,
               budget: ContextBudget) -> Evidence: ...
