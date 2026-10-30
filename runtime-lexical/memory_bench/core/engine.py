"""One isolated retrieval engine with pluggable evidence assembly.

This is a local library interface, not an HTTP server. The caller supplies an
embedding provider; the engine has no API-key lookup or model-client factory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .models import CandidateSet, ContextBudget, Memory, SearchResult
from .protocols import DomainAdapter, EmbeddingProvider, SparseRetriever
from .retrieval import BM25Retriever, DenseRetriever, HybridCandidateRanker, RetrievalParameters


@dataclass
class _IndexedScope:
    memories: tuple[Memory, ...]
    sparse: SparseRetriever
    dense: DenseRetriever | None
    domain: object


class MemoryEngine:
    def __init__(self, *, embedder: EmbeddingProvider | None = None, adapter: DomainAdapter,
                 retrieval: RetrievalParameters | None = None,
                 budget: ContextBudget | None = None, retriever: str = "hybrid",
                 sparse_factory: Callable[..., SparseRetriever] | None = None):
        if retriever not in {"bm25", "hybrid"}:
            raise ValueError("Engine retriever must be bm25 or hybrid")
        if retriever == "hybrid" and embedder is None:
            raise ValueError("Hybrid retrieval requires an explicit embedding provider")
        self.embedder, self.adapter = embedder, adapter
        self.retriever = retriever
        self.sparse_factory = sparse_factory or BM25Retriever
        self.retrieval = retrieval or RetrievalParameters()
        self.budget = budget or ContextBudget()
        self._scopes: dict[str, _IndexedScope] = {}

    def add(self, scope_id: str, memories: Sequence[Memory], source: object) -> None:
        if not isinstance(scope_id, str) or not scope_id:
            raise ValueError("A nonempty isolated scope is required")
        if scope_id in self._scopes:
            raise ValueError("Scope already indexed; use a new engine for a new version")
        memories = tuple(memories)
        if any(m.scope_id != scope_id for m in memories):
            raise ValueError("Memories cannot cross scopes")
        if len({m.id for m in memories}) != len(memories):
            raise ValueError("Memory IDs must be unique inside a scope")
        domain = self.adapter.index(scope_id, memories, source)
        sparse = self.sparse_factory(memories, k1=self.retrieval.k1, b=self.retrieval.b)
        dense = DenseRetriever(memories, self.embedder) if self.retriever == "hybrid" else None
        # Commit only after both the adapter and embedding provider succeed.
        self._scopes[scope_id] = _IndexedScope(memories, sparse, dense, domain)

    def search(self, scope_id: str, query: str) -> SearchResult:
        if scope_id not in self._scopes:
            raise KeyError("Scope has not been indexed")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query text must be nonempty")
        state, config = self._scopes[scope_id], self.retrieval
        sparse = state.sparse.search(query, config.candidate_k)
        if state.dense is None:
            dense, fused = [], sparse
        else:
            dense = state.dense.search(query, config.candidate_k)
            ranker = HybridCandidateRanker(state.memories, sparse, dense, state.dense.vectors)
            fused = ranker.rank(bm25_weight=config.bm25_weight, rrf_k=config.rrf_k,
                                mmr_lambda=config.mmr_lambda, top_k=max(1, len(ranker.ordinals)))
        candidates = CandidateSet(tuple(sparse), tuple(dense), tuple(fused))
        evidence = self.adapter.search(state.domain, query, candidates, self.budget)
        if len(evidence.context) > self.budget.max_chars or len(evidence.memories) > self.budget.max_documents:
            raise ValueError("Adapter evidence exceeds the configured context budget")
        if any(m.scope_id != scope_id for m in evidence.memories):
            raise ValueError("Adapter evidence cannot cross scopes")
        if len({m.id for m in evidence.memories}) != len(evidence.memories):
            raise ValueError("Adapter evidence contains duplicate memory IDs")
        if any(m.scope_id != scope_id for m in evidence.media):
            raise ValueError("Adapter media cannot cross scopes")
        if len({m.id for m in evidence.media}) != len(evidence.media):
            raise ValueError("Adapter evidence contains duplicate media IDs")
        if sum(m.byte_size for m in evidence.media) > self.budget.max_media_bytes:
            raise ValueError("Adapter media exceeds the configured media budget")
        return SearchResult(scope_id, evidence, candidates)
