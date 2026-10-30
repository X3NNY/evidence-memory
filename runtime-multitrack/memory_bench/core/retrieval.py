"""Isolated BM25, dense cosine, and reciprocal-rank-fusion retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .models import Memory, RankedMemory


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.casefold(), flags=re.UNICODE)


class BM25Retriever:
    def __init__(self, memories: Sequence[Memory], k1: float = 1.2, b: float = 0.75):
        if (any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v)
                for v in (k1, b)) or k1 <= 0 or not 0 <= b <= 1):
            raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
        self.memories = tuple(memories)
        scopes = {memory.scope_id for memory in self.memories}
        if len(scopes) > 1:
            raise ValueError("A retrieval index must contain exactly one isolated scope")
        self.k1, self.b = k1, b
        lengths = []
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for ordinal, memory in enumerate(self.memories):
            counts = Counter(tokenize(memory.text))
            lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                postings[term].append((ordinal, frequency))
        self.lengths = np.asarray(lengths, dtype=np.float64)
        average = float(self.lengths.mean()) if len(lengths) else 1.0
        average = average or 1.0
        self.norm = self.k1 * (1 - self.b + self.b * self.lengths / average)
        count = len(self.memories)
        self.postings: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
        for term, occurrences in postings.items():
            ordinals, frequencies = zip(*occurrences)
            idf = math.log1p((count - len(occurrences) + 0.5) / (len(occurrences) + 0.5))
            self.postings[term] = (np.asarray(ordinals, dtype=np.int64),
                                   np.asarray(frequencies, dtype=np.float64), idf)

    def search(self, query: str, top_k: int = 100) -> list[RankedMemory]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        scores = np.zeros(len(self.memories), dtype=np.float64)
        for term, query_frequency in Counter(tokenize(query)).items():
            if term not in self.postings:
                continue
            ordinals, frequencies, idf = self.postings[term]
            scores[ordinals] += (query_frequency * idf * frequencies * (self.k1 + 1)
                                 / (frequencies + self.norm[ordinals]))
        # Input ordinals define ties. IDs and labels never affect ranking.
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [RankedMemory(self.memories[int(ordinal)], float(scores[ordinal]), rank)
                for rank, ordinal in enumerate(order, start=1)]


class DenseRetriever:
    def __init__(self, memories: Sequence[Memory], embedder):
        self.memories = tuple(memories)
        scopes = {memory.scope_id for memory in self.memories}
        if len(scopes) > 1:
            raise ValueError("A retrieval index must contain exactly one isolated scope")
        self.scope_id = next(iter(scopes), "empty")
        self.embedder = embedder
        self.vectors = embedder.encode([memory.text for memory in self.memories],
                                       text_type="document", scope_id=self.scope_id)

    def search(self, query: str, top_k: int = 100) -> list[RankedMemory]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not self.memories:
            return []
        vector = self.embedder.encode([query], text_type="query", scope_id=self.scope_id)[0]
        scores = self.vectors @ vector
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [RankedMemory(self.memories[int(ordinal)], float(scores[ordinal]), rank)
                for rank, ordinal in enumerate(order, start=1)]


class HybridRetriever:
    """Fuse equally weighted BM25/dense ranks without mixing their score scales."""

    def __init__(self, memories: Sequence[Memory], embedder, *, k1: float = 1.2,
                 b: float = 0.75, candidate_k: int = 100, rrf_k: int = 60):
        if (not isinstance(candidate_k, int) or isinstance(candidate_k, bool) or candidate_k < 1
                or not isinstance(rrf_k, int) or isinstance(rrf_k, bool) or rrf_k < 1):
            raise ValueError("Hybrid candidate_k and rrf_k must be positive integers")
        self.memories = tuple(memories)
        self.ordinals = {memory.id: ordinal for ordinal, memory in enumerate(self.memories)}
        if len(self.ordinals) != len(self.memories):
            raise ValueError("Memory IDs must be unique inside a scope")
        self.sparse = BM25Retriever(self.memories, k1=k1, b=b)
        self.dense = DenseRetriever(self.memories, embedder)
        self.candidate_k, self.rrf_k = candidate_k, rrf_k

    def search(self, query: str, top_k: int = 100) -> list[RankedMemory]:
        if top_k < 1 or top_k > self.candidate_k:
            raise ValueError("top_k must be positive and no larger than hybrid candidate_k")
        scores: dict[int, float] = defaultdict(float)
        for ranking in (self.sparse.search(query, self.candidate_k),
                        self.dense.search(query, self.candidate_k)):
            for item in ranking:
                scores[self.ordinals[item.memory.id]] += 1.0 / (self.rrf_k + item.rank)
        order = sorted(scores, key=lambda ordinal: (-scores[ordinal], ordinal))[:top_k]
        return [RankedMemory(self.memories[ordinal], scores[ordinal], rank)
                for rank, ordinal in enumerate(order, start=1)]


class HybridCandidateRanker:
    """Reuse sparse/dense candidates for weighted RRF and optional cosine MMR.

    Inputs contain only memories, retrieval ranks, and document vectors. Gold
    labels, question categories, answers, and source-ID strings are not features.
    Candidate ties follow the original memory order, as in HybridRetriever.
    """

    def __init__(self, memories: Sequence[Memory], sparse: Sequence[RankedMemory],
                 dense: Sequence[RankedMemory], vectors: np.ndarray):
        self.memories = tuple(memories)
        if len({m.scope_id for m in memories}) > 1:
            raise ValueError("Candidates must belong to one isolated scope")
        ordinals = {m.id: i for i, m in enumerate(memories)}
        if len(ordinals) != len(memories):
            raise ValueError("Memory IDs must be unique inside a scope")
        if vectors.ndim != 2 or len(vectors) != len(memories) or not np.isfinite(vectors).all():
            raise ValueError("Candidate vectors must align with the memories")
        selected = {ordinals[item.memory.id] for ranking in (sparse, dense) for item in ranking}
        self.ordinals = np.asarray(sorted(selected), dtype=np.int64)
        positions = {int(ordinal): i for i, ordinal in enumerate(self.ordinals)}
        self.ranks = np.zeros((2, len(self.ordinals)), dtype=np.float64)
        for channel, ranking in enumerate((sparse, dense)):
            for item in ranking:
                ordinal = ordinals[item.memory.id]
                if item.memory != memories[ordinal] or item.rank < 1 or self.ranks[channel, positions[ordinal]]:
                    raise ValueError("Invalid or duplicate ranked candidate")
                self.ranks[channel, positions[ordinal]] = item.rank
        candidate_vectors = vectors[self.ordinals]
        self.similarities = np.clip(candidate_vectors @ candidate_vectors.T, 0.0, 1.0)

    def rank(self, *, bm25_weight: float = 0.5, rrf_k: int = 60,
             mmr_lambda: float = 1.0, top_k: int = 100) -> list[RankedMemory]:
        if (any(not isinstance(v, (float, int)) or isinstance(v, bool) or not math.isfinite(v)
                for v in (bm25_weight, mmr_lambda)) or not 0 <= bm25_weight <= 1
                or not 0 < mmr_lambda <= 1):
            raise ValueError("Weights must be finite, 0 <= bm25_weight <= 1, 0 < mmr_lambda <= 1")
        if (not isinstance(rrf_k, int) or isinstance(rrf_k, bool) or rrf_k < 1
                or not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1):
            raise ValueError("rrf_k and top_k must be positive integers")
        relevance = np.zeros(len(self.ordinals), dtype=np.float64)
        for channel, weight in enumerate((bm25_weight, 1 - bm25_weight)):
            present = self.ranks[channel] > 0
            # Factor two preserves the original equal-RRF scores at weight=0.5.
            relevance[present] += 2 * weight / (rrf_k + self.ranks[channel, present])
        if mmr_lambda == 1 or not len(relevance):
            order = np.argsort(-relevance, kind="stable")[:top_k]
        else:
            scaled = relevance / max(float(relevance.max()), np.finfo(float).eps)
            redundancy = np.zeros(len(relevance))
            used = np.zeros(len(relevance), dtype=bool)
            order = []
            for _ in range(min(top_k, len(relevance))):
                utility = mmr_lambda * scaled - (1 - mmr_lambda) * redundancy
                utility[used] = -np.inf
                pick = int(np.argmax(utility))
                order.append(pick)
                used[pick] = True
                redundancy = np.maximum(redundancy, self.similarities[:, pick])
        return [RankedMemory(self.memories[int(self.ordinals[position])], float(relevance[position]), rank)
                for rank, position in enumerate(order, 1)]


@dataclass(frozen=True)
class RetrievalParameters:
    k1: float = 1.2
    b: float = .75
    candidate_k: int = 100
    bm25_weight: float = .3
    rrf_k: int = 20
    mmr_lambda: float = .7

    def __post_init__(self):
        numbers = (self.k1, self.b, self.bm25_weight, self.mmr_lambda)
        if (any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in numbers)
                or self.k1 <= 0 or not 0 <= self.b <= 1 or not 0 <= self.bm25_weight <= 1
                or not 0 < self.mmr_lambda <= 1):
            raise ValueError("Invalid retrieval weights or BM25 parameters")
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1
               for v in (self.candidate_k, self.rrf_k)):
            raise ValueError("Retrieval candidate_k and rrf_k must be positive integers")
