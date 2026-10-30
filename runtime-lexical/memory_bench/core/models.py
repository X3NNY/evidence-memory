"""Label-free records exchanged between the retrieval engine and adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Memory:
    id: str
    scope_id: str
    source_ids: tuple[str, ...]
    session_id: str
    text: str


@dataclass(frozen=True)
class RankedMemory:
    memory: Memory
    score: float
    rank: int


@dataclass(frozen=True)
class ContextBudget:
    max_chars: int = 48000
    max_documents: int = 100
    max_media_bytes: int = 30 * 1024 * 1024

    def __post_init__(self):
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1
               for v in (self.max_chars, self.max_documents)):
            raise ValueError("Context budgets must be positive integers")
        if not isinstance(self.max_media_bytes, int) or isinstance(self.max_media_bytes, bool) or self.max_media_bytes < 0:
            raise ValueError("Media budget must be a nonnegative integer")


@dataclass(frozen=True)
class MediaReference:
    id: str
    scope_id: str
    path: str
    media_type: str
    byte_size: int
    sha256: str

    def __post_init__(self):
        if (not self.id or not self.scope_id or not self.path
                or not isinstance(self.byte_size, int) or isinstance(self.byte_size, bool) or self.byte_size < 1):
            raise ValueError("Invalid original-media reference")


@dataclass(frozen=True)
class CandidateSet:
    sparse: tuple[RankedMemory, ...]
    dense: tuple[RankedMemory, ...]
    fused: tuple[RankedMemory, ...]


@dataclass(frozen=True)
class Evidence:
    context: str
    audit: dict[str, Any]
    memories: tuple[Memory, ...]
    unit_ranking: tuple[tuple[int, float], ...] = ()
    media: tuple[MediaReference, ...] = ()


@dataclass(frozen=True)
class SearchResult:
    scope_id: str
    evidence: Evidence
    candidates: CandidateSet
