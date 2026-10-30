"""Domain-independent retrieval; importing this package loads no text adapter."""

from .engine import MemoryEngine
from .models import CandidateSet, ContextBudget, Evidence, MediaReference, Memory, RankedMemory, SearchResult
from .protocols import DomainAdapter, EmbeddingProvider
from .retrieval import RetrievalParameters

__all__ = ["MemoryEngine", "Memory", "RankedMemory", "ContextBudget", "CandidateSet",
           "Evidence", "MediaReference", "SearchResult", "DomainAdapter", "EmbeddingProvider", "RetrievalParameters"]
