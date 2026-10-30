"""Text-domain adapter; benchmark labels never enter its interface."""

from __future__ import annotations

from dataclasses import dataclass

from ...core.models import CandidateSet, ContextBudget, Evidence, Memory
from .session_packing import pack_sessions
from .structure import ConversationIndex, PassageQuery


@dataclass(frozen=True)
class TextSource:
    # Each session is (date, ((opaque_source_id, speaker, raw_prefixed_text), ...)).
    sessions: tuple[tuple[str, tuple[tuple[str, str, str], ...]], ...]

    @classmethod
    def from_sessions(cls, sessions):
        return cls(tuple((date, tuple(tuple(turn) for turn in turns)) for date, turns in sessions))


@dataclass(frozen=True)
class TextV7Parameters:
    chunk_words: int = 256
    passage_words: int = 128
    normalization: str = "stemmed"
    weight: float = .25

    def __post_init__(self):
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1
               for v in (self.chunk_words, self.passage_words)):
            raise ValueError("Text chunk sizes must be positive integers")
        if self.normalization not in {"raw", "content", "stemmed"} or self.weight not in {.1, .25, .5}:
            raise ValueError("Unsupported text policy parameters")


class TextAdapter:
    def __init__(self, parameters: TextV7Parameters | None = None):
        self.parameters = parameters or TextV7Parameters()

    def index(self, scope_id: str, memories: tuple[Memory, ...], source: object) -> ConversationIndex:
        if not isinstance(source, TextSource):
            raise TypeError("TextAdapter requires TextSource containing raw sessions only")
        return ConversationIndex(scope_id, memories, source.sessions,
                                 chunk_words=self.parameters.chunk_words,
                                 passage_words=self.parameters.passage_words, cache_aligned=False)

    def search(self, index: ConversationIndex, query: str, candidates: CandidateSet,
               budget: ContextBudget) -> Evidence:
        # Only v7 content retrieval needs the optional lexical dependency.
        from .content_memory import ContentQuery

        prepared = PassageQuery(index, query, candidates.fused, candidates.sparse, candidates.dense)
        policy = ContentQuery(prepared, query)
        ranking, audit_policy = policy.rank(normalization=self.parameters.normalization,
                                            weight=self.parameters.weight)
        text, audit, memories = pack_sessions(index, ranking, max_chars=budget.max_chars,
                                              max_documents=budget.max_documents)
        audit["experiment_policy"] = audit_policy
        return Evidence(text, audit, tuple(memories), tuple(ranking))
