"""Name-neutral, morphology-normalized retrieval of actual conversation text."""

from __future__ import annotations

from functools import lru_cache
import re

import numpy as np

try:
    from nltk.stem import PorterStemmer
except ImportError as error:
    raise ImportError("This optional experiment needs the lexical extra (nltk); no corpus download is required") from error

from .attributed_memory import RESERVED_ROLES
from ...core.models import Memory
from ...core.retrieval import BM25Retriever, tokenize
from .session_packing import SessionQuery, pack_sessions


# Generic question/function words, not topic synonyms or benchmark answers.
# Retain negation and temporal/comparison terms to avoid erasing relationships.
STOPWORDS = frozenset("""a an the this that these those i my me mine we our you your
he his him she her it its they their them what which where when how why who whom
is are was were be been being do does did have has had having will would could can
should to of in on at from for as by with and or but if some any all both each other
kind kinds type types about into there here s m re ve ll d hey hi""".split())
_porter = PorterStemmer()


@lru_cache(maxsize=65536)
def stem(token):
    return _porter.stem(token)


class ContentIndex:
    """Indexes bodies only; participants and function words are not topics."""

    def __init__(self, index, speakers):
        self.speakers = tuple(dict.fromkeys(speakers))
        self.participant_tokens = {term for name in speakers for term in tokenize(name)}
        self.indexes = {}
        for mode in ("content", "stemmed"):
            memories = [Memory(f"{index.scope_id}/content/{p.ordinal}", index.scope_id, (), str(p.session_no),
                               " ".join(self.normalize(p.text.partition(": ")[2], mode))) for p in index.passages]
            self.indexes[mode] = BM25Retriever(memories)

    def normalize(self, text, mode):
        tokens = [t for t in tokenize(text) if t not in STOPWORDS and t not in self.participant_tokens]
        return [stem(t) for t in tokens] if mode == "stemmed" else tokens

    def ranks(self, query, mode):
        tokens = self.normalize(query, mode)
        ranks = np.full(len(self.indexes[mode].memories), np.inf)
        if tokens:
            for item in self.indexes[mode].search(" ".join(tokens), max(1, len(ranks))):
                if item.score > 0:
                    ranks[int(item.memory.id.rsplit("/", 1)[1])] = item.rank
        return ranks, tokens


class ContentQuery(SessionQuery):
    """Refine named-participant questions, without preferring one speaker."""

    def __init__(self, prepared, query):
        super().__init__(prepared, query)
        self.query = query
        self.v6_ranking, self.v6_policy = super().rank(protect=False, density=.5)
        speakers = [name for name in dict.fromkeys(self.base.roles) if name.casefold() not in RESERVED_ROLES]
        self.named_question = any(re.search(r"(?<!\w)"+re.escape(name)+r"(?!\w)", query, flags=re.I) for name in speakers)
        self.local_ranks, self.query_terms = {}, {}
        if self.named_question:
            if not hasattr(self.index, "_content_lexical"):
                self.index._content_lexical = ContentIndex(self.index, speakers)
            for mode in ("content", "stemmed"):
                self.local_ranks[mode], self.query_terms[mode] = self.index._content_lexical.ranks(query, mode)

    def replay_output(self):
        return pack_sessions(self.index, self.v6_ranking)

    def rank(self, *, normalization="stemmed", weight=.25):
        if normalization not in {"raw", "content", "stemmed"} or weight not in {.1, .25, .5}:
            raise ValueError("Unsupported content retrieval parameters")
        lexical = (self.base.prepared.lexical if normalization == "raw" else self.local_ranks.get(normalization))
        active = self.named_question and lexical is not None and np.isfinite(lexical).any()
        audit = {"normalization": normalization, "weight": weight, "active": bool(active),
                 "named_participant_query": self.named_question,
                 "query_content_terms": self.query_terms.get(normalization, []),
                 "speaker_preference": False, "raw_source_only": True}
        if not active:
            return list(self.v6_ranking), audit
        original_weight = self.policy["effective_lexical_weight"]
        scores = ((1-original_weight)/(20+self.base.prepared.parents)
                  + original_weight/(20+self.base.lexical)+.15/(20+self.base.sessions))
        lexical_scores = 1/(20+lexical)+.15/(20+self.base.sessions)
        if self.base.personal:
            users = np.asarray([role == "user" for role in self.base.roles])
            scores[users] *= 2.
            lexical_scores[users] *= 2.
        scores = (1-weight)*scores+weight*lexical_scores
        eligible = self.eligible | (lexical <= 100)
        exponent = .5 if original_weight else 0.
        scores /= np.asarray([max(256, len(p.text))**exponent for p in self.index.passages])
        candidates = np.flatnonzero(eligible)
        order = candidates[np.argsort(-scores[candidates], kind="stable")]
        return [(int(i), float(scores[i])) for i in order], audit

    def pack(self, ranking, *, parameters, baseline_audit):
        return pack_sessions(self.index, ranking)
