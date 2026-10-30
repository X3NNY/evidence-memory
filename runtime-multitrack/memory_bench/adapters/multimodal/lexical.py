"""Local lexical ranking of original chunks, with bounded dialogue continuity.

No labels, question categories, corpus question bank, embeddings or model client
are accepted. Search returns the original memories, never expanded evidence.
"""

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import math
import re

import numpy as np

from ...core.models import Memory, RankedMemory
from ...core.retrieval import BM25Retriever, tokenize


STOPWORDS = frozenset("""a an the this that these those i my me mine we our you your
he his him she her it its they their them what which where when how why who whom
is are was were be been being do does did have has had having will would could can
should to of in on at from for as by with and or but if some any all both each other
kind kinds type types about into there here s m re ve ll d hey hi please according
conversation user assistant image images picture pictures photo photos answer
using arabic numeral instead e g mentioned total show shows shown""".split())


@lru_cache(maxsize=65536)
def _stem(word):
    # Porter needs the installed lexical extra, no NLTK corpora or downloads.
    from nltk.stem import PorterStemmer
    return PorterStemmer().stem(word)


def normalize(text, mode):
    words = [word for word in tokenize(text) if word not in STOPWORDS]
    return [_stem(word) if mode == "stemmed" else word for word in words]


def fields(text):
    """Read only the raw field prefixes written by the conversation adapter."""
    text = re.sub(r"\ADate: [^\n]*\n", "", text)
    segments = re.split(r"(?m)^(user|assistant|Image caption): ", text)
    dialogue, captions = [], []
    if segments[0].strip():
        dialogue.append(segments[0])  # A continuation of an existing raw chunk.
    for position in range(1, len(segments), 2):
        (captions if segments[position] == "Image caption" else dialogue).append(segments[position+1])
    return "\n".join(dialogue), "\n".join(captions)


@dataclass(frozen=True)
class LexicalParameters:
    normalization: str = "stemmed"
    fielded: bool = False
    neighbor_weight: float = 0.0
    neighbor_radius: int = 2
    rrf_k: int = 20

    def __post_init__(self):
        if self.normalization not in {"content", "stemmed"} or not isinstance(self.fielded, bool):
            raise ValueError("Invalid lexical normalization or field policy")
        if (isinstance(self.neighbor_weight, bool) or not isinstance(self.neighbor_weight, (int, float))
                or not math.isfinite(self.neighbor_weight) or not 0 <= self.neighbor_weight <= 1):
            raise ValueError("Neighbor weight must be between zero and one")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (self.neighbor_radius, self.rrf_k)):
            raise ValueError("Lexical radii and rank constants must be positive integers")


class MultimodalLexicalRetriever:
    def __init__(self, memories, *, parameters=None, k1=1.2, b=.75):
        self.memories = tuple(memories)
        self.parameters = parameters or LexicalParameters()
        if len({m.id for m in self.memories}) != len(self.memories):
            raise ValueError("Memory identities must be unique")
        self.positions = {memory.id: i for i, memory in enumerate(self.memories)}
        self.fields = [fields(memory.text) for memory in self.memories]
        self.indexes = {}
        for channel in (("body", "dialogue", "caption") if self.parameters.fielded else ("body",)):
            normalized = []
            for memory, (dialogue, caption) in zip(self.memories, self.fields):
                raw = dialogue+"\n"+caption if channel == "body" else dialogue if channel == "dialogue" else caption
                normalized.append(Memory(memory.id, memory.scope_id, memory.source_ids, memory.session_id,
                                         " ".join(normalize(raw, self.parameters.normalization))))
            self.indexes[channel] = BM25Retriever(normalized, k1=k1, b=b)
        # Raw source identity is an opaque join key. Within-session order comes
        # from input order; no source-ID string or session name is tokenized.
        sessions = defaultdict(list)
        for i, memory in enumerate(self.memories):
            key = memory.source_ids
            if not sessions[memory.session_id] or sessions[memory.session_id][-1][0] != key:
                sessions[memory.session_id].append((key, []))
            sessions[memory.session_id][-1][1].append(i)
        self.neighbors = [[] for _ in self.memories]
        for groups in sessions.values():
            for position, (_, ordinals) in enumerate(groups):
                for other in range(max(0, position-self.parameters.neighbor_radius), min(len(groups), position+self.parameters.neighbor_radius+1)):
                    if other == position:
                        continue
                    for i in ordinals:
                        self.neighbors[i].append((tuple(groups[other][1]), abs(position-other)))

    def _scores(self, channel, text, *, rank_scores=False):
        normalized = " ".join(dict.fromkeys(normalize(text, self.parameters.normalization)))
        values = np.zeros(len(self.memories))
        if not normalized:
            return values
        for item in self.indexes[channel].search(normalized, max(1, len(self.memories))):
            if item.score > 0:
                values[self.positions[item.memory.id]] = 1/(self.parameters.rrf_k+item.rank) if rank_scores else item.score
        return values

    def score_vector(self, query):
        question, separator, query_caption = query.partition("\nQuery image caption: ")
        if self.parameters.fielded:
            values = (.5*self._scores("body", query, rank_scores=True)
                      +.35*self._scores("dialogue", question, rank_scores=True)
                      +.15*self._scores("caption", query_caption if separator else question, rank_scores=True))
        else:
            values = self._scores("body", query)
        maximum = float(values.max()) if len(values) else 0
        if maximum:
            values /= maximum
        if self.parameters.neighbor_weight:
            propagated = np.zeros(len(values))
            for i, adjacent in enumerate(self.neighbors):
                propagated[i] = max((float(values[list(others)].max())/(distance+1) for others, distance in adjacent), default=0)
            values = (1-self.parameters.neighbor_weight)*values+self.parameters.neighbor_weight*propagated
        return values

    def search(self, query, top_k=100):
        if not isinstance(query, str) or not query.strip() or not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("Search requires a question and a positive cutoff")
        values = self.score_vector(query)
        order = np.argsort(-values, kind="stable")[:top_k]
        return [RankedMemory(self.memories[int(i)], float(values[i]), rank) for rank, i in enumerate(order, 1)]
