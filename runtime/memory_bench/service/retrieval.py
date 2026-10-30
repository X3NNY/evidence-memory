"""Online projection of raw messages into the existing v7 retrieval pipeline.

Stable per-message chunks make appends cheap. Original text offsets remain
available, and response records contain complete chunks without truncation.
"""

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import re

import numpy as np

from ..core.engine import MemoryEngine
from ..core.models import ContextBudget, Evidence, Memory
from ..core.retrieval import BM25Retriever, RetrievalParameters
from ..adapters.text.adapter import TextAdapter
from ..adapters.text.structure import ConversationIndex, Passage, PassageQuery
from .models import timestamp_iso
from .evidence import EvidencePolicy
from .responses import CachedRecord, SourceRecord, ResponseCache, response


def raw_ranges(text, max_words=256, max_bytes=6000):
    """Lossless contiguous ranges, including long CJK/code tokens and spaces."""
    if max_words < 1 or max_bytes < 4:
        raise ValueError("Invalid chunk limits")
    start, size, words, in_word = 0, 0, 0, False
    for position, char in enumerate(text):
        begins_word = not char.isspace() and not in_word
        width = len(char.encode("utf-8"))
        if position > start and (size + width > max_bytes or begins_word and words >= max_words):
            # Prefer an existing whitespace boundary for a byte-limited chunk.
            boundary = position
            if size + width > max_bytes and in_word:
                candidate = max(text.rfind(" ", start, position), text.rfind("\n", start, position),
                                text.rfind("\t", start, position))
                if candidate >= start and text[start:candidate + 1].strip():
                    boundary = candidate + 1
            yield start, boundary
            start = boundary
            remainder = text[boundary:position]
            size, words = len(remainder.encode("utf-8")), len(re.findall(r"\S+", remainder))
            in_word = bool(remainder and not remainder[-1].isspace())
            begins_word = not char.isspace() and not in_word
        size += width
        words += int(begins_word)
        in_word = not char.isspace()
    if start < len(text):
        yield start, len(text)


@dataclass(frozen=True)
class RawChunk:
    id: str
    source_id: str
    session_id: str
    role: str
    raw: str
    start: int
    end: int
    created_at: str
    source_date: str | None

    @property
    def rendered(self):
        date = f"Conversation timestamp: {self.source_date}\n" if self.source_date else ""
        return date + f"{self.role}: {self.raw}"


def project_rows(rows):
    chunks = []
    for row in rows:
        source_date = timestamp_iso(row["source_timestamp"]) if row["source_timestamp"] is not None else None
        pending_start = 0
        for start, end in raw_ranges(row["content"]):
            raw = row["content"][start:end]
            if not raw.strip():
                # Keep trailing whitespace on the previous chunk, never invent text.
                if chunks and chunks[-1].source_id == row["memory_id"]:
                    from dataclasses import replace
                    previous = chunks.pop()
                    chunks.append(replace(previous, raw=previous.raw + raw, end=end))
                    pending_start = end
                continue
            start = min(start, pending_start)
            raw = row["content"][start:end]
            identity = f"{row['memory_id']}:{start}:{end}"
            chunks.append(RawChunk("chunk_" + hashlib.sha256(identity.encode()).hexdigest(),
                                   row["memory_id"], row["session_id"], row["role"], raw, start, end,
                                   source_date or row["persisted_at"], source_date))
            pending_start = end
    return tuple(chunks)


class OnlineConversationIndex(ConversationIndex):
    """Same passage/ranking interface, without frozen benchmark chunk IDs."""
    def __init__(self, scope_id, memories, chunks):
        self.scope_id, self.memories = scope_id, tuple(memories)
        self.chunks = tuple(chunks)
        self.memory_ordinals = {m.id: i for i, m in enumerate(memories)}
        groups = OrderedDict()
        for ordinal, chunk in enumerate(chunks):
            groups.setdefault(chunk.session_id, []).append((ordinal, chunk))
        self.dates = tuple("" for _ in groups)
        self.date_headers = tuple("" for _ in groups)
        self.chunk_sessions = np.zeros(len(chunks), dtype=np.int64)
        self.turn_lengths, self.passages = {}, []
        self.cache_aligned = False
        for session_no, group in enumerate(groups.values()):
            turns, word_offsets = {}, {}
            for ordinal, chunk in group:
                self.chunk_sessions[ordinal] = session_no
                turn_no = turns.setdefault(chunk.source_id, len(turns))
                key = f"{session_no}:{turn_no}"
                offset = word_offsets.get(key, 0)
                length = len(chunk.raw.split())
                word_offsets[key] = offset + length
                self.turn_lengths[key] = word_offsets[key]
                for start, end in raw_ranges(chunk.raw, 128):
                    body = chunk.raw[start:end]
                    if not body.strip():
                        continue
                    count = len(body.split())
                    text = f"{chunk.role}: {body}"
                    self.passages.append(Passage(len(self.passages), session_no, turn_no,
                                                 offset, offset + count, text, chunk.source_id,
                                                 (ordinal,), len(chunk.role) + 2 if offset else 0))
                    offset += count
        self.passages = tuple(self.passages)
        passages = [Memory(f"{scope_id}/passage/{p.ordinal}", scope_id, (), str(p.session_no), p.text)
                    for p in self.passages]
        self.sparse = BM25Retriever(passages)


class OnlineTextAdapter(TextAdapter):
    def __init__(self, *, body_weight=0.):
        super().__init__()
        if body_weight not in {0., .25}:
            raise ValueError("Unsupported online evidence policy")
        self.body_weight = body_weight

    def index(self, scope_id, memories, source):
        from .dialogue import dialogue_index
        # An online adapter belongs to one user and one index generation.
        self.domain = dialogue_index(OnlineConversationIndex(scope_id, memories, source))
        return self.domain

    def search(self, index, query, candidates, budget):
        from ..adapters.text.content_memory import ContentIndex, ContentQuery
        prepared = PassageQuery(index, query, candidates.fused, candidates.sparse, candidates.dense)
        ranking, policy = ContentQuery(prepared, query).rank()
        if self.body_weight:
            if not hasattr(index, "_online_content"):
                index._online_content = ContentIndex(index, ["user", "assistant"])
            lexical, _ = index._online_content.ranks(query, "stemmed")
            base = {i: rank for rank, (i, _) in enumerate(ranking, 1)}
            eligible = set(base) | set(np.flatnonzero(lexical <= 100))
            ranking = [(int(i), (1 - self.body_weight) / (20 + base.get(i, np.inf))
                                  + self.body_weight / (20 + lexical[i])) for i in eligible]
            ranking.sort(key=lambda item: (-item[1], item[0]))
        # Map fine passages to complete stable chunks. A repeated record does
        # not consume another result slot; different dates/roles remain distinct.
        from .dialogue import rank_chunks
        parent_ranking, dialogue_policy = rank_chunks(index, query, ranking)
        selected, seen, scores = [], set(), []
        for ordinal, score in parent_ranking:
            chunk = index.chunks[ordinal]
            identity = chunk.rendered
            if identity in seen:
                continue
            selected.append(index.memories[ordinal])
            seen.add(identity)
            scores.append(float(score))
            if len(selected) >= budget.max_documents:
                break
        # Keep response ordering explicit; scores are already nonincreasing.
        audit = {"online_v7_passage_ranking": True, "v7_policy": policy,
                 "explicit_dialogue_speakers": list(getattr(index, "explicit_dialogue_speakers", ())),
                 "dialogue_content_fusion": dialogue_policy,
                 "body_stemmed_rrf_weight": self.body_weight,
                 "complete_chunks": True, "global_character_truncation": False,
                 "response_scores": scores, "source_text_only": True}
        return Evidence("\n\n".join(m.text for m in selected), audit, tuple(selected), tuple(ranking))


class ProjectedProvider:
    def __init__(self, vectors, memories, chunks):
        self.vectors = vectors
        self.bodies = {m.text: chunk.raw for m, chunk in zip(memories, chunks)}

    def encode(self, texts, *, text_type, scope_id):
        if text_type == "document":
            texts = [self.bodies[text].strip() for text in texts]
        return self.vectors.encode(texts, text_type=text_type, scope_id=scope_id)


class HybridIndex:
    def __init__(self, user_id, chunks, vectors, *, body_weight=0.,
                 query_cache_entries=16, query_cache_bytes=4 * 1024 * 1024, window_context_chars=48000,
                 session_frames=True):
        self.chunks = tuple(chunks)
        memories = [Memory(c.id, user_id, (c.source_id,), c.session_id, c.rendered) for c in chunks]
        self.dates = {c.id: c.created_at for c in chunks}
        adapter = OnlineTextAdapter(body_weight=body_weight)
        self.engine = MemoryEngine(adapter=adapter,
                                   embedder=ProjectedProvider(vectors, memories, chunks),
                                   retrieval=RetrievalParameters(),
                                   # Selection is top-k whole records; no hidden HTTP character cap.
                                   budget=ContextBudget(max_chars=sum(len(m.text) + 2 for m in memories) + 1,
                                                        max_documents=100))
        self.engine.add(user_id, memories, self.chunks)
        self.user_id = user_id
        self.domain = adapter.domain
        self.evidence = EvidencePolicy(window_context_chars, session_frames)
        self.responses = ResponseCache(max_entries=query_cache_entries, max_bytes=query_cache_bytes)

    def _search(self, query):
        result = self.engine.search(self.user_id, query)
        scores = result.evidence.audit["response_scores"]
        records = tuple(CachedRecord(memory.id, memory.text, score, self.dates[memory.id])
                        for memory, score in zip(result.evidence.memories, scores))
        return self.evidence.compose(self.domain, records, query)

    def source_views(self, query):
        records = self.responses.peek(query)
        if records is None:
            return {}
        result = {}
        for record in records:
            if isinstance(record, SourceRecord):
                result[record.id] = json.loads(record.source_json)
        return result

    def search(self, query, top_k):
        return response(self.responses.get(query, lambda: self._search(query)), top_k)
