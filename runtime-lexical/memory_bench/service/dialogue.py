"""Recover explicit serialized dialogue structure without model inference.

No person names, query terms, dataset IDs, answers or relevance labels are
configured here. Ambiguous/non-dialogue text keeps the existing role index.
"""

from collections import Counter, defaultdict
from copy import copy
from dataclasses import dataclass
import re

import numpy as np

from ..core.models import Memory
from ..core.retrieval import BM25Retriever
from ..adapters.text.structure import Passage


LABEL = re.compile(r"(?<![\w/])([A-Z][a-z]{1,30}(?:[ -][A-Z][a-z]{1,30}){0,2}):(?=\s)")
METADATA = frozenset({"date", "time", "timestamp", "conversation", "note", "notes", "warning", "title",
                      "summary", "source", "document", "caption", "image", "user", "assistant", "system",
                      "developer", "tool", "question", "answer", "context", "overview", "details", "example"})
DATE_PREFIX = re.compile(
    r"^\s*(?:Date|Conversation timestamp):\s*(?:"
    r"(?:\d{1,2}:\d{2}\s*(?:am|pm)\s+on\s+)?\d{1,2}\s+[A-Za-z]+,?\s+\d{4}"
    r"|\d{4}-\d{2}-\d{2}(?:[T ][0-9:.]+(?:Z|[+-][0-9:]+)?)?)[ \t\r\n]*", re.I)


@dataclass(frozen=True)
class TurnSlice:
    speaker: str
    body: str
    new_turn: bool
    start: int
    end: int


def speakers_in(chunks):
    counts = Counter()
    sequences = []
    for chunk in chunks:
        labels = [m.group(1) for m in LABEL.finditer(chunk.raw) if m.group(1).casefold() not in METADATA]
        counts.update(labels)
        sequences.append(labels)
    speakers = tuple(name for name, count in counts.items() if count >= 3)
    if not 2 <= len(speakers) <= 8:
        return ()
    known = set(speakers)
    # Repeated alternating explicit prefixes in the same raw text are stronger
    # evidence than isolated capitalized headings or quoted name mentions.
    coherent = False
    for labels in sequences:
        labels = [name for name in labels if name in known]
        if len(labels) >= 4 and sum(a != b for a, b in zip(labels, labels[1:])) >= 3:
            coherent = True
            break
    return speakers if coherent else ()


def turn_slices(raw, speakers, *, fallback_role, carry=None):
    pattern = re.compile(r"(?<!\w)(" + "|".join(re.escape(n) for n in sorted(speakers, key=len, reverse=True)) + r"):\s+")
    date = DATE_PREFIX.match(raw)
    start = date.end() if date else 0
    matches = list(pattern.finditer(raw, start))
    records = []
    if not matches:
        if raw[start:].strip():
            records.append(TurnSlice(carry or fallback_role, raw[start:], carry is None, start, len(raw)))
        return records, date.group().strip() if date else None
    if raw[start:matches[0].start()].strip():
        records.append(TurnSlice(carry or fallback_role, raw[start:matches[0].start()], carry is None, start, matches[0].start()))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        if raw[match.end():end].strip():
            records.append(TurnSlice(match.group(1), raw[match.end():end], True, match.start(), end))
    return records, date.group().strip() if date else None


def dialogue_index(base):
    """Return a separate index view; raw chunks and document vectors stay intact."""
    speakers = speakers_in(base.chunks)
    if not speakers:
        return base
    from .retrieval import raw_ranges
    index = copy(base)
    index.passages, index.turn_lengths = [], {}
    index.turn_source_spans, index.turn_speakers, index.chunk_turn_keys, index.turn_order = {}, {}, defaultdict(list), []
    groups = defaultdict(list)
    for ordinal, chunk in enumerate(index.chunks):
        groups[int(index.chunk_sessions[ordinal])].append((ordinal, chunk))
    for session_no, chunks in groups.items():
        turn_no, word_offset = -1, 0
        prior_source, prior_end, prior_date, prior_speaker, prior_role = None, None, None, None, None
        for chunk_no, chunk in chunks:
            date_match = DATE_PREFIX.match(chunk.raw)
            date = date_match.group().strip() if date_match else None
            continuation = (chunk.source_id == prior_source and chunk.start == prior_end
                            or date is not None and date == prior_date and chunk.role == prior_role)
            carry = prior_speaker if continuation else None
            slices, _ = turn_slices(chunk.raw, speakers, fallback_role=chunk.role, carry=carry)
            for piece in slices:
                if piece.new_turn or turn_no < 0:
                    turn_no += 1
                    word_offset = 0
                key = f"{session_no}:{turn_no}"
                if key not in index.turn_source_spans:
                    index.turn_source_spans[key] = []
                    index.turn_speakers[key] = piece.speaker
                    index.turn_order.append(key)
                index.turn_source_spans[key].append((chunk_no, piece.start, piece.end))
                index.chunk_turn_keys[chunk_no].append(key)
                for begin, end in raw_ranges(piece.body, 128):
                    body = piece.body[begin:end]
                    if not body.strip():
                        continue
                    words = len(body.split())
                    index.passages.append(Passage(len(index.passages), session_no, turn_no,
                        word_offset, word_offset + words, f"{piece.speaker}: {body}", chunk.source_id,
                        (chunk_no,), len(piece.speaker) + 2 if word_offset else 0))
                    word_offset += words
                    index.turn_lengths[f"{session_no}:{turn_no}"] = word_offset
                prior_speaker = piece.speaker
            prior_source, prior_end, prior_date, prior_role = chunk.source_id, chunk.end, date, chunk.role
    if not index.passages:
        return base
    index.passages = tuple(index.passages)
    memories = [Memory(f"{index.scope_id}/passage/{p.ordinal}", index.scope_id, (), str(p.session_no), p.text)
                for p in index.passages]
    index.sparse = BM25Retriever(memories)
    index.explicit_dialogue_speakers = speakers
    return index


class DialogueContentIndex:
    """Body topics at the complete cached-chunk scale; no new embeddings."""

    def __init__(self, index, speakers):
        from ..core.retrieval import tokenize
        self.participant_tokens = {term for name in speakers for term in tokenize(name)}
        memories = [Memory(m.id, m.scope_id, (), m.session_id, " ".join(self.normalize(c.raw)))
                    for m, c in zip(index.memories, index.chunks)]
        self.sparse = BM25Retriever(memories)

    def normalize(self, text):
        from ..core.retrieval import tokenize
        from ..adapters.text.content_memory import STOPWORDS, stem
        return [stem(t) for t in tokenize(text) if t not in STOPWORDS and t not in self.participant_tokens]


def rank_chunks(index, query, passage_ranking):
    """Equal-rank fusion of passage evidence and source-wide body topics.

    Named serialized dialogue is detected from raw text alone. Ordinary role
    messages and queries with no named participant retain their existing order
    and scores. There is no speaker multiplier or hard speaker filter.
    """
    from ..adapters.text.attributed_memory import queried_speakers

    fine, seen = [], set()
    for passage_no, score in passage_ranking:
        for ordinal in index.passages[passage_no].chunk_ordinals:
            if ordinal not in seen:
                fine.append((ordinal, float(score)))
                seen.add(ordinal)
    speakers = getattr(index, "explicit_dialogue_speakers", ())
    if not queried_speakers(query, speakers):
        return fine, {"active": False, "raw_source_only": True}
    if not hasattr(index, "_dialogue_parent_content"):
        index._dialogue_parent_content = DialogueContentIndex(index, speakers)
    content = index._dialogue_parent_content
    terms = content.normalize(query)
    lexical = {index.memory_ordinals[r.memory.id]: r.rank
               for r in content.sparse.search(" ".join(terms), 100) if r.score > 0}
    ranks = {ordinal: rank for rank, (ordinal, _) in enumerate(fine, 1)}
    scores = {i: .5 / (20 + ranks.get(i, np.inf)) + .5 / (20 + lexical.get(i, np.inf))
              for i in set(ranks) | set(lexical)}
    order = sorted(scores, key=lambda i: (-scores[i], i))
    from .continuity import bridge_turns
    ranked, continuity = bridge_turns(index, [(i, float(scores[i])) for i in order])
    return ranked, {
        "active": True, "passage_weight": .5, "parent_body_weight": .5, "rrf_k": 20,
        "body_matches": len(lexical),
        "speaker_preference": False, "raw_source_only": True, **continuity}
