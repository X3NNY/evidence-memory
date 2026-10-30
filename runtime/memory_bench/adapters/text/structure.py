"""Label-free conversation hierarchy over the existing cached chunk index.

Raw turns are split at sentence/word boundaries for output only. Their dense
signal is inherited from cached parent chunks; no new vectors or facts are
generated. IDs are used only for lookup/provenance, never as text features.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import re

import numpy as np

from ...core.models import Memory
from ...core.context import render_context
from ...core.retrieval import BM25Retriever, RankedMemory


@dataclass(frozen=True)
class Passage:
    ordinal: int
    session_no: int
    turn_no: int
    word_start: int
    word_end: int
    text: str
    source_id: str
    chunk_ordinals: tuple[int, ...]
    continuation_prefix_chars: int = 0

    @property
    def turn_key(self) -> str:
        return f"{self.session_no}:{self.turn_no}"


def word_ranges(words: list[str], max_words: int):
    if max_words < 1:
        raise ValueError("max_words must be positive")
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        if end < len(words):
            for boundary in range(end, start + max_words // 2, -1):
                if re.search(r"[.!?][\"')\]]*$", words[boundary - 1]):
                    end = boundary
                    break
        yield start, end
        start = end


class ConversationIndex:
    def __init__(self, scope_id: str, memories: tuple[Memory, ...], sessions: list,
                 *, chunk_words: int = 256, passage_words: int = 128, cache_aligned: bool = False):
        if any(m.scope_id != scope_id for m in memories):
            raise ValueError("Conversation index requires one isolated scope")
        self.scope_id = scope_id
        self.cache_aligned = cache_aligned
        self.memories = memories
        self.memory_ordinals = {m.id: i for i, m in enumerate(memories)}
        self.dates = tuple(timestamp for timestamp, turns in sessions)
        self.date_headers = tuple(f"Date: {timestamp}\n" if timestamp else "" for timestamp in self.dates)
        self.turn_lengths = {}
        self.passages = []
        self.chunk_sessions = np.full(len(memories), -1, dtype=np.int64)
        for session_no, (timestamp, turns) in enumerate(sessions):
            prefix = (f"Date: {timestamp}\n" if timestamp else "").split()[:max(0, chunk_words - 1)]
            budget = max(1, chunk_words - len(prefix))
            tokens, turn_records = [], []
            for turn_no, (source_id, speaker, text) in enumerate(turns):
                words = text.split()
                self.turn_lengths[f"{session_no}:{turn_no}"] = len(words)
                turn_records.append((turn_no, source_id, speaker, words, len(tokens)))
                tokens.extend(words)
            chunk_lookup = []
            for chunk_no, offset in enumerate(range(0, len(tokens), budget)):
                mid = f"{scope_id}/s{session_no:04d}/m{chunk_no:04d}"
                ordinal = self.memory_ordinals[mid]
                if memories[ordinal].text != " ".join(prefix + tokens[offset:offset + budget]):
                    raise ValueError(f"Raw text differs from cached index: {mid}")
                self.chunk_sessions[ordinal] = session_no
                chunk_lookup.append(ordinal)
            for turn_no, source_id, speaker, words, offset in turn_records:
                ranges = []
                for begin, finish in word_ranges(words, passage_words):
                    if cache_aligned:
                        start = begin
                        while start < finish:
                            end = min(finish, ((offset + start) // budget + 1) * budget - offset)
                            ranges.append((start, end))
                            start = end
                    else:
                        ranges.append((begin, finish))
                for start, end in ranges:
                    parents = tuple(chunk_lookup[i] for i in range((offset + start) // budget,
                                                                   (offset + end - 1) // budget + 1))
                    body = " ".join(words[start:end])
                    # A split continuation still identifies its original speaker.
                    if start:
                        body = f"{speaker}: " + body
                    self.passages.append(Passage(len(self.passages), session_no, turn_no, start, end,
                                                 body, source_id, parents, len(f"{speaker}: ") if start else 0))
        self.passages = tuple(self.passages)
        if (self.chunk_sessions < 0).any():
            raise ValueError("Raw conversations do not cover every cached chunk")
        index_memories = [Memory(f"{scope_id}/passage/{p.ordinal}", scope_id, (), str(p.session_no), p.text)
                          for p in self.passages]
        self.sparse = BM25Retriever(index_memories)

    def adjacent(self, ordinal: int, direction: int) -> int | None:
        other = ordinal + direction
        if 0 <= other < len(self.passages) and self.passages[other].session_no == self.passages[ordinal].session_no:
            return other
        return None



class PassageQuery:
    """Prepared once per query; all inputs are label-free retrieval signals."""

    def __init__(self, index: ConversationIndex, query: str, parents: list[RankedMemory],
                 sparse: list[RankedMemory], dense: list[RankedMemory]):
        self.index = index
        parent_ranks = np.full(len(index.memories), np.inf)
        for item in parents:
            if item.memory.scope_id != index.scope_id:
                raise ValueError("Parent rankings cannot cross scopes")
            parent_ranks[index.memory_ordinals[item.memory.id]] = item.rank
        self.parents = np.asarray([min(parent_ranks[list(p.chunk_ordinals)]) for p in index.passages])
        self.lexical = np.full(len(index.passages), np.inf)
        for item in index.sparse.search(query, max(1, len(index.passages))):
            if item.score > 0:
                # Index memory order and passage order are identical.
                ordinal = int(item.memory.id.rsplit("/", 1)[1])
                self.lexical[ordinal] = item.rank
        session_scores = np.zeros(len(index.dates))
        for ranking in (sparse, dense):
            values = defaultdict(list)
            for item in ranking:
                if item.memory.scope_id != index.scope_id:
                    raise ValueError("Session rankings cannot cross scopes")
                if item.score > 0:
                    session_no = int(index.chunk_sessions[index.memory_ordinals[item.memory.id]])
                    values[session_no].append(1 / (60 + item.rank))
            scores = {sid: sorted(parts, reverse=True)[0] + (0.25 * sorted(parts, reverse=True)[1] if len(parts) > 1 else 0)
                      for sid, parts in values.items()}
            for rank, sid in enumerate(sorted(scores, key=lambda sid: (-scores[sid], sid)), 1):
                session_scores[sid] += 1 / (60 + rank)
        session_order = sorted(range(len(index.dates)), key=lambda sid: (-session_scores[sid], sid))
        self.session_ranks = np.empty(len(index.dates), dtype=np.int64)
        for rank, sid in enumerate(session_order, 1):
            self.session_ranks[sid] = rank

    def rank(self, *, lexical_weight: float, session_expand: int, neighbor_radius: int):
        if lexical_weight not in {0, .25, .5} or session_expand not in {0, 5} or neighbor_radius not in {0, 1}:
            raise ValueError("Unsupported structured retrieval parameters")
        session_rank = np.asarray([self.session_ranks[p.session_no] for p in self.index.passages])
        eligible = self.parents <= 100
        if lexical_weight:
            eligible |= self.lexical <= 100
        if session_expand:
            eligible |= session_rank <= session_expand
        scores = (1 - lexical_weight) / (20 + self.parents) + lexical_weight / (20 + self.lexical)
        if session_expand:
            scores += .15 / (20 + session_rank)
        if neighbor_radius:
            anchors = np.flatnonzero(eligible)
            anchors = anchors[np.argsort(-scores[anchors], kind="stable")[:40]]
            original = scores.copy()
            for ordinal in anchors:
                for direction in (-1, 1):
                    neighbor = self.index.adjacent(int(ordinal), direction)
                    if neighbor is not None:
                        eligible[neighbor] = True
                        scores[neighbor] = max(scores[neighbor], .5 * original[ordinal])
        candidates = np.flatnonzero(eligible)
        order = candidates[np.argsort(-scores[candidates], kind="stable")]
        return [(int(ordinal), float(scores[ordinal])) for ordinal in order]


class ProtectedContextTooLarge(ValueError):
    """The legacy evidence cannot fit with the requested formatting/budget."""


def pack_passages(index: ConversationIndex, ranked: list[tuple[int, float]], *, max_chars=48000,
                  max_documents=100, protected_chunks: set[str] | None = None,
                  merge_continuations: bool = False):
    """Greedily select whole passages, coalescing adjacent source text exactly.

    Incremental costs include all document/date headers and separators. A second
    pass can fill space freed when a bridging passage merges adjacent blocks.
    """
    if max_chars < 1 or max_documents < 1:
        raise ValueError("Context budgets must be positive")
    if len({ordinal for ordinal, score in ranked}) != len(ranked):
        raise ValueError("Passages must not be repeated")
    if any(not 0 <= ordinal < len(index.passages) or not np.isfinite(score) for ordinal, score in ranked):
        raise ValueError("Invalid passage ranking")
    header_cost = [0]
    for n in range(1, max_documents + 1):
        header_cost.append(header_cost[-1] + len(f"[Document {n}]\n"))
    protected_chunks = protected_chunks or set()
    if protected_chunks and not index.cache_aligned:
        raise ValueError("Protected source tokens require cache-aligned passages")
    if not protected_chunks <= set(index.memory_ordinals):
        raise ValueError("Protected chunk IDs are outside this isolated scope")
    protected_ordinals = {index.memory_ordinals[mid] for mid in protected_chunks}
    selected = {p.ordinal for p in index.passages if protected_ordinals.intersection(p.chunk_ordinals)}
    priority_map = dict(ranked)
    for ordinal in sorted(selected - set(priority_map)):
        ranked.append((ordinal, 0.))
    selected_initially = set(selected)
    count, blocks, body_chars, date_chars = 0, 0, 0, 0
    continuation_savings = 0
    for ordinal in sorted(selected):
        passage = index.passages[ordinal]
        left = index.adjacent(ordinal, -1)
        count += 1
        body_chars += len(passage.text)
        if left not in selected:
            blocks += 1
            date_chars += len(index.date_headers[passage.session_no])
        elif merge_continuations and index.passages[left].turn_no == passage.turn_no:
            continuation_savings += passage.continuation_prefix_chars
    if (blocks > max_documents or body_chars + count - blocks + date_chars - continuation_savings
            + header_cost[min(blocks, max_documents)] + 2 * max(0, blocks - 1) > max_chars):
        raise ProtectedContextTooLarge("Protected raw context exceeds the budget")
    for _ in range(2):
        changed = False
        for ordinal, score in ranked:
            if ordinal in selected:
                continue
            passage = index.passages[ordinal]
            left, right = index.adjacent(ordinal, -1), index.adjacent(ordinal, 1)
            delta_blocks = 1 - int(left in selected) - int(right in selected)
            new_blocks = blocks + delta_blocks
            if new_blocks > max_documents:
                continue
            new_body = body_chars + len(passage.text)
            new_dates = date_chars + delta_blocks * len(index.date_headers[passage.session_no])
            new_savings = continuation_savings
            if merge_continuations:
                if left in selected and index.passages[left].turn_no == passage.turn_no:
                    new_savings += passage.continuation_prefix_chars
                if right in selected and index.passages[right].turn_no == passage.turn_no:
                    new_savings += index.passages[right].continuation_prefix_chars
            total = new_body + count + 1 - new_blocks + new_dates - new_savings + header_cost[new_blocks] + 2 * max(0, new_blocks - 1)
            if total > max_chars:
                continue
            selected.add(ordinal)
            count, blocks, body_chars, date_chars = count + 1, new_blocks, new_body, new_dates
            continuation_savings = new_savings
            changed = True
        if not changed:
            break
    groups = []
    for ordinal in sorted(selected):
        if groups and index.adjacent(groups[-1][-1], 1) == ordinal:
            groups[-1].append(ordinal)
        else:
            groups.append([ordinal])
    priorities = {ordinal: (score, rank) for rank, (ordinal, score) in enumerate(ranked)}
    groups.sort(key=lambda group: (-max(priorities[i][0] for i in group), min(priorities[i][1] for i in group)))
    memories, ranges = [], []
    for group in groups:
        passages = [index.passages[i] for i in group]
        body = passages[0].text
        for previous, passage in zip(passages, passages[1:]):
            if merge_continuations and previous.turn_no == passage.turn_no:
                body += " " + passage.text[passage.continuation_prefix_chars:]
            else:
                body += "\n" + passage.text
        text = index.date_headers[passages[0].session_no] + body
        identity = ",".join(str(i) for i in group) + "\0" + text
        memory_id = f"{index.scope_id}/evidence/{hashlib.sha256(identity.encode()).hexdigest()[:24]}"
        memories.append(Memory(memory_id, index.scope_id, tuple(dict.fromkeys(p.source_id for p in passages)),
                               str(passages[0].session_no), text))
        ranges.extend({"turn_key": p.turn_key, "start": p.word_start, "end": p.word_end,
                       "passage_ordinal": p.ordinal} for p in passages)
    memory_rows = {(m.scope_id, m.id): {"memory_id": m.id, "scope_id": m.scope_id, "text": m.text,
                                      "source_ids": list(m.source_ids), "session_id": m.session_id} for m in memories}
    row = {"scope_id": index.scope_id, "ranked": [{"memory_id": m.id} for m in memories]}
    context, audit = render_context(row, memory_rows,
                                    {"max_chars": max_chars, "max_memories": max_documents, "packing": "complete"})
    predicted = body_chars + count - blocks + date_chars - continuation_savings + header_cost[blocks] + 2 * max(0, blocks - 1)
    if len(context) != predicted or len(memories) != blocks or audit["truncated_memory_ids"]:
        raise AssertionError("Incremental context cost differs from actual rendering")
    audit.update({"selected_passage_count": len(selected), "selected_token_ranges": ranges,
                  "candidate_passage_count": len(ranked), "partial_source_turns_possible": True,
                  "protected_passage_count": len(selected_initially), "protected_chunk_count": len(protected_chunks),
                  "protected_tokens_preserved": selected_initially <= selected,
                  "continuation_chars_saved": continuation_savings})
    return context, audit, memories


