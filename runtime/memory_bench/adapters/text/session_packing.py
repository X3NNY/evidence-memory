"""Exact-budget session documents with explicit source gaps."""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
import hashlib

import numpy as np

from .coherent_memory import CoherentQuery
from ...core.models import Memory
from ...core.context import render_context


GAP = "\n[...]\n"


def pack_sessions(index, ranked, *, protected_passages=None, max_chars=48000, max_documents=100):
    """Keep one dated document per session; gaps never imply adjacency.

    Marginal costs include actual role prefixes, gap markers, date/document
    headers and separators. Every protected passage is retained before refill.
    """
    if max_chars < 1 or max_documents < 1:
        raise ValueError("Context budgets must be positive")
    ranked = list(ranked)
    if len({i for i, _ in ranked}) != len(ranked):
        raise ValueError("Passages must not be repeated")
    if any(not 0 <= i < len(index.passages) or not np.isfinite(score) for i, score in ranked):
        raise ValueError("Invalid passage ranking")
    protected = set(protected_passages or ())
    if any(not isinstance(i, (int, np.integer)) or not 0 <= i < len(index.passages) for i in protected):
        raise ValueError("Protected passage ordinal is outside this scope")
    known = {i for i, _ in ranked}
    ranked += [(i, 0.) for i in sorted(protected-known)]
    header_costs = [0]
    for n in range(1, max_documents+1):
        header_costs.append(header_costs[-1]+len(f"[Document {n}]\n"))

    def separator(left, right):
        if left is None or right is None:
            return 0
        previous, passage = index.passages[left], index.passages[right]
        if index.adjacent(left, 1) != right:
            return len(GAP)
        return 1-passage.continuation_prefix_chars if previous.turn_no == passage.turn_no else 1

    groups = defaultdict(list)
    selected = set(protected)
    body_cost, dates_cost = 0, 0
    for i in sorted(selected):
        passage = index.passages[i]
        group = groups[passage.session_no]
        body_cost += len(passage.text)+separator(group[-1] if group else None, i)
        if not group:
            dates_cost += len(index.date_headers[passage.session_no])
        group.append(i)
    documents = len(groups)

    def total_cost(body, dates, count):
        return body+dates+header_costs[count]+2*max(0, count-1)

    if documents > max_documents or total_cost(body_cost, dates_cost, documents) > max_chars:
        raise ValueError("Protected session documents exceed the context budget")
    # A bridging passage can replace a gap with shorter adjacency separators.
    for _ in range(2):
        changed = False
        for i, score in ranked:
            if i in selected:
                continue
            passage = index.passages[i]
            existing = groups.get(passage.session_no, [])
            position = bisect_left(existing, i)
            left = existing[position-1] if position else None
            right = existing[position] if position < len(existing) else None
            new_count = documents+int(not existing)
            if new_count > max_documents:
                continue
            new_body = body_cost+len(passage.text)+separator(left, i)+separator(i, right)-separator(left, right)
            new_dates = dates_cost+(len(index.date_headers[passage.session_no]) if not existing else 0)
            if total_cost(new_body, new_dates, new_count) > max_chars:
                continue
            groups[passage.session_no].insert(position, i)
            selected.add(i)
            body_cost, dates_cost, documents = new_body, new_dates, new_count
            changed = True
        if not changed:
            break
    priorities = {i: (score, rank) for rank, (i, score) in enumerate(ranked)}
    ordered = sorted(groups.values(), key=lambda group: (-max(priorities[i][0] for i in group),
                                                         min(priorities[i][1] for i in group)))
    memories, ranges, gaps = [], [], 0
    for group in ordered:
        passages = [index.passages[i] for i in group]
        body = passages[0].text
        for previous, passage in zip(passages, passages[1:]):
            if index.adjacent(previous.ordinal, 1) != passage.ordinal:
                body += GAP+passage.text
                gaps += 1
            elif previous.turn_no == passage.turn_no:
                body += " "+passage.text[passage.continuation_prefix_chars:]
            else:
                body += "\n"+passage.text
        text = index.date_headers[passages[0].session_no]+body
        identity = ",".join(str(i) for i in group)+"\0"+text
        memory_id = f"{index.scope_id}/session_evidence/{hashlib.sha256(identity.encode()).hexdigest()[:24]}"
        memories.append(Memory(memory_id, index.scope_id, tuple(dict.fromkeys(p.source_id for p in passages)),
                               str(passages[0].session_no), text))
        ranges.extend({"turn_key": p.turn_key, "start": p.word_start, "end": p.word_end,
                       "passage_ordinal": p.ordinal} for p in passages)
    memory_rows = {(m.scope_id, m.id): {"memory_id": m.id, "scope_id": m.scope_id, "text": m.text,
                   "source_ids": list(m.source_ids), "session_id": m.session_id} for m in memories}
    row = {"scope_id": index.scope_id, "ranked": [{"memory_id": m.id} for m in memories]}
    text, audit = render_context(row, memory_rows, {"max_chars": max_chars, "max_memories": max_documents, "packing": "complete"})
    if len(text) != total_cost(body_cost, dates_cost, documents) or audit["truncated_memory_ids"]:
        raise AssertionError("Session cost prediction differs from actual rendering")
    audit.update({"session_documents": True, "gap_marker": GAP, "explicit_gap_count": gaps,
                  "selected_passage_count": len(selected), "candidate_passage_count": len(ranked),
                  "selected_token_ranges": ranges, "protected_passage_count": len(protected),
                  "protected_tokens_preserved": protected <= selected,
                  "partial_source_turns_possible": True})
    return text, audit, memories


class SessionQuery(CoherentQuery):
    """Same v5 retrieval, with optional cost-aware refill in session documents."""

    def rank(self, *, protect=True, density=0.):
        if not isinstance(protect, bool) or density not in {0., .25, .5}:
            raise ValueError("Unsupported session-packing parameters")
        # Budget normalization addresses long turns fragmented across cache
        # chunks; short dialogue sources retain their original ranking.
        exponent = density if self.policy["effective_lexical_weight"] else 0.
        ranking = [(i, score / max(256, len(self.index.passages[i].text))**exponent)
                   for i, score in self.baseline]
        ranking.sort(key=lambda item: (-item[1], item[0]))
        return ranking, {"protect_incumbent": protect, "density": density, "effective_density": exponent,
                         "minimum_cost_chars": 256, "session_documents": True, "explicit_gap_marker": GAP}

    def pack(self, ranking, *, parameters, baseline_audit):
        protected = ({r["passage_ordinal"] for r in baseline_audit["selected_token_ranges"]}
                     if parameters["protect"] else None)
        return pack_sessions(self.index, ranking, protected_passages=protected)
