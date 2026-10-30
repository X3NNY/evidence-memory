"""Lightweight conversation context edges over cached retrieval signals."""

from __future__ import annotations

import numpy as np

from .attributed_memory import AttributedQuery


V5_PARAMETERS = {"speaker_gain": 2., "named_lexical_weight": 0., "candidate_k": 100,
                 "name_attribution": False}


class CoherentQuery:
    """Propagate relevance to source context, without any relevance labels."""

    def __init__(self, prepared, query: str):
        self.index = prepared.index
        self.base = AttributedQuery(prepared, query)
        self.baseline, self.policy = self.base.rank(**V5_PARAMETERS)
        count = len(self.index.passages)
        self.scores = np.zeros(count)
        self.eligible = np.zeros(count, dtype=bool)
        for ordinal, score in self.baseline:
            self.scores[ordinal] = score
            self.eligible[ordinal] = True
        # Numerical session/turn positions and actual speaker prefixes establish
        # edges. Dataset IDs, encoded source IDs and QA annotations are unused.
        session_best, thread_best = {}, {}
        for ordinal, score in self.baseline:
            passage = self.index.passages[ordinal]
            role = self.base.roles[ordinal]
            session_best[passage.session_no] = max(session_best.get(passage.session_no, 0.), score)
            key = passage.session_no, role
            thread_best[key] = max(thread_best.get(key, 0.), score)
        self.session_support = np.asarray([session_best.get(p.session_no, 0.) for p in self.index.passages])
        self.thread_support = np.asarray([thread_best.get((p.session_no, role), 0.)
                                          for p, role in zip(self.index.passages, self.base.roles)])
        self.opening = np.asarray([p.turn_no < 3 for p in self.index.passages])

    def rank(self, *, edges="opening", strength=.5):
        if edges not in {"opening", "speaker_thread", "both"} or strength not in {.25, .5, .75}:
            raise ValueError("Unsupported conversation-context parameters")
        scores = self.scores.copy()
        eligible = self.eligible.copy()
        promoted = np.zeros(len(scores), dtype=bool)
        if edges in {"opening", "both"}:
            support = strength * self.session_support
            promoted |= self.opening & (support > scores)
            scores[self.opening] = np.maximum(scores[self.opening], support[self.opening])
            eligible |= self.opening & (self.session_support > 0)
        if edges in {"speaker_thread", "both"}:
            support = strength * self.thread_support
            promoted |= support > scores
            scores = np.maximum(scores, support)
            eligible |= self.thread_support > 0
        candidates = np.flatnonzero(eligible)
        order = candidates[np.argsort(-scores[candidates], kind="stable")]
        audit = {"edges": edges, "strength": strength, "opening_turns": 3,
                 "baseline_candidate_passages": len(self.baseline), "candidate_passages": len(order),
                 "promoted_passages": int(promoted.sum()), "raw_context_only": True}
        return [(int(i), float(scores[i])) for i in order], audit

    def replay(self):
        return list(self.baseline)
