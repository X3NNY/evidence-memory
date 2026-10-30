"""Attribute retrieval to explicitly queried conversation speakers."""

from __future__ import annotations

import re

import numpy as np

from .adaptive_memory import adaptive_rank


RESERVED_ROLES = frozenset({"user", "assistant", "system", "developer", "tool", "speaker"})


def queried_speakers(query: str, speakers: list[str]) -> tuple[str, ...]:
    """Select the first named participant, or both in an explicit joint query.

    '... Alice ... with Bob' asks about Alice. Explicit 'both', comparisons,
    or 'Alice and Bob' keep both participants. Matches require full names, so
    Ann cannot match Anna. Source identifiers are never consulted.
    """
    found = []
    for speaker in dict.fromkeys(speakers):
        if not speaker or speaker.casefold() in RESERVED_ROLES:
            continue
        match = re.search(r"(?<!\w)" + re.escape(speaker) + r"(?!\w)", query, flags=re.I)
        if match:
            found.append((match.start(), match.end(), speaker))
    found.sort(key=lambda item: (item[0], -(item[1]-item[0]), item[2]))
    # An overlapping short name must not duplicate its full-name match.
    disjoint = []
    for item in found:
        if not disjoint or item[0] >= disjoint[-1][1]:
            disjoint.append(item)
    if not disjoint:
        return ()
    joint = bool(re.search(r"\b(?:both|compare|comparison|common|together|each)\b", query, flags=re.I))
    joint |= any(re.search(r"\band\b", query[left[1]:right[0]], flags=re.I)
                 for left, right in zip(disjoint, disjoint[1:]))
    return tuple(item[2] for item in (disjoint if joint else disjoint[:1]))


class AttributedQuery:
    """Prepared ranks see source structure and question text only."""

    def __init__(self, prepared, query: str):
        self.prepared = prepared
        self.index = prepared.index
        self.query = query
        self.v4_ranked, self.v4_policy = adaptive_rank(prepared, query, personal_mode="user_boost",
                                                       session_expand=5, neighbor_radius=0)
        self.roles = [p.text.partition(": ")[0] for p in self.index.passages]
        self.named_targets = queried_speakers(query, self.roles)
        self.personal = self.v4_policy["personal_query"]
        targets = ("user",) if self.personal and "user" in self.roles else self.named_targets
        self.targets = targets
        self.target_mask = np.asarray([role in targets for role in self.roles])
        self.lexical = prepared.lexical.copy()
        # Same within-user lexical ranks as v4. Named-speaker ranks are enabled
        # only when a small lexical interpolation is explicitly requested.
        self.target_lexical = self.lexical.copy()
        if self.target_mask.any():
            ordered = np.flatnonzero(self.target_mask & np.isfinite(self.lexical))
            ordered = ordered[np.argsort(self.lexical[ordered], kind="stable")]
            self.target_lexical[ordered] = np.arange(1, len(ordered)+1)
        if targets == ("user",):
            self.lexical = self.target_lexical.copy()
        self.sessions = np.asarray([prepared.session_ranks[p.session_no] for p in self.index.passages])

    def rank(self, *, speaker_gain=1.5, named_lexical_weight=0., candidate_k=100, name_attribution=True):
        if speaker_gain not in {1.25, 1.5, 2.} or named_lexical_weight not in {0., .15} or candidate_k not in {100, 200}:
            raise ValueError("Unsupported attributed retrieval parameters")
        weight = self.v4_policy["lexical_weight"]
        lexical = self.lexical
        if name_attribution and self.named_targets and self.targets != ("user",) and weight == 0 and named_lexical_weight:
            weight = named_lexical_weight
            lexical = self.target_lexical
        eligible = (self.prepared.parents <= candidate_k) | (self.sessions <= 5)
        if weight:
            eligible |= lexical <= candidate_k
        scores = (1-weight) / (20+self.prepared.parents) + weight / (20+lexical) + .15 / (20+self.sessions)
        target_mask = self.target_mask if name_attribution or self.targets == ("user",) else np.zeros(len(scores), dtype=bool)
        scores[target_mask] *= speaker_gain
        candidates = np.flatnonzero(eligible)
        order = candidates[np.argsort(-scores[candidates], kind="stable")]
        audit = {**self.v4_policy, "selected_speakers": list(self.targets) if target_mask.any() else [], "speaker_gain": speaker_gain,
                 "effective_lexical_weight": weight, "candidate_k": candidate_k,
                 "named_lexical_weight": named_lexical_weight, "name_attribution": name_attribution}
        return [(int(i), float(scores[i])) for i in order], audit

    def replay_v4(self):
        """Recompute the old scoring path independently before any experiment."""
        scores = ((1-self.v4_policy["lexical_weight"]) / (20+self.prepared.parents)
                  + self.v4_policy["lexical_weight"] / (20+self.lexical) + .15 / (20+self.sessions))
        users = np.asarray([role == "user" for role in self.roles])
        if self.personal:
            scores[users] *= 1.5
        eligible = (self.prepared.parents <= 100) | (self.sessions <= 5)
        if self.v4_policy["lexical_weight"]:
            eligible |= self.lexical <= 100
        order = np.flatnonzero(eligible)
        order = order[np.argsort(-scores[order], kind="stable")]
        ranked = [(int(i), float(scores[i])) for i in order]
        if ranked != self.v4_ranked:
            raise AssertionError("v4 passage rankings changed")
        return ranked
