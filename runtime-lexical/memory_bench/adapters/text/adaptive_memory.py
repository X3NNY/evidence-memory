"""Structure-aware passage scoring using source text and cached parent ranks."""

from __future__ import annotations

import re

import numpy as np


def structural_features(index):
    lengths = np.asarray(list(index.turn_lengths.values()))
    if not len(lengths) or lengths.sum() == 0:
        raise ValueError("A source index must contain nonempty turns")
    return {"median_turn_words": float(np.median(lengths)),
            "mean_turn_words": float(np.mean(lengths)),
            "p90_turn_words": float(np.percentile(lengths, 90)),
            "long_turn_fraction": float(np.mean(lengths > 128)),
            "long_turn_token_fraction": float(lengths[lengths > 256].sum() / lengths.sum())}


def adaptive_rank(prepared, query: str, *, personal_mode="none", session_expand=0,
                  neighbor_radius=0, lexical_policy="structure"):
    """Prefer turn-level lexical evidence when most source tokens are fragmented.

    An explicit user role can supply personal facts for first-person queries.
    Named dialogue speakers keep their original scores. Neither source IDs nor
    dataset identities nor relevance annotations are consulted.
    """
    if personal_mode not in {"none", "user_rank", "user_boost"}:
        raise ValueError("Unsupported personal memory mode")
    if session_expand not in {0, 5} or neighbor_radius not in {0, 1}:
        raise ValueError("Unsupported expansion parameters")
    if lexical_policy not in {"structure", "always", "never"}:
        raise ValueError("Unsupported lexical policy")
    index = prepared.index
    pressure = structural_features(index)["long_turn_token_fraction"]
    weight = .5 if lexical_policy == "always" or lexical_policy == "structure" and pressure > .5 else 0.
    personal = bool(re.search(r"\b(?:I|my|me|mine|we|our|ours)\b", query, flags=re.I))
    # The source loader always prefixes a passage with its actual speaker/role.
    users = np.asarray([p.text.startswith("user: ") for p in index.passages])
    lexical = prepared.lexical.copy()
    if personal and personal_mode != "none" and users.any():
        ordered = np.flatnonzero(users & np.isfinite(lexical))
        ordered = ordered[np.argsort(lexical[ordered], kind="stable")]
        lexical[ordered] = np.arange(1, len(ordered) + 1)
    session_ranks = np.asarray([prepared.session_ranks[p.session_no] for p in index.passages])
    eligible = prepared.parents <= 100
    if weight:
        eligible |= lexical <= 100
    if session_expand:
        eligible |= session_ranks <= session_expand
    scores = (1-weight) / (20+prepared.parents) + weight / (20+lexical)
    if session_expand:
        scores += .15 / (20+session_ranks)
    if personal and personal_mode == "user_boost":
        scores[users] *= 1.5
    if neighbor_radius:
        anchors = np.flatnonzero(eligible)
        anchors = anchors[np.argsort(-scores[anchors], kind="stable")[:40]]
        original = scores.copy()
        for ordinal in anchors:
            for direction in (-1, 1):
                neighbor = index.adjacent(int(ordinal), direction)
                if neighbor is not None:
                    eligible[neighbor] = True
                    scores[neighbor] = max(scores[neighbor], .5*original[ordinal])
    candidates = np.flatnonzero(eligible)
    order = candidates[np.argsort(-scores[candidates], kind="stable")]
    audit = {"long_turn_token_fraction": pressure, "lexical_weight": weight,
             "personal_query": personal, "explicit_user_passages": int(users.sum()),
             "personal_mode": personal_mode}
    return [(int(i), float(scores[i])) for i in order], audit
