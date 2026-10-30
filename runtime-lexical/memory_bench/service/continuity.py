"""One-hop completion of explicitly recovered turns split across raw chunks."""

from collections import defaultdict


def split_turn_links(index):
    """Link adjacent parents only when the same named turn crosses the boundary.

    Turn keys already encode session boundaries. Equal dates alone, neighboring
    chunks, or a shared speaker in separate turns do not create an edge here.
    """
    speakers = set(getattr(index, "explicit_dialogue_speakers", ()))
    if not speakers:
        return {}
    turns = defaultdict(list)
    for passage in index.passages:
        if passage.text.partition(": ")[0] not in speakers:
            continue
        parents = turns[passage.turn_key]
        for ordinal in passage.chunk_ordinals:
            if not parents or parents[-1] != ordinal:
                parents.append(ordinal)
    links = defaultdict(set)
    for parents in turns.values():
        for left, right in zip(parents, parents[1:]):
            if left != right:
                links[left].add(right)
                links[right].add(left)
    return {i: tuple(sorted(neighbors)) for i, neighbors in links.items()}


def complete_turns(index, ranking):
    """Keep original evidence scores; a shared turn adds half the anchor score.

    Scores propagate once from the original ranking. New boosts never become
    anchors, so a chain of chunk boundaries cannot expand through a session.
    """
    if not hasattr(index, "_split_named_turn_links"):
        index._split_named_turn_links = split_turn_links(index)
    original = dict(ranking)
    scores = dict(original)
    boosted = 0
    for anchor, score in ranking:
        for other in index._split_named_turn_links.get(anchor, ()):
            value = .5 * score
            if value > scores.get(other, 0.):
                scores[other] = value
                boosted += 1
    order = sorted(scores, key=lambda i: (-scores[i], i))
    return [(i, scores[i]) for i in order], {"shared_turn_completion": True,
        "score_fraction": .5, "propagation_hops": 1, "boost_events": boosted,
        "raw_source_only": True}


def bridge_turns(index, ranking):
    """Fill one gap only when both source-continuous sides are strong anchors.

    Both anchors must exist in the original ranking and sit on opposite sides
    of the gap. Half the weaker anchor score limits the completion's priority.
    Isolated matches and iterated propagation never expand their neighbors.
    """
    if not hasattr(index, "_split_named_turn_links"):
        index._split_named_turn_links = split_turn_links(index)
    original = dict(ranking)
    scores = dict(original)
    events = []
    for middle, neighbors in index._split_named_turn_links.items():
        left = [(original[i], i) for i in neighbors if i < middle and i in original]
        right = [(original[i], i) for i in neighbors if i > middle and i in original]
        if not left or not right:
            continue
        a, b = max(left), max(right)
        value = .5 * min(a[0], b[0])
        if value > scores.get(middle, 0.):
            scores[middle] = value
            events.append({"parent_ordinal": middle, "left_anchor": a[1], "right_anchor": b[1]})
    order = sorted(scores, key=lambda i: (-scores[i], i))
    return [(i, scores[i]) for i in order], {"shared_turn_bridge": True,
        "score_fraction": .5, "propagation_hops": 1, "boosted_gaps": len(events),
        "bridges": events, "raw_source_only": True}
