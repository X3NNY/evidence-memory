"""Use spare consumer context space to complete source turns without inference.

The original prefix and ranking are protected. This never truncates Search
responses; a configured consumer budget only bounds additional source text.
"""

from dataclasses import replace
import hashlib
import json

from .dialogue import DATE_PREFIX


def record_cost(record, rank):
    return len(record.content) + len(f"[Document {rank}]\n") + (2 if rank > 1 else 0)


def prefix_size(records, max_chars):
    count, size = 0, 0
    for rank, record in enumerate(records, 1):
        cost = record_cost(record, rank)
        if size + cost > max_chars:
            break
        count, size = rank, size + cost
    return count, size


def _body_date(chunk):
    match = DATE_PREFIX.match(chunk.raw)
    return (match.group(), match.end()) if match else ("", 0)


def render_window(index, anchor, before, after):
    """Only concatenate actual adjacent source text; keep anchor metadata."""
    chunk = index.chunks[anchor]
    date, offset = _body_date(chunk)
    body, previous = "", None
    for i, start, end in [*before, (anchor, offset, len(chunk.raw)), *after]:
        source = index.chunks[i]
        part = source.raw[start:end]
        contiguous = (previous is not None and previous[0] == source.source_id
                      and previous[1] == source.start + start)
        if body and part and not contiguous and not body[-1].isspace() and not part[0].isspace():
            body += " "
        body += part
        previous = (source.source_id, source.start + end)
    envelope = chunk.rendered[:-len(chunk.raw)] if chunk.raw else chunk.rendered
    return envelope + date + body


def enrich_windows(index, records, *, max_chars=48000, neighbor_turns=0):
    if max_chars < 0 or neighbor_turns < 0:
        raise ValueError("Window limits must be nonnegative")
    records = tuple(records)
    if not max_chars or not getattr(index, "explicit_dialogue_speakers", ()) or not hasattr(index, "turn_source_spans"):
        return records, {}, {"active": False}
    count, used = prefix_size(records, max_chars)
    if not count:
        return records, {}, {"active": False, "reason": "no_complete_prefix"}
    positions = {c.id: i for i, c in enumerate(index.chunks)}
    covered = {positions[r.id] for r in records[:count]}
    named = set(index.explicit_dialogue_speakers)
    turn_positions = {key: i for i, key in enumerate(index.turn_order)}
    proposals = []
    for rank, record in enumerate(records[:count]):
        anchor = positions[record.id]
        keys = index.chunk_turn_keys.get(anchor, ())
        if not keys:
            continue
        first, last = keys[0], keys[-1]
        before = [span for span in index.turn_source_spans[first] if span[0] < anchor]
        after = [span for span in index.turn_source_spans[last] if span[0] > anchor]
        options = []
        if before and index.turn_speakers[first] in named:
            # Very short continuation records otherwise lack the named source.
            orphan = len(keys) == 1 and len(index.chunks[anchor].raw.split()) < 128
            options.append((0 if orphan else 1, "before", before))
        if after and index.turn_speakers[last] in named:
            options.append((1, "after", after))
        if neighbor_turns and after:
            following = []
            start = turn_positions[last] + 1
            for key in index.turn_order[start:start + neighbor_turns]:
                if key.split(":", 1)[0] != last.split(":", 1)[0]:
                    break
                following.extend(index.turn_source_spans[key])
                options.append((2, "after", after + following.copy()))
        for priority, side, spans in options:
            if not any(i not in covered for i, _, _ in spans):
                continue
            date, _ = _body_date(index.chunks[anchor])
            if any(index.chunks[i].source_date != index.chunks[anchor].source_date or _body_date(index.chunks[i])[0] != date
                   for i, _, _ in spans):
                continue
            text = "".join(index.chunks[i].raw[a:b] for i, a, b in spans)
            if not text.strip() or len(text.split()) > 128 or len(text) > 2048:
                continue
            proposals.append((priority, len(text), rank, side, tuple(spans)))
    proposals.sort(key=lambda p: (p[0], p[1], p[2], p[3]))
    selected, output = {}, list(records)
    for _, _, rank, side, spans in proposals:
        anchor = positions[records[rank].id]
        before, after = selected.get(rank, ((), ()))
        current = before if side == "before" else after
        if len(spans) < len(current):
            continue
        next_before = spans if side == "before" else before
        next_after = spans if side == "after" else after
        text = render_window(index, anchor, next_before, next_after)
        total_words = sum(len(index.chunks[i].raw[a:b].split()) for i, a, b in (*next_before, *next_after))
        if total_words > 128 or len(text) - len(records[rank].content) > 2048:
            continue
        cost = len(text) - len(output[rank].content)
        if cost <= 0 or used + cost > max_chars:
            continue
        used += cost
        selected[rank] = (next_before, next_after)
        output[rank] = replace(records[rank], content=text)
    provenance = {}
    for rank, (before, after) in selected.items():
        anchor = positions[records[rank].id]
        spans = [(index.chunks[i].id, a, b) for i, a, b in before]
        spans.append((records[rank].id, 0, len(index.chunks[anchor].raw)))
        spans.extend((index.chunks[i].id, a, b) for i, a, b in after)
        identity = json.dumps({"anchor": records[rank].id, "before": before, "after": after}, separators=(",", ":"))
        mid = "window_" + hashlib.sha256(identity.encode()).hexdigest()
        output[rank] = replace(output[rank], id=mid)
        provenance[mid] = {"anchor_id": records[rank].id, "spans": spans,
                           "before": list(before), "after": list(after)}
    assert len(output) == len(records) and used <= max_chars
    assert all(a.score == b.score and a.created_at == b.created_at for a, b in zip(records, output))
    assert prefix_size(output, max_chars)[0] >= count
    return tuple(output), provenance, {"active": True, "baseline_prefix_records": count,
        "baseline_prefix_chars": prefix_size(records, max_chars)[1], "enriched_prefix_chars": used,
        "enriched_records": len(selected), "consumer_budget_chars": max_chars,
        "ranking_unchanged": True, "baseline_prefix_source_preserved": True}
