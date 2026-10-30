"""Lossless session frames and budgeted complete-turn evidence refill."""

from collections import OrderedDict, defaultdict
import hashlib
import json
import math
import re

from ..core.models import Memory
from ..core.retrieval import BM25Retriever, tokenize
from .dialogue import DATE_PREFIX
from .responses import CachedRecord
from .windows import prefix_size, record_cost


GAP = "\n[...]\n"


def merge_ranges(ranges):
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def body_offset(chunk):
    date = DATE_PREFIX.match(chunk.raw)
    return date.end() if date else 0


def source_group(chunk):
    offset = body_offset(chunk)
    return chunk.session_id, chunk.role, chunk.source_date, chunk.raw[:offset]


def add_spans(target, spans, index):
    for ordinal, start, end in spans:
        start = max(start, body_offset(index.chunks[ordinal]))
        if start < end:
            target[ordinal] = merge_ranges([*target.get(ordinal, ()), (start, end)])


def covered_range(ranges, start, end):
    return any(a <= start and end <= b for a, b in ranges)


def render_frame(index, group, next_chunks):
    anchor = index.chunks[group["anchor"]]
    envelope = anchor.rendered[:-len(anchor.raw)]
    if group.get("omit_role"):
        envelope = envelope.removesuffix(anchor.role + ": ")
    header = envelope + anchor.raw[:body_offset(anchor)]
    body, spans, gaps, previous = "", [], [], None
    for ordinal in sorted(group["ranges"]):
        chunk = index.chunks[ordinal]
        for start, end in group["ranges"][ordinal]:
            separated = False
            if previous is not None:
                p, a, b = previous
                contiguous = (p == ordinal and b == start or next_chunks.get(p) == ordinal
                              and b == len(index.chunks[p].raw) and start == body_offset(chunk))
                separated = not contiguous
                if separated:
                    body += GAP
                elif body and not body[-1].isspace() and not chunk.raw[start].isspace():
                    same_input = index.chunks[p].source_id == chunk.source_id and index.chunks[p].start + b == chunk.start + start
                    if not same_input:
                        body += " "
            body += chunk.raw[start:end]
            spans.append((chunk.id, start, end))
            gaps.append(separated)
            previous = (ordinal, start, end)
    content = header + body
    meta = {"kind": "frame", "anchor_id": anchor.id, "spans": spans, "gap_before": gaps}
    if group.get("omit_role"):
        meta.update({"role_omitted": True, "source_role": anchor.role})
    return content, meta


def pack_frames(index, records, source_views, query, *, max_chars=48000, fill=True,
                cost_power=.5, distance_penalty=False, omit_role=False, debug=False, structured=False):
    """Protect the old prefix before selecting any additional source text.

    Session IDs are lookup keys only. Ranking is inherited from the first
    anchor; relevance refill sees source text, query and cached parent ranks.
    """
    records = tuple(records)
    if not max_chars or not getattr(index, "explicit_dialogue_speakers", ()) or not hasattr(index, "turn_source_spans"):
        return records, dict(source_views), {"active": False}
    from ..adapters.text.attributed_memory import queried_speakers
    targets = queried_speakers(query, index.explicit_dialogue_speakers)
    if not targets:
        return records, dict(source_views), {"active": False, "reason": "no_queried_participant"}
    count, old_size = prefix_size(records, max_chars)
    if count < 2:
        return records, dict(source_views), {"active": False}
    positions = {c.id: i for i, c in enumerate(index.chunks)}
    by_source_group = defaultdict(list)
    for i, chunk in enumerate(index.chunks):
        by_source_group[source_group(chunk)].append(i)
    sessions = defaultdict(list)
    for i, chunk in enumerate(index.chunks):
        sessions[chunk.session_id].append(i)
    next_chunks = {a: b for values in sessions.values() for a, b in zip(values, values[1:])}
    local_positions = {i: order for values in by_source_group.values() for order, i in enumerate(values)}
    groups, parent_ranks = OrderedDict(), {}
    for rank, record in enumerate(records, 1):
        meta = source_views.get(record.id)
        anchor = positions[meta["anchor_id"] if meta else record.id]
        parent_ranks.setdefault(anchor, rank)
        if rank > count:
            continue
        chunk = index.chunks[anchor]
        key = source_group(chunk)
        if key not in groups:
            groups[key] = {"anchor": anchor, "record": record, "rank": rank, "ranges": {}, "omit_role": omit_role}
        spans = [(positions[mid], a, b) for mid, a, b in meta["spans"]] if meta else [(anchor, 0, len(chunk.raw))]
        assert all(source_group(index.chunks[i]) == key for i, _, _ in spans)
        add_spans(groups[key]["ranges"], spans, index)
    if len(groups) >= count:
        return records, dict(source_views), {"active": False, "reason": "no_repeated_session"}
    group_list = list(groups.values())
    rendered = [render_frame(index, group, next_chunks)[0] for group in group_list]
    used = sum(len(text) + len(f"[Document {rank}]\n") + (2 if rank > 1 else 0)
               for rank, text in enumerate(rendered, 1))
    compact_size = used
    if used > old_size:
        return records, dict(source_views), {"active": False, "reason": "packing_would_expand_prefix"}
    added, proposal_debug = [], []
    if fill:
        from ..adapters.text.content_memory import STOPWORDS, stem
        participants = {t for name in index.explicit_dialogue_speakers for t in tokenize(name)}

        def normalized(text):
            return [stem(t) for t in tokenize(text) if t not in participants and t not in STOPWORDS]

        if not hasattr(index, "_frame_turn_bm25"):
            texts = []
            for key in index.turn_order:
                text = " ".join(index.chunks[i].raw[a:b] for i, a, b in index.turn_source_spans[key])
                texts.append(Memory(key, index.scope_id, (), key.split(":")[0], " ".join(normalized(text))))
            index._frame_turn_bm25 = BM25Retriever(texts)
        lexical = {r.memory.id: r.rank for r in index._frame_turn_bm25.search(" ".join(normalized(query)), len(index.turn_order)) if r.score > 0}
        caption_ranks = {}
        if structured:
            if not hasattr(index, "_frame_caption_bm25"):
                texts = []
                for key in index.turn_order:
                    text = " ".join(index.chunks[i].raw[a:b] for i, a, b in index.turn_source_spans[key])
                    caption = re.search(r"\bImage caption:\s*(.*)", text, re.I | re.S)
                    if caption:
                        texts.append(Memory(key, index.scope_id, (), key.split(":")[0], " ".join(normalized(caption.group(1)))))
                index._frame_caption_bm25 = BM25Retriever(texts)
            caption_ranks = {r.memory.id: r.rank for r in index._frame_caption_bm25.search(" ".join(normalized(query)), max(1, len(index.turn_order))) if r.score > 0}
        proposals = []
        named = set(index.explicit_dialogue_speakers)
        for order, key in enumerate(index.turn_order):
            if index.turn_speakers[key] not in named:
                continue
            spans = index.turn_source_spans[key]
            source_key = source_group(index.chunks[spans[0][0]])
            if source_key not in groups or any(source_group(index.chunks[i]) != source_key for i, _, _ in spans):
                continue
            group = groups[source_key]
            if all(covered_range(group["ranges"].get(i, ()), a, b) for i, a, b in spans):
                continue
            candidate_ranges = {i: list(values) for i, values in group["ranges"].items()}
            add_spans(candidate_ranges, spans, index)
            before = sum(b - a for values in group["ranges"].values() for a, b in values)
            after = sum(b - a for values in candidate_ranges.values() for a, b in values)
            extra = after - before
            if extra > 2048 or sum(len(index.chunks[i].raw[a:b].split()) for i, a, b in spans) > 128:
                continue
            parent = min((parent_ranks.get(i, math.inf) for i, _, _ in spans), default=math.inf)
            if not math.isfinite(parent):
                continue
            partial = any(i in group["ranges"] and any(max(a, c) < min(b, d) for c, d in group["ranges"][i]) for i, a, b in spans)
            relevance = .5 / (20 + parent) + .5 / (20 + lexical.get(key, math.inf)) + .15 / (20 + group["rank"])
            distance = min(abs(local_positions[i] - local_positions[j]) for i, _, _ in spans for j in group["ranges"])
            density = relevance * (1.25 if partial else 1.) / max(64, extra) ** cost_power
            if distance_penalty:
                density /= 1 + distance
            priority = 2
            if structured and index.turn_speakers[key] in targets:
                caption_visible = any(
                    any(c <= a + m.start() and a + m.end() <= d for c, d in group["ranges"].get(i, ()))
                    and any((j, end) > (i, a + m.end()) and not covered_range(group["ranges"].get(j, ()), begin, end)
                            for j, begin, end in spans)
                    for i, a, b in spans for m in re.finditer(r"\bImage caption:\s*", index.chunks[i].raw[a:b], re.I))
                if partial and caption_visible:
                    priority = 0
                elif key in caption_ranks:
                    priority = 1
                    density += .25 / (20 + caption_ranks[key])
            if debug:
                proposal_debug.append({"turn_key": key, "score": density, "partial": partial, "extra": extra,
                                       "parent_rank": parent, "lexical_rank": lexical.get(key), "distance": distance})
            proposals.append((priority, -density, order, key, source_key, spans))
        if structured and len(targets) == 1:
            turn_positions = {key: i for i, key in enumerate(index.turn_order)}
            for source_key, group in groups.items():
                present = [key for key in index.turn_order if any(
                    any(max(a, c) < min(b, d) for c, d in group["ranges"].get(i, ()))
                    for i, a, b in index.turn_source_spans[key])]
                if not present:
                    continue
                last = present[-1]
                position = turn_positions[last]
                if position < 1 or index.turn_speakers[last] == targets[0]:
                    continue
                if all(covered_range(group["ranges"].get(i, ()), a, b) for i, a, b in index.turn_source_spans[last]):
                    continue
                previous = index.turn_order[position - 1]
                question_text = " ".join(index.chunks[i].raw[a:b] for i, a, b in index.turn_source_spans[previous])
                if previous.split(":")[0] != last.split(":")[0] or index.turn_speakers[previous] != targets[0] or "?" not in question_text:
                    continue
                spans = list(index.turn_source_spans[last])
                for next_key in index.turn_order[position + 1:position + 4]:
                    if next_key.split(":")[0] != last.split(":")[0]:
                        break
                    spans.extend(index.turn_source_spans[next_key])
                if any(source_group(index.chunks[i]) != source_key for i, _, _ in spans):
                    continue
                if sum(len(index.chunks[i].raw[a:b].split()) for i, a, b in spans) > 128:
                    continue
                if sum(b - a for _, a, b in spans) > 2048:
                    continue
                proposals.append((-1, -.5 / (20 + group["rank"]), position, "followup:" + last, source_key, spans))
        proposals.sort(key=lambda item: (item[0], item[1], item[2]))
        for _, _, _, key, source_key, spans in proposals:
            group = groups[source_key]
            previous = group["ranges"]
            candidate_ranges = {i: list(values) for i, values in previous.items()}
            add_spans(candidate_ranges, spans, index)
            old_text = render_frame(index, group, next_chunks)[0]
            group["ranges"] = candidate_ranges
            new_text = render_frame(index, group, next_chunks)[0]
            cost = len(new_text) - len(old_text)
            if cost <= 0 or used + cost > max_chars:
                group["ranges"] = previous
                continue
            used += cost
            added.append(key)
    output, provenance = [], {}
    for group in group_list:
        text, meta = render_frame(index, group, next_chunks)
        identity = json.dumps(meta, ensure_ascii=False, separators=(",", ":"))
        mid = "frame_" + hashlib.sha256(identity.encode()).hexdigest()
        anchor = group["record"]
        output.append(CachedRecord(mid, text, anchor.score, anchor.created_at))
        provenance[mid] = meta
    output.extend(records[count:])
    for record in records[count:]:
        if record.id in source_views:
            provenance[record.id] = source_views[record.id]
    assert len(output) <= len(records) and sum(record_cost(r, i) for i, r in enumerate(output[:len(groups)], 1)) <= max_chars
    assert all(a.score >= b.score for a, b in zip(output, output[1:]))
    return tuple(output), provenance, {"active": True, "baseline_prefix_records": count,
        "baseline_prefix_chars": old_size, "compact_prefix_records": len(groups), "compact_prefix_chars": compact_size,
        "final_frame_chars": used, "added_turns": added, "ranking_granularity_changed": True,
        "cost_power": cost_power, "distance_penalty": distance_penalty,
        **({"proposals": sorted(proposal_debug, key=lambda r: -r["score"])} if debug else {}),
        "baseline_source_prefix_preserved": True}
