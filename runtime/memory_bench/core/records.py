"""Raw-record evidence packing shared by coding and image-caption adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from .context import render_context
from .models import CandidateSet, ContextBudget, Evidence, MediaReference, Memory


def raw_chunks(text: str, max_words: int = 256):
    """Preserve original whitespace, indentation and patch lines within chunks."""
    if not isinstance(max_words, int) or isinstance(max_words, bool) or max_words < 1:
        raise ValueError("Chunk size must be a positive integer")
    words = list(re.finditer(r"\S+", text))
    for start in range(0, len(words), max_words):
        end = min(len(words), start + max_words)
        begin_char = 0 if start == 0 else words[start-1].end()
        end_char = len(text) if end == len(words) else words[end-1].end()
        yield text[begin_char:end_char]


@dataclass(frozen=True)
class RecordSource:
    media_by_memory: dict[str, tuple[MediaReference, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class RecordIndex:
    scope_id: str
    source: RecordSource


class RecordAdapter:
    def index(self, scope_id, memories, source):
        if not isinstance(source, RecordSource):
            raise TypeError("RecordAdapter accepts label-free RecordSource only")
        if not set(source.media_by_memory) <= {m.id for m in memories}:
            raise ValueError("Media references point outside the indexed memories")
        if any(media.scope_id != scope_id for values in source.media_by_memory.values() for media in values):
            raise ValueError("Source media cannot cross scopes")
        return RecordIndex(scope_id, source)

    def search(self, index: RecordIndex, query: str, candidates: CandidateSet, budget: ContextBudget):
        ranked = candidates.fused
        rows = {(item.memory.scope_id, item.memory.id): {"text": item.memory.text} for item in ranked}
        row = {"scope_id": index.scope_id, "ranked": [{"memory_id": item.memory.id} for item in ranked]}
        text, audit = render_context(row, rows, {"max_chars": budget.max_chars,
                                                "max_memories": budget.max_documents, "packing": "complete"})
        used = set(audit["used_memory_ids"])
        memories = tuple(item.memory for item in ranked if item.memory.id in used)
        media, seen, omitted, byte_count, media_by_memory = [], set(), [], 0, {}
        for memory in memories:
            for item in index.source.media_by_memory.get(memory.id, ()):
                if item.id in seen:
                    continue
                seen.add(item.id)
                if item.byte_size > 10 * 1024 * 1024 or byte_count + item.byte_size > budget.max_media_bytes:
                    omitted.append(item.id)
                    continue
                media.append(item)
                media_by_memory.setdefault(memory.id, []).append(item.id)
                byte_count += item.byte_size
        audit.update({"media_ids": [item.id for item in media], "media_bytes": byte_count,
                      "media_budget_bytes": budget.max_media_bytes, "omitted_media_ids": omitted,
                      "media_by_memory": media_by_memory})
        return Evidence(text, audit, memories, media=tuple(media))
