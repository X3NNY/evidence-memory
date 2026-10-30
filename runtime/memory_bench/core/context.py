"""Deterministic raw-evidence rendering with exact character accounting."""

from __future__ import annotations

import hashlib
from typing import Any


class ContextError(ValueError):
    """Invalid retrieved evidence or packing configuration."""


def render_context(row: dict[str, Any], memories: dict[tuple[str, str], dict[str, Any]], config: dict[str, Any], *, error_type: type[ValueError] = ContextError) -> tuple[str, dict[str, Any]]:
    packing = config.get("packing", "prefix")
    if packing not in {"prefix", "complete"}:
        raise error_type("context.packing must be prefix or complete")
    ranked = row.get("ranked", [])
    pieces = []
    rendered = []
    for item in ranked:
        key = (str(row["scope_id"]), str(item["memory_id"]))
        memory = memories.get(key)
        if memory is None:
            raise error_type(f"Missing retrieved memory {item['memory_id']} in memories.jsonl")
        # Some public datasets encode answer relevance in identifiers such as
        # ``answer_*``. Identifiers stay in audit metadata, never in the prompt.
        header = f"[Document {len(rendered) + 1}]\n"
        rendered.append((str(item["memory_id"]), header + str(memory["text"])))
    original_chars = sum(len(value) for _, value in rendered) + max(0, len(rendered) - 1) * 2
    used_ids = []
    truncated_ids = []
    visible_chars = {}
    used_chars = 0
    max_memories = config.get("max_memories")
    for memory_id, value in rendered:
        if max_memories is not None and len(used_ids) >= max_memories:
            break
        separator = "\n\n" if pieces else ""
        # Complete packing skips a block that cannot fit and tries later blocks.
        # It preserves raw text and keeps numbering contiguous after a skip.
        if packing == "complete":
            memory = memories[(str(row["scope_id"]), memory_id)]
            value = f"[Document {len(used_ids) + 1}]\n" + str(memory["text"])
        header_chars = len(f"[Document {len(used_ids) + 1}]\n")
        remaining = config["max_chars"] - used_chars - len(separator)
        if remaining <= 0:
            break
        if len(value) > remaining:
            if packing == "complete":
                continue
            marker = "\n[TRUNCATED]"
            visible_chars[memory_id] = max(0, remaining - len(marker) - header_chars)
            value = value[:max(0, remaining - len(marker))] + marker[:remaining]
            truncated_ids.append(memory_id)
        else:
            visible_chars[memory_id] = len(value) - header_chars
        pieces.append(separator + value)
        used_chars += len(separator) + len(value)
        used_ids.append(memory_id)
        if truncated_ids:
            break
    context = "".join(pieces)
    return context, {
        "max_chars": config["max_chars"], "max_memories": max_memories,
        "original_evidence_chars": original_chars, "used_evidence_chars": len(context),
        "retrieved_memory_count": len(rendered), "used_memory_count": len(used_ids),
        "omitted_memory_count": len(rendered) - len(used_ids),
        "used_memory_ids": used_ids, "truncated_memory_ids": truncated_ids,
        "visible_memory_chars": visible_chars, "packing": packing,
        "truncated": bool(truncated_ids or len(used_ids) < len(rendered)),
        "evidence_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
        "budget_unit": "characters (not model tokens)",
    }

