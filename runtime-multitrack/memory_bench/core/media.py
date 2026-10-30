"""Local image evidence serialization; no image generation or model calls."""

from __future__ import annotations

import base64
from functools import lru_cache
import hashlib
from pathlib import Path

from .models import Evidence, MediaReference


class OriginalMediaStore:
    def __init__(self, root: Path):
        self.root = root.resolve()

    @lru_cache(maxsize=32)
    def read(self, reference: MediaReference) -> bytes:
        path = (self.root / reference.path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Media path escapes the original data directory")
        data = path.read_bytes()
        if len(data) != reference.byte_size or hashlib.sha256(data).hexdigest() != reference.sha256:
            raise ValueError("Original media content changed")
        return data

    def content_parts(self, context: str, media: tuple[MediaReference, ...]):
        parts = [{"type": "text", "text": context}] if context else []
        for reference in media:
            data = self.read(reference)
            parts.append({"type": "image_url", "image_url": {
                "url": f"data:{reference.media_type};base64," + base64.b64encode(data).decode("ascii")}})
        return parts

    def evidence_parts(self, evidence: Evidence):
        if not evidence.media:
            return self.content_parts(evidence.context, ())
        lookup = {item.id: item for item in evidence.media}
        layout = evidence.audit["media_by_memory"]
        parts = []
        for number, memory in enumerate(evidence.memories, 1):
            text = ("\n\n" if number > 1 else "") + f"[Document {number}]\n" + memory.text
            parts.extend(self.content_parts(text, tuple(lookup[mid] for mid in layout.get(memory.id, ()))))
        if "".join(part["text"] for part in parts if part["type"] == "text") != evidence.context:
            raise ValueError("Interleaved evidence text differs from the rendered context")
        return parts

    def verify_roundtrip(self, parts, media):
        images = [part for part in parts if part["type"] == "image_url"]
        if len(images) != len(media):
            raise ValueError("Image content and source references do not align")
        for part, reference in zip(images, media):
            prefix, encoded = part["image_url"]["url"].split(",", 1)
            if prefix != f"data:{reference.media_type};base64" or base64.b64decode(encoded, validate=True) != self.read(reference):
                raise ValueError("Image evidence did not round-trip to its original bytes")
        return True
