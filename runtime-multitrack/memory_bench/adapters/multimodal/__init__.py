"""Caption-based retrieval with references to original, unmodified images."""

from ...core.records import RecordAdapter


class MultimodalAdapter(RecordAdapter):
    """Uses provided captions; images remain original evidence, not embeddings."""
