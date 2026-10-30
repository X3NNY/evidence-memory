"""Historical code-issue and patch evidence; no task-solving model is invoked."""

from ...core.records import RecordAdapter


class CodingAdapter(RecordAdapter):
    def index(self, scope_id, memories, source):
        index = super().index(scope_id, memories, source)
        if source.media_by_memory:
            raise ValueError("The coding baseline expects text-only historical experience")
        return index
