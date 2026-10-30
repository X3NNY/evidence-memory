"""Compose source-only response views independently of ranking and storage."""

from dataclasses import dataclass
import json

from .frames import pack_frames
from .responses import SourceRecord
from .windows import enrich_windows, prefix_size


@dataclass(frozen=True)
class EvidencePolicy:
    context_chars: int = 48000
    session_frames: bool = True

    def compose(self, domain, records, query):
        records, views, _ = enrich_windows(domain, records, max_chars=self.context_chars)
        if self.session_frames and self.context_chars:
            count, _ = prefix_size(records, self.context_chars)
            if count < len(records):
                records, views, _ = pack_frames(domain, records, views, query,
                    max_chars=self.context_chars, cost_power=0., distance_penalty=True,
                    omit_role=True, structured=True)
        # Both windows and frames expire with the bounded response cache.
        return tuple(SourceRecord(r.id, r.content, r.score, r.created_at,
                     json.dumps(views[r.id], ensure_ascii=False, separators=(",", ":")))
                     if r.id in views else r for r in records)
