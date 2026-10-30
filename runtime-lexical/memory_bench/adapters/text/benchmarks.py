"""Allowlisted conversion of benchmark conversations into label-free text input."""

from pathlib import Path
import re

from ...datasets import DATA_FILES, iter_json_array
from .adapter import TextSource
from .structure import ConversationIndex


def load_text_sources(data_dir: Path, scope_ids: set[str]) -> dict[str, TextSource]:
    """Read an explicit text/metadata allowlist. No QA or relevance annotations."""
    sources = {}
    for dataset, relative in DATA_FILES.items():
        for position, record in enumerate(iter_json_array(data_dir / relative)):
            original = record.get("sample_id", position) if dataset == "locomo-refined" else record["question_id"]
            scope_id = f"{dataset}/{original}"
            if scope_id not in scope_ids:
                continue
            sessions = []
            if dataset == "locomo-refined":
                conversation = record["conversation"]
                keys = sorted((k for k in conversation if re.fullmatch(r"session_\d+", k)),
                              key=lambda k: int(k.rsplit("_", 1)[1]))
                for key in keys:
                    turns = []
                    for turn in conversation[key]:
                        speaker = str(turn.get("speaker", "speaker"))
                        text = f"{speaker}: {turn.get('text', '')}"
                        if turn.get("blip_caption"):
                            text += f" Image caption: {turn['blip_caption']}"
                        turns.append((str(turn["dia_id"]), speaker, text))
                    sessions.append((str(conversation.get(key + "_date_time", "")), turns))
            else:
                for source_id, date, turns in zip(record["haystack_session_ids"], record["haystack_dates"], record["haystack_sessions"]):
                    values = [(str(source_id), str(turn.get("role", "speaker")),
                               f"{turn.get('role', 'speaker')}: {turn.get('content', '')}") for turn in turns]
                    sessions.append((str(date), values))
            sources[scope_id] = TextSource.from_sessions(sessions)
    if set(sources) != set(scope_ids):
        raise ValueError("Raw text indexes do not match selected scopes")
    return sources



def load_conversation_indexes(data_dir, scopes, *, chunk_words=256, passage_words=128, cache_aligned=False):
    sources = load_text_sources(data_dir, set(scopes))
    return {sid: ConversationIndex(sid, scopes[sid].memories, source.sessions, chunk_words=chunk_words,
                                   passage_words=passage_words, cache_aligned=cache_aligned)
            for sid, source in sources.items()}
