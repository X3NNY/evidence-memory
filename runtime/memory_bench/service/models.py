"""Cycle-2 Textual Add/Search envelopes (2026-10-21 API guide)."""

from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator


def nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("must contain non-whitespace text")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("must contain valid UTF-8 text") from None
    return value  # Identifiers and raw text must not be silently stripped.


Text = Annotated[StrictStr, Field(min_length=1), AfterValidator(nonblank)]
Identity = Annotated[Text, Field(max_length=4096)]


def timestamp_iso(value: int) -> str:
    dt = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=value)
    return dt.isoformat(timespec="milliseconds" if value % 1000 else "seconds").replace("+00:00", "Z")


class Envelope(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Message(Envelope):
    role: Literal["user", "assistant"]
    content: Text
    timestamp: StrictInt | None = None

    @field_validator("timestamp")
    @classmethod
    def valid_timestamp(cls, value):
        if value is not None:
            try:
                timestamp_iso(value)
            except (OverflowError, ValueError):
                raise ValueError("timestamp is outside the supported Unix-millisecond range") from None
        return value


class AddRequest(Envelope):
    request_id: Identity
    messages: Annotated[list[Message], Field(min_length=1)]
    user_id: Identity
    session_id: Identity


class AddResponse(Envelope):
    success: Literal[True] = True
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(Envelope):
    query: Text
    options: Annotated[list[Text], Field(min_length=1)] | None = None
    user_id: Identity
    top_k: Annotated[StrictInt, Field(ge=1, le=100)]


class SearchItem(Envelope):
    id: str
    content: str
    score: float
    created_at: str


class SearchResponse(Envelope):
    data: list[SearchItem]
