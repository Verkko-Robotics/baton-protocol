"""Serialization helpers shared by every protocol model.

The one non-obvious requirement here is **forward compatibility**. BATON is meant
to be implemented by parties who upgrade on their own schedule, so a v1.0
reader will regularly encounter envelopes written by a v1.4 producer. If the
reader drops the fields it doesn't recognize, a round-trip through an older
service silently destroys data — the failure mode that kills interoperating
protocols.

So every model keeps an ``unknown`` dict. Keys the reader doesn't know go in on
the way in and come back out on the way out, untouched.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

_ISO_Z_SUFFIX = "Z"


def utcnow() -> datetime:
    """Timezone-aware current UTC time.

    ``datetime.utcnow()`` is deprecated and returns a naive datetime, which
    compares and serializes incorrectly across boundaries.
    """
    return datetime.now(timezone.utc)


def to_iso(value: datetime) -> str:
    """Serialize a datetime to ISO 8601 with an explicit UTC offset."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def from_iso(value: str | datetime | None) -> datetime:
    """Parse an ISO 8601 timestamp, tolerating a trailing ``Z``."""
    if value is None:
        return utcnow()
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = value.strip()
    if text.endswith(_ISO_Z_SUFFIX):
        text = text[: -len(_ISO_Z_SUFFIX)] + "+00:00"
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def split_unknown(
    data: Mapping[str, Any],
    known: Iterable[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Partition an incoming mapping into recognized and unrecognized keys.

    Returns ``(recognized, unknown)``. Callers hand ``unknown`` straight to the
    model's ``unknown`` field so it survives the round-trip.
    """
    known_set = set(known)
    recognized: dict[str, Any] = {}
    unknown: dict[str, Any] = {}
    for key, value in data.items():
        if key in known_set:
            recognized[key] = value
        else:
            unknown[key] = value
    return recognized, unknown


def merge_unknown(
    payload: dict[str, Any],
    unknown: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-attach preserved unknown keys to an outgoing payload.

    Known fields win: a producer must never be able to shadow a field this
    implementation actually understands by smuggling it through ``unknown``.
    """
    for key, value in unknown.items():
        payload.setdefault(key, value)
    return payload


def drop_none(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove ``None`` values so absent and null are indistinguishable on the wire."""
    return {k: v for k, v in payload.items() if v is not None}
