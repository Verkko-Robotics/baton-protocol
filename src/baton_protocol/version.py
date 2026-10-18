"""Version identifiers for BATON.

Two versions are tracked separately and must not be conflated:

``__version__``
    The version of *this implementation* (the Python reference library).
    Moves for bug fixes and internal refactors.

``PROTOCOL_VERSION``
    The version of *the wire format*. Every serialized envelope carries it.
    Implementations in other languages interoperate on this number alone.

Wire-format compatibility rules:

* MAJOR — a field was removed or its meaning changed. Not backward compatible.
* MINOR — a field was added. Older readers must ignore it and preserve it on
  round-trip (see :mod:`baton_protocol._codec`).
* PATCH — clarification only, no structural change.
"""

from __future__ import annotations

__version__ = "0.1.0"

PROTOCOL_VERSION = "1.0.0"

_MIN_READABLE_MAJOR = 1


def parse_version(version: str) -> tuple[int, int, int]:
    """Parse a ``MAJOR.MINOR.PATCH`` string into a comparable tuple."""
    parts = version.split(".")
    if len(parts) != 3:
        raise ValueError(f"malformed protocol version: {version!r}")
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError as exc:
        raise ValueError(f"malformed protocol version: {version!r}") from exc


def is_compatible(version: str) -> bool:
    """Whether this library can read an envelope stamped with ``version``.

    A newer MINOR is readable: unknown fields are preserved rather than
    dropped, so a v1.0 reader can safely round-trip a v1.3 envelope.
    """
    major, _minor, _patch = parse_version(version)
    ours = parse_version(PROTOCOL_VERSION)
    return _MIN_READABLE_MAJOR <= major <= ours[0]
