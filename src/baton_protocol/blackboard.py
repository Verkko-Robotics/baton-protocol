"""The Blackboard — the only channel through which specialists share data.

Specialists never hold references to one another. One posts a fact, another
reads it by ``capability.key`` address. That indirection is what makes a task
auditable: every piece of data that moved between capabilities is a durable,
attributed, ordered entry rather than an untraceable function call.

Three properties are load-bearing:

**Append-only.** Entries are immutable and nothing is ever deleted. A read
returns the highest-``seq`` matching entry, so corrections are expressed by
posting again, and the superseded value stays visible for audit.

**Fail-closed visibility.** ``visible_to=[]`` means *nobody*, and is honoured
literally. An earlier draft used ``visible_to or ["*"]``, which quietly
converted "show this to no one" into "show this to everyone" — the worst
possible direction for a defaulting bug in an access check.

**Tenant-scoped.** One blackboard belongs to one task belonging to one tenant.
A cross-tenant post raises rather than being filtered out, because it can only
mean the host wired something up wrong.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Iterable, Sequence

from ._codec import utcnow
from .errors import TenantMismatch
from .models import VISIBLE_TO_ALL, BlackboardEntry


class Blackboard:
    """Append-only shared fact store scoped to a single task."""

    def __init__(
        self,
        task_id: str,
        tenant_id: str,
        *,
        entries: Iterable[BlackboardEntry] = (),
        on_append: Callable[[BlackboardEntry], None] | None = None,
    ) -> None:
        """
        Args:
            task_id: Task this blackboard belongs to.
            tenant_id: Tenant that owns the task. Enforced on every post.
            entries: Existing entries, when rehydrating a suspended task.
            on_append: Durability hook, called after each successful post.
                Typically ``ContextStore.append_blackboard``.
        """
        self.task_id = task_id
        self.tenant_id = tenant_id
        self._entries: list[BlackboardEntry] = list(entries)
        self._on_append = on_append
        # Guards the compound read-modify-write in `post`. `list.append` is
        # atomic under the GIL but `_next_seq` followed by append is not, and
        # duplicate sequence numbers would break "latest wins" ordering.
        self._lock = threading.RLock()
        self._seq = max((e.seq for e in self._entries), default=0)

    # ── Write ───────────────────────────────────────────────────────────────

    def post(
        self,
        *,
        specialist_id: str,
        capability: str,
        key: str,
        value: Any,
        tenant_id: str | None = None,
        visible_to: Sequence[str] | None = None,
        schema_ref: str | None = None,
        derived_from: Sequence[int] = (),
    ) -> BlackboardEntry:
        """Append a fact.

        Args:
            visible_to: Capabilities permitted to read this entry. ``None``
                means visible to all capabilities on the task. An empty
                sequence means visible to none, and is honoured as written.
            tenant_id: The posting specialist's tenant. Checked against the
                blackboard's own when supplied.
            derived_from: ``seq`` of the entries this fact was derived from.

        Raises:
            TenantMismatch: If ``tenant_id`` is not this blackboard's tenant.
        """
        if tenant_id is not None and tenant_id != self.tenant_id:
            raise TenantMismatch(self.tenant_id, tenant_id)
        if not key:
            raise ValueError("blackboard key must be a non-empty string")

        with self._lock:
            self._seq += 1
            entry = BlackboardEntry(
                capability=capability,
                key=key,
                value=value,
                posted_by=specialist_id,
                seq=self._seq,
                created_at=utcnow(),
                visible_to=(
                    (VISIBLE_TO_ALL,) if visible_to is None else tuple(visible_to)
                ),
                schema_ref=schema_ref,
                derived_from=tuple(derived_from),
            )
            self._entries.append(entry)

        if self._on_append is not None:
            self._on_append(entry)
        return entry

    # ── Read ────────────────────────────────────────────────────────────────

    def read(
        self,
        *,
        capability: str,
        key: str,
        requesting_capability: str,
        default: Any = None,
    ) -> Any:
        """Latest value posted by ``capability`` under ``key``.

        Returns ``default`` when nothing matches *or* when the requester may not
        see what matched — an unreadable entry is indistinguishable from an
        absent one, so visibility rules cannot be probed.
        """
        entry = self.read_entry(
            capability=capability,
            key=key,
            requesting_capability=requesting_capability,
        )
        return default if entry is None else entry.value

    def read_entry(
        self,
        *,
        capability: str,
        key: str,
        requesting_capability: str,
    ) -> BlackboardEntry | None:
        """Latest readable entry matching ``capability`` and ``key``."""
        with self._lock:
            candidates = [
                e
                for e in self._entries
                if e.capability == capability
                and e.key == key
                and self._can_read(e, requesting_capability)
            ]
        return max(candidates, key=lambda e: e.seq) if candidates else None

    def read_address(
        self, address: str, requesting_capability: str
    ) -> BlackboardEntry | None:
        """Latest readable entry at a ``capability.key`` address.

        Matches on the joined address rather than splitting it, because either
        half may itself contain a dot.
        """
        with self._lock:
            candidates = [
                e
                for e in self._entries
                if e.qualified_key == address
                and self._can_read(e, requesting_capability)
            ]
        return max(candidates, key=lambda e: e.seq) if candidates else None

    def read_all(self, requesting_capability: str) -> dict[str, Any]:
        """Every fact visible to ``requesting_capability``, latest per address.

        Keyed by ``"capability.key"``. This is what gets rendered into a
        specialist's prompt.
        """
        with self._lock:
            latest: dict[str, BlackboardEntry] = {}
            for entry in self._entries:
                if not self._can_read(entry, requesting_capability):
                    continue
                current = latest.get(entry.qualified_key)
                if current is None or entry.seq > current.seq:
                    latest[entry.qualified_key] = entry
        return {addr: e.value for addr, e in latest.items()}

    def history(
        self,
        requesting_capability: str,
        *,
        capability: str | None = None,
        key: str | None = None,
    ) -> list[BlackboardEntry]:
        """Full readable history in post order, oldest first.

        Unlike :meth:`read_all` this keeps superseded values, which is what an
        auditor needs to see how a conclusion was revised.
        """
        with self._lock:
            return [
                e
                for e in self._entries
                if self._can_read(e, requesting_capability)
                and (capability is None or e.capability == capability)
                and (key is None or e.key == key)
            ]

    def snapshot(self) -> list[BlackboardEntry]:
        """Every entry, ignoring visibility.

        For persistence and audit by the orchestrator. Never expose the result
        to a specialist: it deliberately bypasses the access rules.
        """
        with self._lock:
            return list(self._entries)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    # ── Internal ────────────────────────────────────────────────────────────

    @staticmethod
    def _can_read(entry: BlackboardEntry, requesting_capability: str) -> bool:
        """Whether ``requesting_capability`` may read ``entry``.

        A capability can always read what it posted itself; otherwise it needs
        an explicit grant or the wildcard.
        """
        if entry.capability == requesting_capability:
            return True
        if VISIBLE_TO_ALL in entry.visible_to:
            return True
        return requesting_capability in entry.visible_to
