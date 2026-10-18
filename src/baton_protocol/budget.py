"""Task-wide resource budgets.

A budget is shared by every specialist in a task tree, including ones spawned
by delegation. This is the point: a per-specialist cap bounds each individual
run but does nothing about a fan-out of thirty specialists that each stay under
their own limit. Cost is a property of the task, so the ceiling lives on the
task.

The budget is passed by reference down every delegation, so a child charging
tokens is immediately visible to its parent and to its siblings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ._codec import drop_none, merge_unknown, split_unknown
from .errors import BudgetExceeded, DelegationDepthExceeded

_KNOWN = frozenset(
    {
        "max_total_tokens",
        "max_total_spawns",
        "max_spawn_depth",
        "max_tool_calls",
        "tokens_used",
        "spawns_used",
        "tool_calls_used",
    }
)


@dataclass
class TaskBudget:
    """Mutable, shared spend ceiling for one task tree.

    All four limits are hard: exceeding one raises :class:`BudgetExceeded` and
    fails the task rather than degrading quietly, because a runaway agent tree
    on a metered product is a billing incident, not a quality problem.
    """

    max_total_tokens: int = 400_000
    max_total_spawns: int = 12
    max_spawn_depth: int = 3
    max_tool_calls: int = 60

    tokens_used: int = 0
    spawns_used: int = 0
    tool_calls_used: int = 0

    unknown: dict[str, Any] = field(default_factory=dict, repr=False)

    # ── Charging ────────────────────────────────────────────────────────────

    def charge_tokens(self, count: int) -> None:
        """Record token spend. Raises once the ceiling is crossed."""
        if count < 0:
            raise ValueError("token count must be non-negative")
        self.tokens_used += count
        if self.tokens_used > self.max_total_tokens:
            raise BudgetExceeded("token", self.tokens_used, self.max_total_tokens)

    def charge_tool_call(self) -> None:
        """Record one tool invocation against the task ceiling."""
        if self.tool_calls_used + 1 > self.max_tool_calls:
            raise BudgetExceeded(
                "tool call", self.tool_calls_used + 1, self.max_tool_calls
            )
        self.tool_calls_used += 1

    def charge_spawn(self, depth: int) -> None:
        """Record one specialist spawn at ``depth``.

        Every check runs before the counter moves. A refused delegation must not
        consume budget — it did no work, and spawn refusal is recoverable, so a
        caller that retries differently would otherwise be charged twice.
        """
        if depth > self.max_spawn_depth:
            raise DelegationDepthExceeded(depth, self.max_spawn_depth)
        if self.spawns_used + 1 > self.max_total_spawns:
            raise BudgetExceeded("spawn", self.spawns_used + 1, self.max_total_spawns)
        self.spawns_used += 1

    # ── Inspection ──────────────────────────────────────────────────────────

    @property
    def tokens_remaining(self) -> int:
        return max(0, self.max_total_tokens - self.tokens_used)

    @property
    def spawns_remaining(self) -> int:
        return max(0, self.max_total_spawns - self.spawns_used)

    def would_exceed_tokens(self, count: int) -> bool:
        """Whether charging ``count`` tokens would breach the ceiling.

        Lets a caller skip an expensive model call it cannot afford instead of
        making it and then failing.
        """
        return self.tokens_used + count > self.max_total_tokens

    # ── Serialization ───────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return merge_unknown(
            drop_none(
                {
                    "max_total_tokens": self.max_total_tokens,
                    "max_total_spawns": self.max_total_spawns,
                    "max_spawn_depth": self.max_spawn_depth,
                    "max_tool_calls": self.max_tool_calls,
                    "tokens_used": self.tokens_used,
                    "spawns_used": self.spawns_used,
                    "tool_calls_used": self.tool_calls_used,
                }
            ),
            self.unknown,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskBudget":
        known, unknown = split_unknown(data, _KNOWN)
        return cls(
            max_total_tokens=int(known.get("max_total_tokens", 400_000)),
            max_total_spawns=int(known.get("max_total_spawns", 12)),
            max_spawn_depth=int(known.get("max_spawn_depth", 3)),
            max_tool_calls=int(known.get("max_tool_calls", 60)),
            tokens_used=int(known.get("tokens_used", 0)),
            spawns_used=int(known.get("spawns_used", 0)),
            tool_calls_used=int(known.get("tool_calls_used", 0)),
            unknown=unknown,
        )
