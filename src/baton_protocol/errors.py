"""Exception hierarchy for BATON.

The split that matters here is between errors the *model* should see and
recover from, and errors that terminate a run.

:class:`RecoverableError` and its subclasses are fed back into the message
history as a tool result so the model can correct itself. Everything else
propagates and fails the specialist.
"""

from __future__ import annotations

from typing import Any


class BatonError(Exception):
    """Base class for every protocol error."""


# ── Recoverable: surfaced to the model, not fatal ───────────────────────────


class RecoverableError(BatonError):
    """An error the model is expected to observe and work around.

    The loop converts these into ``role: "tool"`` messages instead of
    aborting, because a hallucinated tool name or a malformed argument is a
    normal part of an agentic run, not a system failure.
    """


class ToolNotAuthorized(RecoverableError):
    """The specialist asked for a tool outside its authorized set."""

    def __init__(self, tool_name: str, authorized: list[str]) -> None:
        self.tool_name = tool_name
        self.authorized = authorized
        super().__init__(
            f"tool {tool_name!r} is not available to this specialist; "
            f"authorized tools: {sorted(authorized)}"
        )


class ToolExecutionFailed(RecoverableError):
    """The tool ran but reported failure."""

    def __init__(self, tool_name: str, error: str) -> None:
        self.tool_name = tool_name
        self.error = error
        super().__init__(f"tool {tool_name!r} failed: {error}")


class SchemaViolation(RecoverableError):
    """Output did not match the subtask's declared output schema."""

    def __init__(self, message: str, violations: list[str] | None = None) -> None:
        self.violations = violations or []
        super().__init__(message)


# ── Fatal: terminate the specialist or the task ─────────────────────────────


class BudgetExceeded(BatonError):
    """A task-wide token, tool-call or spawn budget was exhausted.

    Deliberately task-wide: a per-specialist cap does not stop a fan-out of
    specialists from collectively overspending.

    ``kind`` decides how the loop reacts. Token and tool-call exhaustion are
    fatal, since no further work is possible. Spawn exhaustion is reported to
    the model instead — the specialist can no longer get help, but it has
    already been paid for and can still finish alone.
    """

    def __init__(self, kind: str, used: int, limit: int) -> None:
        self.kind = kind
        self.used = used
        self.limit = limit
        super().__init__(f"task {kind} budget exhausted: {used} used, limit {limit}")


class DelegationDenied(BatonError):
    """A specialist tried to delegate to a capability it may not request."""

    def __init__(self, requester: str, target: str, allowed: list[str]) -> None:
        self.requester = requester
        self.target = target
        super().__init__(
            f"{requester!r} may not delegate to {target!r}; "
            f"allowed: {sorted(allowed)}"
        )


class DelegationCycle(BatonError):
    """Delegation would re-enter a capability already on the call stack."""

    def __init__(self, ancestry: list[str], target: str) -> None:
        self.ancestry = ancestry
        self.target = target
        super().__init__(
            "delegation cycle detected: " + " -> ".join([*ancestry, target])
        )


class DelegationDepthExceeded(BatonError):
    """Delegation would exceed the task's maximum spawn depth."""

    def __init__(self, depth: int, max_depth: int) -> None:
        self.depth = depth
        self.max_depth = max_depth
        super().__init__(f"delegation depth {depth} exceeds maximum {max_depth}")


class CapabilityNotRegistered(BatonError):
    """No specialist is registered for the requested capability."""

    def __init__(self, capability: str, available: list[str]) -> None:
        self.capability = capability
        super().__init__(
            f"no specialist registered for capability {capability!r}; "
            f"available: {sorted(available)}"
        )


class CapabilityCollision(BatonError):
    """Two different specialist configs claim the same capability name.

    A capability name is an authority address: it decides which tools a
    specialist may use and who it may delegate to. Merging registries with a
    plain dict update lets the later definition silently replace the earlier
    one, so a team member named like a built-in would inherit (or strip) that
    built-in's grants. Raised instead of overwritten.
    """

    def __init__(self, capabilities: list[str]) -> None:
        self.capabilities = sorted(capabilities)
        super().__init__(
            "capability name collision with differing configs: "
            f"{self.capabilities}"
        )


class TenantMismatch(BatonError):
    """An operation crossed a tenant boundary.

    Raised rather than filtered. A cross-tenant read is a bug in the host
    application, and silently returning nothing would hide it.
    """

    def __init__(self, expected: str, actual: str) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"tenant boundary violation: expected {expected!r}, got {actual!r}"
        )


class IterationLimitExceeded(BatonError):
    """The agentic loop ran out of iterations without a final answer.

    Carries whatever partial state existed so the caller can decide between
    discarding it and returning a degraded result.
    """

    def __init__(self, capability: str, max_iterations: int, partial: Any = None) -> None:
        self.capability = capability
        self.max_iterations = max_iterations
        self.partial = partial
        super().__init__(
            f"specialist {capability!r} exceeded max_iterations ({max_iterations})"
        )


class SpecialistTimeout(BatonError):
    """A specialist exceeded its wall-clock allowance."""

    def __init__(self, capability: str, timeout_seconds: float) -> None:
        self.capability = capability
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"specialist {capability!r} timed out after {timeout_seconds}s"
        )


class IncompatibleProtocolVersion(BatonError):
    """An envelope was stamped with a wire version this library cannot read."""

    def __init__(self, found: str, supported: str) -> None:
        self.found = found
        self.supported = supported
        super().__init__(
            f"envelope protocol version {found!r} is not readable by "
            f"this implementation (supports {supported!r})"
        )


class UnresolvableDependencies(BatonError):
    """A subtask graph is cyclic or references unknown subtasks."""

    def __init__(self, message: str, subtask_ids: list[str] | None = None) -> None:
        self.subtask_ids = subtask_ids or []
        super().__init__(message)


class ApprovalRequired(BatonError):
    """Signals that a side-effecting tool call needs human authorization.

    Control flow, not a failure. The loop catches this, serializes its state
    and returns a suspended run for later resumption.
    """

    def __init__(self, tool_name: str, arguments: dict[str, Any], call_id: str) -> None:
        self.tool_name = tool_name
        self.arguments = arguments
        self.call_id = call_id
        super().__init__(f"tool {tool_name!r} requires approval before execution")


class ApprovalDenied(BatonError):
    """A human rejected a pending side-effecting tool call."""

    def __init__(self, tool_name: str, reason: str | None = None) -> None:
        self.tool_name = tool_name
        self.reason = reason
        super().__init__(
            f"approval denied for tool {tool_name!r}"
            + (f": {reason}" if reason else "")
        )
