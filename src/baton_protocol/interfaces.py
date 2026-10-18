"""Gateway interfaces — the seam between the protocol and the outside world.

Everything the protocol needs that involves I/O is declared here as an abstract
base class and injected at construction. Nothing in this package imports a
database driver, an HTTP client or a model SDK, and nothing ever should: that
constraint is what makes BATON implementable on top of any stack, and what lets
the entire framework be tested without a network.

A host application implements these against its own infrastructure. The
reference in-memory implementations in :mod:`baton_protocol.testing` show the
expected semantics and are the fixtures the conformance suite runs against.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Sequence

from ._codec import utcnow
from .models import (
    BlackboardEntry,
    ContextEnvelope,
    ContextSlice,
    LLMResponse,
    PendingApproval,
    SpecialistResult,
    ToolResult,
    ToolSpec,
)


class Clock(ABC):
    """Injectable time source, so timeouts are testable without sleeping."""

    @abstractmethod
    def now(self) -> datetime:
        """Current timezone-aware UTC time."""

    @abstractmethod
    def monotonic(self) -> float:
        """Monotonic seconds, for measuring elapsed durations."""


class SystemClock(Clock):
    """Default :class:`Clock` backed by the system time."""

    def now(self) -> datetime:
        return utcnow()

    def monotonic(self) -> float:
        import time

        return time.monotonic()


class ContextStore(ABC):
    """Durable storage for :class:`ContextEnvelope` records.

    Implementations must make ``save`` atomic with respect to ``revision``: two
    concurrent writers must not silently interleave. If the backing store
    supports conditional writes, use them and raise on a revision conflict.
    """

    @abstractmethod
    def load(self, task_id: str) -> ContextEnvelope | None:
        """Fetch an envelope, or ``None`` if no such task exists."""

    @abstractmethod
    def save(self, envelope: ContextEnvelope) -> None:
        """Persist an envelope, overwriting any prior revision."""

    @abstractmethod
    def append_blackboard(self, task_id: str, entry: BlackboardEntry) -> None:
        """Durably append one blackboard entry.

        Separate from ``save`` because the blackboard is append-only and hot:
        rewriting the whole envelope on every post would be wasteful and would
        widen the window for lost updates.
        """


class MemoryStore(ABC):
    """Long-term memory consolidation and retrieval.

    The protocol treats memory as opaque: it hands over a summary plus metadata
    and asks for relevant prior items back. Whether that is embeddings, a vector
    index, a knowledge graph or a text log is entirely the host's business.
    """

    @abstractmethod
    def consolidate(
        self,
        *,
        principal_id: str,
        capability: str,
        summary: str,
        metadata: dict[str, Any],
    ) -> str:
        """Persist one memory item. Returns an implementation-defined id."""

    @abstractmethod
    def recall(
        self,
        *,
        principal_id: str,
        capability: str,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Retrieve prior items relevant to ``query``, most relevant first."""


class ToolGateway(ABC):
    """Tool discovery, classification and execution.

    ``available_tools`` is the runtime half of tool authorization. A specialist's
    static ``config.tools`` says what it is *designed* to use; this says what the
    principal has actually connected. The effective set is the intersection —
    offering a model a tool whose credentials are missing produces a guaranteed
    failure several seconds and several thousand tokens later.
    """

    @abstractmethod
    def available_tools(self, principal_id: str) -> list[ToolSpec]:
        """Tools this principal has authorized, with their schemas."""

    @abstractmethod
    def execute(
        self,
        *,
        tool_name: str,
        principal_id: str,
        arguments: dict[str, Any],
        idempotency_key: str | None = None,
    ) -> ToolResult:
        """Invoke a tool.

        Must not raise for ordinary tool failure — return
        ``ToolResult(success=False, error=...)`` so the loop can feed the error
        back to the model. Reserve exceptions for infrastructure faults.

        When ``idempotency_key`` is supplied, a repeated call with the same key
        must not repeat the side effect.
        """


class LLMGateway(ABC):
    """Model invocation, normalized to :class:`LLMResponse`.

    Implementations are responsible for translating provider-specific payloads —
    including tool-call and usage shapes — into the protocol's neutral form.
    """

    @abstractmethod
    def complete(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[ToolSpec],
        model: str,
        temperature: float,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Run one completion and return a normalized response."""


class ApprovalGateway(ABC):
    """Human authorization for side-effecting tool calls.

    Deliberately asynchronous in spirit: ``request`` records that authorization
    is needed and returns immediately, and the run suspends. Resolution arrives
    later through the host's own approval surface. Blocking a worker on a human
    is not a supported pattern.
    """

    @abstractmethod
    def request(
        self,
        *,
        context: ContextSlice,
        approval: PendingApproval,
    ) -> str:
        """Record an approval request. Returns a host-defined approval id."""

    @abstractmethod
    def status(self, approval_id: str) -> str:
        """One of ``"pending"``, ``"approved"`` or ``"denied"``."""


class UsageMeter(ABC):
    """Records model and tool spend against the principal.

    Non-optional by design. A framework that can fan out into a tree of
    specialists and does not report what each one spent will under-bill in
    proportion to how well its delegation works.
    """

    @abstractmethod
    def record(
        self,
        *,
        principal_id: str,
        capability: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        idempotency_key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record one unit of metered spend."""


class TemplateRenderer(ABC):
    """Renders a specialist's prompt from a named template.

    An interface rather than a hard Jinja2 dependency, so the core stays
    dependency-free and hosts can use whatever templating they already have.
    """

    @abstractmethod
    def render(self, template_name: str, variables: dict[str, Any]) -> str:
        """Render ``template_name`` with ``variables``."""


class SchemaValidator(ABC):
    """Validates specialist output against a JSON Schema."""

    @abstractmethod
    def validate(self, instance: Any, schema: dict[str, Any]) -> list[str]:
        """Return a list of human-readable violations; empty means valid."""


class ResultSink(ABC):
    """Optional observer for completed specialist runs.

    Hooked up to tracing or metrics. Must never raise: an observability failure
    should not fail a task.
    """

    @abstractmethod
    def on_result(self, result: SpecialistResult) -> None:
        """Called once per specialist run, on success and on failure alike."""
