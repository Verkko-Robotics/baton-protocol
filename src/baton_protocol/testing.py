"""In-memory gateway implementations for tests and conformance checks.

These are part of the public API on purpose. They define the expected semantics
of each interface far more precisely than prose can, and they let a host verify
its own adapters behave the same way. They are also what makes the whole
framework — including approval suspension and delegation — testable with no
network, no database and no model provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterator, Sequence

from ._codec import utcnow
from .interfaces import (
    ApprovalGateway,
    Clock,
    ContextStore,
    LLMGateway,
    MemoryStore,
    ResultSink,
    TemplateRenderer,
    ToolGateway,
    UsageMeter,
)
from .validation import JsonSchemaValidator  # re-exported for convenience

__all__ = [
    "CallableRenderer",
    "CollectingSink",
    "FrozenClock",
    "InMemoryContextStore",
    "InMemoryMemoryStore",
    "JsonSchemaValidator",
    "RecordingApprovalGateway",
    "RecordingMeter",
    "ScriptedLLM",
    "ScriptedToolGateway",
    "ToolInvocation",
    "text_response",
    "tool_response",
]
from .models import (
    BlackboardEntry,
    ContextEnvelope,
    ContextSlice,
    LLMResponse,
    PendingApproval,
    SpecialistResult,
    ToolCall,
    ToolResult,
    ToolSpec,
)


class FrozenClock(Clock):
    """Manually advanced clock, so timeout behaviour is testable instantly."""

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 1, 1, tzinfo=timezone.utc)
        self._mono = 0.0

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._mono

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)
        self._mono += seconds


class InMemoryContextStore(ContextStore):
    """Dict-backed :class:`ContextStore`."""

    def __init__(self) -> None:
        self.envelopes: dict[str, ContextEnvelope] = {}
        self.appended: list[tuple[str, BlackboardEntry]] = []

    def load(self, task_id: str) -> ContextEnvelope | None:
        return self.envelopes.get(task_id)

    def save(self, envelope: ContextEnvelope) -> None:
        self.envelopes[envelope.task_id] = envelope

    def append_blackboard(self, task_id: str, entry: BlackboardEntry) -> None:
        self.appended.append((task_id, entry))


class InMemoryMemoryStore(MemoryStore):
    """List-backed :class:`MemoryStore` with naive substring recall."""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def consolidate(
        self,
        *,
        principal_id: str,
        capability: str,
        summary: str,
        metadata: dict[str, Any],
    ) -> str:
        item = {
            "id": str(len(self.items) + 1),
            "principal_id": principal_id,
            "capability": capability,
            "summary": summary,
            "metadata": metadata,
            "created_at": utcnow(),
        }
        self.items.append(item)
        return item["id"]

    def recall(
        self,
        *,
        principal_id: str,
        capability: str,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        matches = [
            item
            for item in self.items
            if item["principal_id"] == principal_id and item["capability"] == capability
        ]
        return matches[-limit:]


@dataclass
class ToolInvocation:
    """One recorded call to :meth:`ScriptedToolGateway.execute`."""

    tool_name: str
    principal_id: str
    arguments: dict[str, Any]
    idempotency_key: str | None


class ScriptedToolGateway(ToolGateway):
    """Tool gateway returning canned results and recording every call.

    Honours ``idempotency_key``: a repeated key returns the first result without
    re-recording an invocation, mirroring what a real gateway must do so a
    retried task does not repeat a side effect.
    """

    def __init__(
        self,
        specs: Sequence[ToolSpec] = (),
        results: dict[str, Any] | None = None,
        *,
        failures: dict[str, str] | None = None,
    ) -> None:
        self._specs = list(specs)
        self._results = dict(results or {})
        self._failures = dict(failures or {})
        self.invocations: list[ToolInvocation] = []
        self._by_key: dict[str, ToolResult] = {}

    def available_tools(self, principal_id: str) -> list[ToolSpec]:
        return list(self._specs)

    def execute(
        self,
        *,
        tool_name: str,
        principal_id: str,
        arguments: dict[str, Any],
        idempotency_key: str | None = None,
    ) -> ToolResult:
        if idempotency_key and idempotency_key in self._by_key:
            return self._by_key[idempotency_key]

        self.invocations.append(
            ToolInvocation(tool_name, principal_id, dict(arguments), idempotency_key)
        )
        if tool_name in self._failures:
            result = ToolResult(success=False, error=self._failures[tool_name])
        else:
            result = ToolResult(
                success=True, data=self._results.get(tool_name, "ok")
            )
        if idempotency_key:
            self._by_key[idempotency_key] = result
        return result


class ScriptedLLM(LLMGateway):
    """Replays a fixed sequence of :class:`LLMResponse` objects.

    Records the messages it was given at each turn, which is how tests assert
    that the message history stays well-formed — in particular that every
    ``role: "tool"`` message is preceded by an assistant turn declaring the
    matching ``tool_call_id``.
    """

    def __init__(self, responses: Sequence[LLMResponse]) -> None:
        self._responses: Iterator[LLMResponse] = iter(responses)
        self.calls: list[list[dict[str, Any]]] = []
        self.offered_tools: list[list[str]] = []

    def complete(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[ToolSpec],
        model: str,
        temperature: float,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls.append([dict(m) for m in messages])
        self.offered_tools.append([t.name for t in tools])
        try:
            return next(self._responses)
        except StopIteration as exc:
            raise AssertionError(
                "ScriptedLLM exhausted: the loop asked for more turns than the "
                "test scripted"
            ) from exc


class RecordingApprovalGateway(ApprovalGateway):
    """Records approval requests and lets a test resolve them."""

    def __init__(self) -> None:
        self.requests: list[tuple[ContextSlice, PendingApproval]] = []
        self._status: dict[str, str] = {}

    def request(
        self, *, context: ContextSlice, approval: PendingApproval
    ) -> str:
        approval_id = f"approval-{len(self.requests) + 1}"
        self.requests.append((context, approval))
        self._status[approval_id] = "pending"
        return approval_id

    def status(self, approval_id: str) -> str:
        return self._status.get(approval_id, "pending")

    def resolve(self, approval_id: str, approved: bool) -> None:
        self._status[approval_id] = "approved" if approved else "denied"


class RecordingMeter(UsageMeter):
    """Captures every metered spend event."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

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
        self.records.append(
            {
                "principal_id": principal_id,
                "capability": capability,
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "idempotency_key": idempotency_key,
                "metadata": metadata or {},
            }
        )

    @property
    def total_tokens(self) -> int:
        return sum(r["prompt_tokens"] + r["completion_tokens"] for r in self.records)


class CallableRenderer(TemplateRenderer):
    """Renders via a plain function, avoiding a templating dependency."""

    def __init__(self, fn: Callable[[str, dict[str, Any]], str]) -> None:
        self._fn = fn

    def render(self, template_name: str, variables: dict[str, Any]) -> str:
        return self._fn(template_name, variables)


@dataclass
class CollectingSink(ResultSink):
    """Collects results for assertions."""

    results: list[SpecialistResult] = field(default_factory=list)

    def on_result(self, result: SpecialistResult) -> None:
        self.results.append(result)


# ── Convenience builders ────────────────────────────────────────────────────


def text_response(content: str, *, tokens: int = 10) -> LLMResponse:
    """A final answer with no tool calls."""
    return LLMResponse(
        content=content, prompt_tokens=tokens, completion_tokens=tokens
    )


def tool_response(
    *calls: tuple[str, dict[str, Any]],
    tokens: int = 10,
    content: str = "",
) -> LLMResponse:
    """A response requesting one or more tool calls."""
    return LLMResponse(
        content=content,
        tool_calls=tuple(
            ToolCall(id=f"call-{i + 1}", name=name, arguments=args)
            for i, (name, args) in enumerate(calls)
        ),
        prompt_tokens=tokens,
        completion_tokens=tokens,
    )
