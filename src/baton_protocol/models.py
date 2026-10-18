"""Wire types for BATON.

Every type here serializes to plain JSON and preserves fields it does not
recognize, so implementations can interoperate across versions. The JSON
Schemas under ``baton_protocol/schemas/`` are the normative definition; this
module is the Python binding to them.

Design constraints worth knowing before extending anything:

* Identifiers (``tenant_id``, ``principal_id``) are **opaque strings**. The
  protocol never interprets them and carries no personal data of its own.
* Timestamps are timezone-aware UTC, serialized as ISO 8601 with an offset.
* Nothing here performs I/O. Persistence, model calls and tool execution all
  live behind the interfaces in :mod:`baton_protocol.interfaces`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Sequence

from ._codec import (
    drop_none,
    from_iso,
    merge_unknown,
    split_unknown,
    to_iso,
    utcnow,
)
from .budget import TaskBudget
from .errors import IncompatibleProtocolVersion
from .version import PROTOCOL_VERSION, is_compatible

#: Sentinel meaning "every capability on this task may read this entry".
VISIBLE_TO_ALL = "*"


def _new_id() -> str:
    return str(uuid.uuid4())


# ── Enums ───────────────────────────────────────────────────────────────────


class SpecialistStatus(str, Enum):
    """Lifecycle state of a single specialist run."""

    IDLE = "idle"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    AWAITING_DELEGATE = "awaiting_delegate"
    COMPLETED = "completed"
    FAILED = "failed"
    #: Hit the iteration ceiling but produced usable partial output.
    TRUNCATED = "truncated"


class SubtaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class MemoryScope(str, Enum):
    """How long a specialist's consolidated memory stays relevant.

    This governs *retention only*. What context a specialist receives at spawn
    is controlled separately by :attr:`SpecialistConfig.context_keys` — the two
    were conflated in earlier drafts, which made it impossible to give a
    session-scoped specialist project context.
    """

    SHORT_TERM = "short_term"
    SESSION = "session"
    PROJECT = "project"


class GapKind(str, Enum):
    """Why a task cannot proceed on the information it has."""

    MISSING_INPUT = "missing_input"
    MISSING_CAPABILITY = "missing_capability"
    MISSING_AUTHORIZATION = "missing_authorization"
    AMBIGUOUS_INTENT = "ambiguous_intent"
    EXTERNAL_DEPENDENCY = "external_dependency"


# ── Tool surface ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolSpec:
    """A tool as advertised to the model.

    ``side_effect`` drives the approval gate, so it defaults to ``True``:
    an unclassified tool is treated as dangerous.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    side_effect: bool = True

    def to_openai_function(self) -> dict[str, Any]:
        """Render as an OpenAI-compatible function-calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
                or {"type": "object", "properties": {}},
            },
        }


@dataclass(frozen=True)
class ToolCall:
    """A model's request to invoke a tool."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """Outcome of a tool invocation."""

    success: bool
    data: Any = None
    error: str | None = None

    def as_message_content(self) -> str:
        """Render for insertion into the model's message history."""
        if self.success:
            return str(self.data)
        return f"ERROR: {self.error}"


@dataclass(frozen=True)
class LLMResponse:
    """Provider-neutral model response.

    Gateways normalize whatever their provider returns into this. The reference
    loop never touches provider-shaped payloads, which is what lets the same
    loop run against OpenAI, Anthropic or a local model.
    """

    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    prompt_tokens: int = 0
    completion_tokens: int = 0
    #: Kept for observability. Never interpreted by the protocol.
    raw: Any = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def is_final(self) -> bool:
        """A response with no tool calls is the specialist's final answer."""
        return not self.tool_calls


# ── Task decomposition ──────────────────────────────────────────────────────

_SUBTASK_KNOWN = frozenset(
    {
        "id",
        "required_capability",
        "parameters",
        "depends_on",
        "expected_output_schema",
        "idempotency_key",
        "status",
        "description",
        "consumes",
    }
)


@dataclass
class Subtask:
    """One unit of work assigned to exactly one capability.

    ``depends_on`` is what makes blackboard reads safe. Without it, a consumer
    only sees its producer's output because of accidental list ordering.
    """

    required_capability: str
    id: str = field(default_factory=_new_id)
    parameters: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    expected_output_schema: dict[str, Any] | None = None
    #: Stable key so a retried task does not re-fire side-effecting tools.
    idempotency_key: str | None = None
    status: SubtaskStatus = SubtaskStatus.PENDING
    description: str | None = None
    #: Blackboard addresses (``capability.key``) this subtask needs. Declaring
    #: them lets the orchestrator refuse to run it when an input never arrived,
    #: instead of discovering that from a specialist's wrong answer.
    consumes: list[str] = field(default_factory=list)
    unknown: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return merge_unknown(
            drop_none(
                {
                    "id": self.id,
                    "required_capability": self.required_capability,
                    "parameters": self.parameters,
                    "depends_on": list(self.depends_on),
                    "expected_output_schema": self.expected_output_schema,
                    "idempotency_key": self.idempotency_key,
                    "status": self.status.value,
                    "description": self.description,
                    # Emitted only when set, so documents that never use the
                    # feature serialize exactly as they did before it existed.
                    "consumes": list(self.consumes) or None,
                }
            ),
            self.unknown,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Subtask":
        known, unknown = split_unknown(data, _SUBTASK_KNOWN)
        if "required_capability" not in known:
            raise ValueError("subtask requires 'required_capability'")
        return cls(
            required_capability=str(known["required_capability"]),
            id=str(known.get("id") or _new_id()),
            parameters=dict(known.get("parameters") or {}),
            depends_on=list(known.get("depends_on") or []),
            expected_output_schema=known.get("expected_output_schema"),
            idempotency_key=known.get("idempotency_key"),
            status=SubtaskStatus(known.get("status", SubtaskStatus.PENDING.value)),
            description=known.get("description"),
            consumes=[str(a) for a in (known.get("consumes") or [])],
            unknown=unknown,
        )


_GAP_KNOWN = frozenset({"id", "kind", "description", "blocking", "resolved", "subtask_id"})


@dataclass
class GapFlag:
    """A recorded hole in the task's information or authority.

    Surfacing gaps explicitly is what allows a task to stop and ask rather than
    inventing a plausible answer.
    """

    kind: GapKind
    description: str
    id: str = field(default_factory=_new_id)
    blocking: bool = True
    resolved: bool = False
    subtask_id: str | None = None
    unknown: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return merge_unknown(
            drop_none(
                {
                    "id": self.id,
                    "kind": self.kind.value,
                    "description": self.description,
                    "blocking": self.blocking,
                    "resolved": self.resolved,
                    "subtask_id": self.subtask_id,
                }
            ),
            self.unknown,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GapFlag":
        known, unknown = split_unknown(data, _GAP_KNOWN)
        return cls(
            kind=GapKind(known["kind"]),
            description=str(known.get("description", "")),
            id=str(known.get("id") or _new_id()),
            blocking=bool(known.get("blocking", True)),
            resolved=bool(known.get("resolved", False)),
            subtask_id=known.get("subtask_id"),
            unknown=unknown,
        )


# ── Blackboard ──────────────────────────────────────────────────────────────

_ENTRY_KNOWN = frozenset(
    {
        "id",
        "seq",
        "posted_by",
        "capability",
        "key",
        "value",
        "created_at",
        "visible_to",
        "schema_ref",
        "derived_from",
    }
)


@dataclass(frozen=True)
class BlackboardEntry:
    """An immutable fact posted by one specialist for others to read.

    ``seq`` is a monotonic per-task counter. Wall-clock timestamps are not
    reliable for ordering — two posts inside the same millisecond are common —
    and readers need a deterministic "latest".
    """

    capability: str
    key: str
    value: Any
    posted_by: str
    seq: int
    id: str = field(default_factory=_new_id)
    created_at: datetime = field(default_factory=utcnow)
    #: Capability names permitted to read this, or ``[VISIBLE_TO_ALL]``.
    visible_to: tuple[str, ...] = (VISIBLE_TO_ALL,)
    #: Optional JSON Schema reference describing ``value``, so consumers can
    #: validate rather than trust the shape.
    schema_ref: str | None = None
    #: ``seq`` of every entry this fact was derived from. Lets an auditor walk
    #: from a conclusion back to the facts it rested on.
    derived_from: tuple[int, ...] = ()
    unknown: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def qualified_key(self) -> str:
        """The ``capability.key`` address used when reading."""
        return f"{self.capability}.{self.key}"

    def to_dict(self) -> dict[str, Any]:
        return merge_unknown(
            drop_none(
                {
                    "id": self.id,
                    "seq": self.seq,
                    "posted_by": self.posted_by,
                    "capability": self.capability,
                    "key": self.key,
                    "value": self.value,
                    "created_at": to_iso(self.created_at),
                    "visible_to": list(self.visible_to),
                    "schema_ref": self.schema_ref,
                    "derived_from": list(self.derived_from) or None,
                }
            ),
            self.unknown,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BlackboardEntry":
        known, unknown = split_unknown(data, _ENTRY_KNOWN)
        return cls(
            capability=str(known["capability"]),
            key=str(known["key"]),
            value=known.get("value"),
            posted_by=str(known.get("posted_by", "")),
            seq=int(known.get("seq", 0)),
            id=str(known.get("id") or _new_id()),
            created_at=from_iso(known.get("created_at")),
            visible_to=tuple(known.get("visible_to") or (VISIBLE_TO_ALL,)),
            schema_ref=known.get("schema_ref"),
            derived_from=tuple(int(n) for n in (known.get("derived_from") or ())),
            unknown=unknown,
        )


# ── Specialist configuration ────────────────────────────────────────────────


@dataclass(frozen=True)
class SpecialistConfig:
    """Static declaration of one capability's authority and limits.

    Two separate authority grants are declared here, and both are enforced:
    ``tools`` bounds what the specialist may *do*, and ``can_delegate_to``
    bounds what it may *ask others to do*. Omitting the second grant is the
    larger mistake, since delegation transitively confers every tool the
    delegate holds.
    """

    capability: str
    #: Tool names this capability may use. The effective set is intersected
    #: with what the principal has actually authorized at runtime.
    tools: tuple[str, ...] = ()
    #: Capabilities this one may delegate to. Empty means no delegation.
    can_delegate_to: tuple[str, ...] = ()
    prompt_template: str = ""
    memory_scope: MemoryScope = MemoryScope.SHORT_TERM
    #: Context keys to carve out of the envelope. Nothing else is passed
    #: through, so a specialist cannot read context it has no use for.
    context_keys: tuple[str, ...] = ()
    max_iterations: int = 5
    timeout_seconds: float = 120.0
    #: Per-run token ceiling, checked alongside the shared task budget.
<<<<<<< HEAD
=======
    #: Zero means no per-run ceiling.
>>>>>>> 4d653be (BATON 1.0: specification, reference library, JSON Schemas and conformance suite)
    max_tokens: int = 50_000
    #: Max output tokens for ONE turn. Zero defers to the gateway's default.
    #: A specialist whose job is to write the deliverable needs a far larger
    #: allowance than one doing tool-calling research: the same ceiling that is
    #: generous for a tool call truncates a report into a summary.
    completion_tokens: int = 0
    model: str = ""
    temperature: float = 0.2
    #: JSON Schema the final output must satisfy before it reaches the
    #: blackboard. A violation is retryable inside the loop.
    output_schema: dict[str, Any] | None = None
    #: Skip the human approval gate for side-effecting tools. Intended for
    #: tests and explicitly unattended automations only.
    auto_approve_side_effects: bool = False
    #: Capabilities allowed to read this specialist's output from the
    #: blackboard. ``None`` means every capability on the task.
    output_visible_to: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.capability:
            raise ValueError("capability must be a non-empty string")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")


# ── Context envelope and slices ─────────────────────────────────────────────

_SLICE_KNOWN = frozenset(
    {
        "task_id",
        "tenant_id",
        "principal_id",
        "session_id",
        "intent",
        "capability",
        "data",
        "depth",
        "ancestry",
    }
)


@dataclass(frozen=True)
class ContextSlice:
    """The bounded view of task context handed to one specialist.

    A specialist never receives the full envelope. This is a containment
    boundary, not an optimization: it stops one capability's context leaking
    into another's prompt, and it keeps prompts small enough to stay coherent.
    """

    task_id: str
    tenant_id: str
    principal_id: str
    capability: str
    intent: str = ""
    session_id: str | None = None
    #: Keys selected by :attr:`SpecialistConfig.context_keys`.
    data: dict[str, Any] = field(default_factory=dict)
    #: Delegation depth of the specialist receiving this slice; 0 at the root.
    depth: int = 0
    #: Capability chain from the root to this specialist, used for cycle
    #: detection and for attributing spend.
    ancestry: tuple[str, ...] = ()
    unknown: dict[str, Any] = field(default_factory=dict, repr=False)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return merge_unknown(
            drop_none(
                {
                    "task_id": self.task_id,
                    "tenant_id": self.tenant_id,
                    "principal_id": self.principal_id,
                    "session_id": self.session_id,
                    "intent": self.intent,
                    "capability": self.capability,
                    "data": self.data,
                    "depth": self.depth,
                    "ancestry": list(self.ancestry),
                }
            ),
            self.unknown,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ContextSlice":
        known, unknown = split_unknown(data, _SLICE_KNOWN)
        return cls(
            task_id=str(known["task_id"]),
            tenant_id=str(known["tenant_id"]),
            principal_id=str(known["principal_id"]),
            capability=str(known["capability"]),
            intent=str(known.get("intent", "")),
            session_id=known.get("session_id"),
            data=dict(known.get("data") or {}),
            depth=int(known.get("depth", 0)),
            ancestry=tuple(known.get("ancestry") or ()),
            unknown=unknown,
        )


_ENVELOPE_KNOWN = frozenset(
    {
        "protocol_version",
        "envelope_id",
        "task_id",
        "tenant_id",
        "principal_id",
        "session_id",
        "intent",
        "context_data",
        "subtasks",
        "gap_flags",
        "agent_chain",
        "blackboard_snapshot",
        "budget",
        "created_at",
        "updated_at",
        "revision",
        "extensions",
    }
)


@dataclass
class ContextEnvelope:
    """The persistent, versioned record of one task.

    This is the unit implementations exchange and store. It is the whole point
    of the protocol: a task's context, decomposition, known gaps, shared facts
    and spend travel together as one auditable object, so any participant can
    pick a task up without reconstructing state from logs.
    """

    task_id: str
    tenant_id: str
    principal_id: str
    intent: str = ""
    protocol_version: str = PROTOCOL_VERSION
    envelope_id: str = field(default_factory=_new_id)
    session_id: str | None = None
    context_data: dict[str, Any] = field(default_factory=dict)
    subtasks: list[Subtask] = field(default_factory=list)
    gap_flags: list[GapFlag] = field(default_factory=list)
    #: Capabilities that have touched this task, in order.
    agent_chain: list[str] = field(default_factory=list)
    blackboard_snapshot: list[BlackboardEntry] = field(default_factory=list)
    budget: TaskBudget = field(default_factory=TaskBudget)
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    #: Incremented on every mutation, for optimistic concurrency control in
    #: stores that support it.
    revision: int = 1
    #: Namespace for host-specific data that is not part of the protocol.
    extensions: dict[str, Any] = field(default_factory=dict)
    unknown: dict[str, Any] = field(default_factory=dict, repr=False)

    # ── Subtask graph ───────────────────────────────────────────────────────

    def subtask(self, subtask_id: str) -> Subtask | None:
        return next((s for s in self.subtasks if s.id == subtask_id), None)

    def record_result(self, subtask_id: str, output: Any) -> None:
        """Attach a subtask's output and mark it complete."""
        target = self.subtask(subtask_id)
        if target is None:
            raise KeyError(f"unknown subtask: {subtask_id!r}")
        target.status = SubtaskStatus.COMPLETED
        self.context_data.setdefault("results", {})[subtask_id] = output
        self.touch()

    def flag_gap(self, kind: GapKind, description: str, **kwargs: Any) -> GapFlag:
        gap = GapFlag(kind=kind, description=description, **kwargs)
        self.gap_flags.append(gap)
        self.touch()
        return gap

    @property
    def blocking_gaps(self) -> list[GapFlag]:
        return [g for g in self.gap_flags if g.blocking and not g.resolved]

    def touch(self) -> None:
        self.updated_at = utcnow()
        self.revision += 1

    # ── Slicing ─────────────────────────────────────────────────────────────

    def slice_for(
        self,
        config: SpecialistConfig,
        *,
        depth: int = 0,
        ancestry: Sequence[str] = (),
    ) -> ContextSlice:
        """Carve the slice a specialist is allowed to see.

        Only keys named in ``config.context_keys`` cross the boundary. Identity
        fields always travel, because a specialist cannot meter usage or scope a
        tool call without knowing who it acts for.
        """
        return ContextSlice(
            task_id=self.task_id,
            tenant_id=self.tenant_id,
            principal_id=self.principal_id,
            capability=config.capability,
            intent=self.intent,
            session_id=self.session_id,
            data={
                key: self.context_data[key]
                for key in config.context_keys
                if key in self.context_data
            },
            depth=depth,
            ancestry=tuple(ancestry),
        )

    # ── Serialization ───────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return merge_unknown(
            drop_none(
                {
                    "protocol_version": self.protocol_version,
                    "envelope_id": self.envelope_id,
                    "task_id": self.task_id,
                    "tenant_id": self.tenant_id,
                    "principal_id": self.principal_id,
                    "session_id": self.session_id,
                    "intent": self.intent,
                    "context_data": self.context_data,
                    "subtasks": [s.to_dict() for s in self.subtasks],
                    "gap_flags": [g.to_dict() for g in self.gap_flags],
                    "agent_chain": list(self.agent_chain),
                    "blackboard_snapshot": [
                        e.to_dict() for e in self.blackboard_snapshot
                    ],
                    "budget": self.budget.to_dict(),
                    "created_at": to_iso(self.created_at),
                    "updated_at": to_iso(self.updated_at),
                    "revision": self.revision,
                    "extensions": self.extensions,
                }
            ),
            self.unknown,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ContextEnvelope":
        version = str(data.get("protocol_version", PROTOCOL_VERSION))
        if not is_compatible(version):
            raise IncompatibleProtocolVersion(version, PROTOCOL_VERSION)

        known, unknown = split_unknown(data, _ENVELOPE_KNOWN)
        return cls(
            task_id=str(known["task_id"]),
            tenant_id=str(known["tenant_id"]),
            principal_id=str(known["principal_id"]),
            intent=str(known.get("intent", "")),
            protocol_version=version,
            envelope_id=str(known.get("envelope_id") or _new_id()),
            session_id=known.get("session_id"),
            context_data=dict(known.get("context_data") or {}),
            subtasks=[Subtask.from_dict(s) for s in known.get("subtasks") or []],
            gap_flags=[GapFlag.from_dict(g) for g in known.get("gap_flags") or []],
            agent_chain=list(known.get("agent_chain") or []),
            blackboard_snapshot=[
                BlackboardEntry.from_dict(e)
                for e in known.get("blackboard_snapshot") or []
            ],
            budget=TaskBudget.from_dict(known.get("budget") or {}),
            created_at=from_iso(known.get("created_at")),
            updated_at=from_iso(known.get("updated_at")),
            revision=int(known.get("revision", 1)),
            extensions=dict(known.get("extensions") or {}),
            unknown=unknown,
        )


# ── Delegation ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DelegationRequest:
    """One specialist's request that another capability do some work.

    Carries ``ancestry`` and ``depth`` so the orchestrator can reject cycles
    and runaway recursion at the point of request, before any spend occurs.
    """

    requesting_specialist_id: str
    requesting_capability: str
    required_capability: str
    input_data: dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=_new_id)
    depth: int = 0
    ancestry: tuple[str, ...] = ()
    expected_output_schema: dict[str, Any] | None = None


# ── Results and suspended runs ──────────────────────────────────────────────


@dataclass
class SpecialistResult:
    """The outcome of one specialist run."""

    specialist_id: str
    subtask_id: str
    capability: str
    status: SpecialistStatus
    output: Any = None
    confidence: float = 0.0
    tokens_used: int = 0
    duration_ms: int = 0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        """Truncated runs count as usable: partial output beats no output."""
        return self.status in (
            SpecialistStatus.COMPLETED,
            SpecialistStatus.TRUNCATED,
        )

    def to_dict(self) -> dict[str, Any]:
        return drop_none(
            {
                "specialist_id": self.specialist_id,
                "subtask_id": self.subtask_id,
                "capability": self.capability,
                "status": self.status.value,
                "output": self.output,
                "confidence": self.confidence,
                "tokens_used": self.tokens_used,
                "duration_ms": self.duration_ms,
                "error": self.error,
                "metadata": self.metadata,
            }
        )


_APPROVAL_KNOWN = frozenset(
    {"tool_name", "arguments", "call_id", "iteration", "requested_at", "idempotency_key"}
)


@dataclass
class PendingApproval:
    """A side-effecting tool call frozen awaiting human authorization."""

    tool_name: str
    arguments: dict[str, Any]
    call_id: str
    iteration: int
    idempotency_key: str | None = None
    requested_at: datetime = field(default_factory=utcnow)
    unknown: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return merge_unknown(
            drop_none(
                {
                    "tool_name": self.tool_name,
                    "arguments": self.arguments,
                    "call_id": self.call_id,
                    "iteration": self.iteration,
                    "idempotency_key": self.idempotency_key,
                    "requested_at": to_iso(self.requested_at),
                }
            ),
            self.unknown,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PendingApproval":
        known, unknown = split_unknown(data, _APPROVAL_KNOWN)
        return cls(
            tool_name=str(known["tool_name"]),
            arguments=dict(known.get("arguments") or {}),
            call_id=str(known.get("call_id", "")),
            iteration=int(known.get("iteration", 0)),
            idempotency_key=known.get("idempotency_key"),
            requested_at=from_iso(known.get("requested_at")),
            unknown=unknown,
        )


_RUN_KNOWN = frozenset(
    {
        "specialist_id",
        "capability",
        "subtask_id",
        "task_id",
        "messages",
        "iteration",
        "tokens_used",
        "status",
        "pending_approval",
        "depth",
        "ancestry",
        "started_at",
        "schema_retries",
    }
)


@dataclass
class SpecialistRun:
    """Serializable state of an in-flight specialist run.

    This exists so the agentic loop can be *suspended*. A side-effecting tool
    call may wait hours for a human, far longer than any request or worker
    thread survives, so the loop's position — message history, iteration count,
    accumulated spend — has to be durable rather than living on a stack.
    """

    specialist_id: str
    capability: str
    subtask_id: str
    task_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    iteration: int = 0
    tokens_used: int = 0
    status: SpecialistStatus = SpecialistStatus.IDLE
    pending_approval: PendingApproval | None = None
    depth: int = 0
    ancestry: tuple[str, ...] = ()
    started_at: datetime = field(default_factory=utcnow)
    #: Consecutive output-schema failures, so a model that cannot produce
    #: valid output fails instead of looping until the budget is gone.
    schema_retries: int = 0
    #: Fields written by a newer producer, kept so that a run parked by one
    #: version and resumed by another does not lose them.
    unknown: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return merge_unknown(drop_none(
            {
                "specialist_id": self.specialist_id,
                "capability": self.capability,
                "subtask_id": self.subtask_id,
                "task_id": self.task_id,
                "messages": self.messages,
                "iteration": self.iteration,
                "tokens_used": self.tokens_used,
                "status": self.status.value,
                "pending_approval": (
                    self.pending_approval.to_dict() if self.pending_approval else None
                ),
                "depth": self.depth,
                "ancestry": list(self.ancestry),
                "started_at": to_iso(self.started_at),
                "schema_retries": self.schema_retries,
            }
        ), self.unknown)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SpecialistRun":
        known, unknown = split_unknown(data, _RUN_KNOWN)
        approval = known.get("pending_approval")
        return cls(
            specialist_id=str(known["specialist_id"]),
            capability=str(known["capability"]),
            subtask_id=str(known["subtask_id"]),
            task_id=str(known["task_id"]),
            messages=list(known.get("messages") or []),
            iteration=int(known.get("iteration", 0)),
            tokens_used=int(known.get("tokens_used", 0)),
            status=SpecialistStatus(known.get("status", SpecialistStatus.IDLE.value)),
            pending_approval=(
                PendingApproval.from_dict(approval) if approval else None
            ),
            depth=int(known.get("depth", 0)),
            ancestry=tuple(known.get("ancestry") or ()),
            started_at=from_iso(known.get("started_at")),
            schema_retries=int(known.get("schema_retries", 0)),
            unknown=unknown,
        )
