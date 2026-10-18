"""BATON — Bounded Authority for Traceable Orchestrated Networks.

A vendor-neutral standard for building agent systems where every unit of work is
scoped, every inter-agent interaction is recorded, and every action that touches
the outside world is authorized and metered.

The protocol has four layers:

* **Memory** — long-term consolidation behind :class:`MemoryStore`.
* **Context** — the :class:`ContextEnvelope`, a versioned, persistent record of
  one task: its intent, decomposition, known gaps, shared facts and spend.
* **Orchestration** — the :class:`Orchestrator` and :class:`Blackboard`.
* **Tools** — external capability behind :class:`ToolGateway`.

Nothing in this package performs I/O. Every external touchpoint is an abstract
interface injected at construction, which is what allows BATON to be implemented
on any stack and tested without a network.

Quick start::

    from baton_protocol import (
        ContextEnvelope, Orchestrator, SpecialistConfig, Subtask, LoopDependencies,
    )

    envelope = ContextEnvelope(
        task_id="task-1", tenant_id="acme", principal_id="user-1",
        intent="Screen the shortlist and draft offers",
        subtasks=[Subtask(required_capability="screener")],
    )
    orchestrator = Orchestrator(registry={"screener": config}, deps=deps)
    outcome = orchestrator.run_task(envelope)
"""

from __future__ import annotations

from .blackboard import Blackboard
from .budget import TaskBudget
from .errors import (
    ApprovalDenied,
    ApprovalRequired,
    BudgetExceeded,
    CapabilityCollision,
    CapabilityNotRegistered,
    BatonError,
    DelegationCycle,
    DelegationDenied,
    DelegationDepthExceeded,
    IncompatibleProtocolVersion,
    IterationLimitExceeded,
    RecoverableError,
    SchemaViolation,
    SpecialistTimeout,
    TenantMismatch,
    ToolExecutionFailed,
    ToolNotAuthorized,
    UnresolvableDependencies,
)
from .interfaces import (
    ApprovalGateway,
    Clock,
    ContextStore,
    LLMGateway,
    MemoryStore,
    ResultSink,
    SchemaValidator,
    SystemClock,
    TemplateRenderer,
    ToolGateway,
    UsageMeter,
)
from .loop import (
    MAX_SCHEMA_RETRIES,
    PSEUDO_TOOLS,
    TOOL_BLACKBOARD_LIST,
    TOOL_BLACKBOARD_READ,
    TOOL_DELEGATE,
    TOOL_WILDCARD,
    AgenticLoop,
    LoopDependencies,
    RunOutcome,
)
from .models import (
    VISIBLE_TO_ALL,
    BlackboardEntry,
    ContextEnvelope,
    ContextSlice,
    DelegationRequest,
    GapFlag,
    GapKind,
    LLMResponse,
    MemoryScope,
    PendingApproval,
    SpecialistConfig,
    SpecialistResult,
    SpecialistRun,
    SpecialistStatus,
    Subtask,
    SubtaskStatus,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from .orchestrator import (
    SUSPENDED_RUNS_KEY,
    Orchestrator,
    TaskOutcome,
    merge_registries,
    topological_order,
)
from .audit import (
    CapabilityFlow,
    InformationFlowReport,
    audit_information_flow,
)
from .validation import JsonSchemaValidator
from .version import PROTOCOL_VERSION, __version__, is_compatible, parse_version

__all__ = [
    # Versioning
    "PROTOCOL_VERSION",
    "__version__",
    "is_compatible",
    "parse_version",
    # Context
    "ContextEnvelope",
    "ContextSlice",
    "Subtask",
    "SubtaskStatus",
    "GapFlag",
    "GapKind",
    "TaskBudget",
    # Blackboard
    "Blackboard",
    "BlackboardEntry",
    "VISIBLE_TO_ALL",
    # Specialists
    "SpecialistConfig",
    "SpecialistResult",
    "SpecialistRun",
    "SpecialistStatus",
    "MemoryScope",
    "PendingApproval",
    "DelegationRequest",
    # Execution
    "AgenticLoop",
    "LoopDependencies",
    "RunOutcome",
    "Orchestrator",
    "TaskOutcome",
    "topological_order",
    "merge_registries",
    "audit_information_flow",
    "InformationFlowReport",
    "CapabilityFlow",
    "SUSPENDED_RUNS_KEY",
    "MAX_SCHEMA_RETRIES",
    "PSEUDO_TOOLS",
    "TOOL_BLACKBOARD_READ",
    "TOOL_BLACKBOARD_LIST",
    "TOOL_DELEGATE",
    "TOOL_WILDCARD",
    # Tools and models
    "ToolSpec",
    "ToolCall",
    "ToolResult",
    "LLMResponse",
    # Interfaces
    "ApprovalGateway",
    "Clock",
    "ContextStore",
    "LLMGateway",
    "MemoryStore",
    "ResultSink",
    "SchemaValidator",
    "JsonSchemaValidator",
    "SystemClock",
    "TemplateRenderer",
    "ToolGateway",
    "UsageMeter",
    # Errors
    "BatonError",
    "RecoverableError",
    "ApprovalDenied",
    "ApprovalRequired",
    "BudgetExceeded",
    "CapabilityNotRegistered",
    "CapabilityCollision",
    "DelegationCycle",
    "DelegationDenied",
    "DelegationDepthExceeded",
    "IncompatibleProtocolVersion",
    "IterationLimitExceeded",
    "SchemaViolation",
    "SpecialistTimeout",
    "TenantMismatch",
    "ToolExecutionFailed",
    "ToolNotAuthorized",
    "UnresolvableDependencies",
]
