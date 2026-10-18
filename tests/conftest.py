from __future__ import annotations

import pytest

from baton_protocol import (
    Blackboard,
    ContextEnvelope,
    ContextSlice,
    LoopDependencies,
    SpecialistConfig,
    Subtask,
    TaskBudget,
    ToolSpec,
)
from baton_protocol.testing import (
    FrozenClock,
    RecordingApprovalGateway,
    RecordingMeter,
    ScriptedLLM,
    ScriptedToolGateway,
)

TENANT = "tenant-a"
PRINCIPAL = "user-1"
TASK = "task-1"


@pytest.fixture
def clock() -> FrozenClock:
    return FrozenClock()


@pytest.fixture
def approvals() -> RecordingApprovalGateway:
    return RecordingApprovalGateway()


@pytest.fixture
def meter() -> RecordingMeter:
    return RecordingMeter()


@pytest.fixture
def budget() -> TaskBudget:
    return TaskBudget()


@pytest.fixture
def blackboard() -> Blackboard:
    return Blackboard(task_id=TASK, tenant_id=TENANT)


def read_tool(name: str = "web_search") -> ToolSpec:
    return ToolSpec(
        name=name,
        description="a read-only tool",
        parameters={"type": "object", "properties": {}},
        side_effect=False,
    )


def write_tool(name: str = "send_email") -> ToolSpec:
    return ToolSpec(
        name=name,
        description="a side-effecting tool",
        parameters={"type": "object", "properties": {}},
        side_effect=True,
    )


def make_config(capability: str = "researcher", **overrides) -> SpecialistConfig:
    defaults = dict(
        capability=capability,
        tools=("web_search",),
        model="test-model",
        max_iterations=4,
        timeout_seconds=60.0,
    )
    defaults.update(overrides)
    return SpecialistConfig(**defaults)


def make_slice(config: SpecialistConfig, **overrides) -> ContextSlice:
    defaults = dict(
        task_id=TASK,
        tenant_id=TENANT,
        principal_id=PRINCIPAL,
        capability=config.capability,
        intent="find things out",
    )
    defaults.update(overrides)
    return ContextSlice(**defaults)


def make_envelope(*subtasks: Subtask, **overrides) -> ContextEnvelope:
    defaults = dict(
        task_id=TASK,
        tenant_id=TENANT,
        principal_id=PRINCIPAL,
        intent="do the thing",
        subtasks=list(subtasks),
    )
    defaults.update(overrides)
    return ContextEnvelope(**defaults)


def make_deps(
    responses,
    *,
    specs=(),
    results=None,
    failures=None,
    clock=None,
    approvals=None,
    meter=None,
    **kwargs,
) -> LoopDependencies:
    return LoopDependencies(
        llm=ScriptedLLM(responses),
        tools=ScriptedToolGateway(specs, results, failures=failures),
        approvals=approvals or RecordingApprovalGateway(),
        meter=meter or RecordingMeter(),
        clock=clock or FrozenClock(),
        **kwargs,
    )
