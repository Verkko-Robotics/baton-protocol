"""Context slicing: the security boundary of the whole system.

Each specialist may see only the keys named in its own ``SpecialistConfig.
context_keys`` plus the identity fields it cannot operate without. Everything
else — other specialists' data, the budget, the raw subtask graph, secrets —
stays on the host side and never enters the prompt. These tests pin that
boundary down, because a silent leak here is a data-isolation failure, not a
quality one.
"""

from __future__ import annotations

from baton_protocol import (
    ContextEnvelope,
    LoopDependencies,
    SpecialistConfig,
    Subtask,
    TaskBudget,
)
from baton_protocol.orchestrator import Orchestrator
from baton_protocol.testing import (
    CollectingSink,
    FrozenClock,
    RecordingApprovalGateway,
    RecordingMeter,
    ScriptedLLM,
    ScriptedToolGateway,
    text_response,
)

TENANT = "tenant-a"
PRINCIPAL = "user-1"
TASK = "task-1"


def make_config(capability: str = "researcher", **overrides) -> SpecialistConfig:
    defaults = dict(
        capability=capability,
        tools=(),
        model="test-model",
        max_iterations=4,
        timeout_seconds=60.0,
    )
    defaults.update(overrides)
    return SpecialistConfig(**defaults)


def make_envelope(
    *subtasks: Subtask,
    context_data: dict | None = None,
    **overrides,
) -> ContextEnvelope:
    defaults = dict(
        task_id=TASK,
        tenant_id=TENANT,
        principal_id=PRINCIPAL,
        intent="do the thing",
        subtasks=list(subtasks),
    )
    defaults.update(overrides)
    if context_data is not None:
        defaults["context_data"] = context_data
    return ContextEnvelope(**defaults)


def make_deps(responses) -> LoopDependencies:
    return LoopDependencies(
        llm=ScriptedLLM(responses),
        tools=ScriptedToolGateway(),
        approvals=RecordingApprovalGateway(),
        meter=RecordingMeter(),
        clock=FrozenClock(),
    )


# ── What crosses the boundary ───────────────────────────────────────────────


def test_named_context_keys_cross_the_boundary() -> None:
    config = make_config(context_keys=("region", "period"))
    envelope = make_envelope(context_data={
        "region": "EMEA", "period": "2026-Q1",
        "secret": "top secret", "other": "should not appear",
    })

    slice_ = envelope.slice_for(config)

    assert slice_.data == {"region": "EMEA", "period": "2026-Q1"}
    assert "secret" not in slice_.data
    assert "other" not in slice_.data


def test_identity_fields_always_cross_regardless_of_config() -> None:
    """A specialist needs to know who it acts for to meter usage and scope
    tools. These travel even when ``context_keys`` is empty and even when they
    are not in the envelope's context_data."""
    config = make_config(context_keys=())
    envelope = make_envelope()

    slice_ = envelope.slice_for(config)

    assert slice_.task_id == TASK
    assert slice_.tenant_id == TENANT
    assert slice_.principal_id == PRINCIPAL
    assert slice_.capability == "researcher"
    assert slice_.intent == "do the thing"


def test_empty_context_keys_yields_empty_context_data() -> None:
    config = make_config(context_keys=())
    envelope = make_envelope(context_data={"alpha": 1, "beta": 2})

    slice_ = envelope.slice_for(config)

    assert slice_.data == {}


def test_requested_key_not_present_in_envelope_is_omitted_not_error() -> None:
    """A key a specialist is allowed to see but that the host never set should
    be absent, not a KeyError or None-injected value."""
    config = make_config(context_keys=("present", "absent"))
    envelope = make_envelope(context_data={"present": "here"})

    slice_ = envelope.slice_for(config)

    assert slice_.data == {"present": "here"}


def test_identity_fields_cannot_be_shadowed_by_context_data() -> None:
    """If a host (or model) stuffed a forged principal_id into context_data,
    the real identity must still win the identity slot."""
    config = make_config(context_keys=("principal_id",))
    envelope = make_envelope(principal_id=PRINCIPAL, context_data={"principal_id": "forged"})

    slice_ = envelope.slice_for(config)

    # Identity slot holds the authoritative value, not the forged one.
    assert slice_.principal_id == PRINCIPAL
    # The raw value still travels in context_data, since the key was named.
    assert slice_.data["principal_id"] == "forged"


def test_slice_captures_depth_and_ancestry_for_delegation() -> None:
    config = make_config()
    envelope = make_envelope()

    slice_ = envelope.slice_for(config, depth=2, ancestry=("alpha", "beta"))

    assert slice_.depth == 2
    assert slice_.ancestry == ("alpha", "beta")


def test_slice_is_a_value_not_a_view() -> None:
    """Mutating the envelope after slicing must not change the slice that was
    handed to the specialist (and vice versa)."""
    config = make_config(context_keys=("region",))
    envelope = make_envelope(context_data={"region": "EMEA"})

    slice_ = envelope.slice_for(config)
    envelope.context_data["region"] = "APAC"
    slice_.data["region"] = "TAMPERED"

    assert envelope.context_data["region"] == "APAC"
    assert slice_.data["region"] == "TAMPERED"


# ── Slicing as seen end-to-end through the orchestrator ─────────────────────


def test_slice_handed_to_specialist_carries_exactly_authorized_keys() -> None:
    """The slice object is the boundary the specialist actually receives. A
    secret present in the envelope's context_data but not in the capability's
    context_keys must never appear in the slice, while a granted key must."""
    config = make_config(context_keys=("region",))
    envelope = make_envelope(
        Subtask(id="a", required_capability="researcher"),
        context_data={
            "region": "EMEA",
            "secret": "customer-ssn-123456",
        },
    )

    slice_ = envelope.slice_for(config)

    assert slice_.data == {"region": "EMEA"}
    assert "customer-ssn-123456" not in slice_.data
    # Identity travels even though it is not in context_keys.
    assert slice_.tenant_id == TENANT
    assert slice_.principal_id == PRINCIPAL


def test_subtask_parameters_reach_the_default_prompt() -> None:
    """The loop's *default* prompt builder renders the subtask's own parameters
    (and the intent) — not the shared slice's context_data. This pins down
    which channel actually carries data to the model under default wiring: per-
    subtask parameters. A secret that lives only in the envelope's shared
    context_data never enters the default prompt."""
    registry = {"researcher": make_config()}
    envelope = make_envelope(
        Subtask(id="a", required_capability="researcher", parameters={"query": "find users"}),
        context_data={"secret": "customer-ssn-123456"},
    )
    llm = ScriptedLLM([text_response("done")])
    orchestrator = Orchestrator(
        registry=registry,
        deps=LoopDependencies(
            llm=llm,
            tools=ScriptedToolGateway(),
            approvals=RecordingApprovalGateway(),
            meter=RecordingMeter(),
            clock=FrozenClock(),
        ),
    )

    outcome = orchestrator.run_task(envelope)

    assert not outcome.suspended
    assert llm.calls, "expected the loop to have invoked the model"
    prompt = llm.calls[0][0]["content"]
    # The subtask's own parameters are the authorized per-subtask channel.
    assert "find users" in prompt
    # A secret in the shared context_data is not injected by the default prompt.
    assert "customer-ssn-123456" not in prompt


def test_budget_is_not_leaked_to_a_specialist_that_does_not_declare_it() -> None:
    """The task budget is host-side accounting. Slicing copies only the keys a
    capability named, so a capability that never declares 'budget' can never
    see it, even if it sits in the envelope's context_data."""
    config = make_config(context_keys=("region",))  # does NOT name "budget"
    envelope = make_envelope(context_data={
        "region": "EMEA",
        "budget": TaskBudget(max_total_tokens=999),
    })

    slice_ = envelope.slice_for(config)

    assert "budget" not in slice_.data
    assert "region" in slice_.data
    # The slice carries no budget of its own.
    assert not hasattr(slice_, "budget")


def test_full_subtask_graph_does_not_leak_to_one_specialist() -> None:
    """A specialist is told *its* subtask, not every sibling's parameters. The
    raw graph stays on the envelope."""
    a = Subtask(id="a", required_capability="researcher")
    b = Subtask(id="b", required_capability="writer")
    config = make_config()
    envelope = make_envelope(a, b)

    slice_ = envelope.slice_for(config)

    # The slice carries intent and identity, but no copy of sibling subtasks'
    # parameters. Each specialist's parameters are injected per-subtask at
    # prompt-render time, never via the shared slice.
    assert "subtasks" not in slice_.data


# ── Result routing (the inverse boundary) ───────────────────────────────────


def test_result_sink_receives_specialist_output() -> None:
    registry = {"researcher": make_config()}
    envelope = make_envelope(Subtask(id="a", required_capability="researcher"))
    sink = CollectingSink()
    orchestrator = Orchestrator(
        registry=registry,
        deps=make_deps([text_response("the finding")]),
        result_sink=sink,
    )

    outcome = orchestrator.run_task(envelope)

    assert len(sink.results) == 1
    assert sink.results[0].output == "the finding"
    assert sink.results[0].capability == "researcher"


def test_failed_specialist_output_is_not_published_to_blackboard() -> None:
    """A failed run's output must not be posted as a fact another specialist
    might trust. Only successful outputs cross to the blackboard."""
    registry = {"researcher": make_config()}
    envelope = make_envelope(Subtask(id="a", required_capability="researcher"))
    orchestrator = Orchestrator(
        registry=registry,
        deps=make_deps([text_response("ok", tokens=10)]),
    )
    envelope.budget = TaskBudget(max_total_tokens=5)

    outcome = orchestrator.run_task(envelope)

    assert all(not r.succeeded for r in outcome.results)
    # A failed run posts nothing to the shared blackboard.
    assert envelope.blackboard_snapshot == []


# ── Delegation depth guard ──────────────────────────────────────────────────


def test_slice_ancestry_grows_as_delegation_recurses() -> None:
    """Each delegation level is recorded so the cycle/depth guards have
    something to check against."""
    config = make_config(can_delegate_to=("helper",))
    envelope = make_envelope()

    top = envelope.slice_for(config)
    child = envelope.slice_for(config, depth=1, ancestry=("researcher",))

    assert top.depth == 0
    assert top.ancestry == ()
    assert child.depth == 1
    assert child.ancestry == ("researcher",)


# ── Serialization round-trip preserves the boundary ─────────────────────────


def test_slice_round_trip_preserves_exactly_what_was_authorized() -> None:
    from baton_protocol.models import ContextSlice as CS

    config = make_config(context_keys=("region",))
    envelope = make_envelope(context_data={"region": "EMEA", "secret": "no"})

    slice_ = envelope.slice_for(config)
    revived = CS.from_dict(slice_.to_dict())

    assert revived.data == {"region": "EMEA"}
    assert "secret" not in revived.data
    assert revived.principal_id == PRINCIPAL
