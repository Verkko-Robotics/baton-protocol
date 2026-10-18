"""Orchestration: dependency ordering, publication, delegation, suspension."""

from __future__ import annotations

import pytest

from baton_protocol import (
    SUSPENDED_RUNS_KEY,
    Orchestrator,
    SpecialistConfig,
    SpecialistStatus,
    Subtask,
    SubtaskStatus,
    TaskBudget,
    UnresolvableDependencies,
    topological_order,
)
from baton_protocol.testing import (
    CollectingSink,
    InMemoryContextStore,
    InMemoryMemoryStore,
    text_response,
    tool_response,
)

from conftest import make_deps, make_envelope, read_tool, write_tool


def config_for(capability: str, **overrides) -> SpecialistConfig:
    defaults = dict(capability=capability, model="test-model", tools=())
    defaults.update(overrides)
    return SpecialistConfig(**defaults)


# ── Dependency ordering ─────────────────────────────────────────────────────


class TestTopologicalOrder:
    def test_dependencies_come_first(self) -> None:
        a = Subtask(required_capability="a", id="a")
        b = Subtask(required_capability="b", id="b", depends_on=["a"])
        c = Subtask(required_capability="c", id="c", depends_on=["b"])

        assert [s.id for s in topological_order([c, b, a])] == ["a", "b", "c"]

    def test_independent_subtasks_keep_their_authored_order(self) -> None:
        """Determinism matters: runs should be reproducible."""
        subtasks = [Subtask(required_capability=c, id=c) for c in ("x", "y", "z")]

        assert [s.id for s in topological_order(subtasks)] == ["x", "y", "z"]

    def test_cycles_raise_rather_than_dropping_work(self) -> None:
        a = Subtask(required_capability="a", id="a", depends_on=["b"])
        b = Subtask(required_capability="b", id="b", depends_on=["a"])

        with pytest.raises(UnresolvableDependencies):
            topological_order([a, b])

    def test_unknown_dependency_raises(self) -> None:
        a = Subtask(required_capability="a", id="a", depends_on=["ghost"])

        with pytest.raises(UnresolvableDependencies):
            topological_order([a])


# ── Task execution ──────────────────────────────────────────────────────────


def test_results_are_published_to_the_blackboard_for_downstream_readers() -> None:
    """The only channel between specialists.

    The producer never hands its output to the consumer; it posts a fact and the
    consumer reads it by address.
    """
    producer = Subtask(required_capability="screener", id="s1")
    consumer = Subtask(required_capability="writer", id="s2", depends_on=["s1"])
    envelope = make_envelope(producer, consumer)

    deps = make_deps(
        [
            text_response("candidate ada"),
            tool_response(
                ("blackboard_read", {"capability": "screener", "key": "output"})
            ),
            text_response("offer drafted"),
        ]
    )
    orchestrator = Orchestrator(
        registry={"screener": config_for("screener"), "writer": config_for("writer")},
        deps=deps,
    )

    outcome = orchestrator.run_task(envelope)

    assert [r.status for r in outcome.results] == [
        SpecialistStatus.COMPLETED,
        SpecialistStatus.COMPLETED,
    ]
    writer_turn = deps.llm.calls[-1]
    tool_msg = next(m for m in writer_turn if m.get("role") == "tool")
    assert "candidate ada" in tool_msg["content"]


def test_output_visibility_is_restricted_when_declared() -> None:
    envelope = make_envelope(Subtask(required_capability="screener", id="s1"))
    deps = make_deps([text_response("classified")])
    orchestrator = Orchestrator(
        registry={
            "screener": config_for("screener", output_visible_to=("auditor",))
        },
        deps=deps,
    )

    orchestrator.run_task(envelope)
    entry = envelope.blackboard_snapshot[0]

    assert entry.visible_to == ("auditor",)


def test_dependents_are_skipped_when_a_prerequisite_fails() -> None:
    """Running a specialist whose input never arrived produces a confidently
    wrong answer, which is worse than an unfinished task."""
    producer = Subtask(required_capability="screener", id="s1")
    consumer = Subtask(required_capability="writer", id="s2", depends_on=["s1"])
    envelope = make_envelope(producer, consumer, budget=TaskBudget(max_total_tokens=5))

    deps = make_deps([text_response("too expensive", tokens=10)])
    orchestrator = Orchestrator(
        registry={"screener": config_for("screener"), "writer": config_for("writer")},
        deps=deps,
    )

    outcome = orchestrator.run_task(envelope)

    assert producer.status is SubtaskStatus.FAILED
    assert consumer.status is SubtaskStatus.SKIPPED
    assert len(outcome.results) == 1
    assert envelope.blocking_gaps == []  # skip is recorded as non-blocking


def test_completed_subtasks_are_not_re_executed_on_a_second_run() -> None:
    """Retrying a task (reset the failed subtask + downstream to PENDING,
    call run_task again on the same envelope) must not redo upstream work."""
    producer = Subtask(required_capability="screener", id="s1")
    consumer = Subtask(required_capability="writer", id="s2", depends_on=["s1"])
    envelope = make_envelope(producer, consumer, budget=TaskBudget(max_total_tokens=5))

    # First run: producer succeeds, writer's response is too expensive and fails.
    deps = make_deps(
        [text_response("classified", tokens=1), text_response("too expensive", tokens=10)]
    )
    orchestrator = Orchestrator(
        registry={"screener": config_for("screener"), "writer": config_for("writer")},
        deps=deps,
    )
    outcome = orchestrator.run_task(envelope)
    assert producer.status is SubtaskStatus.COMPLETED
    assert consumer.status is SubtaskStatus.FAILED
    assert len(outcome.results) == 2

    # Retry: bump the budget, reset only the failed subtask, re-run.
    envelope.budget = TaskBudget(max_total_tokens=1000)
    consumer.status = SubtaskStatus.PENDING

    deps2 = make_deps([text_response("finished")])
    orchestrator2 = Orchestrator(
        registry={"screener": config_for("screener"), "writer": config_for("writer")},
        deps=deps2,
    )
    outcome2 = orchestrator2.run_task(envelope)

    # Only the retried subtask ran; the producer's earlier result stands.
    assert len(outcome2.results) == 1
    assert outcome2.results[0].subtask_id == "s2"
    assert consumer.status is SubtaskStatus.COMPLETED
    assert envelope.context_data["results"]["s1"] == "classified"
    assert envelope.context_data["results"]["s2"] == "finished"


def test_envelope_is_persisted_with_the_blackboard_snapshot() -> None:
    envelope = make_envelope(Subtask(required_capability="screener", id="s1"))
    store = InMemoryContextStore()
    orchestrator = Orchestrator(
        registry={"screener": config_for("screener")},
        deps=make_deps([text_response("done")]),
        context_store=store,
    )

    orchestrator.run_task(envelope)

    saved = store.load(envelope.task_id)
    assert saved is not None
    assert len(saved.blackboard_snapshot) == 1
    assert store.appended  # durability hook fired per post


def test_each_specialist_consolidates_only_its_own_output() -> None:
    """Writing the whole blackboard per specialist makes memory writes
    quadratic in the number of specialists and stores the same facts N times."""
    envelope = make_envelope(
        Subtask(required_capability="screener", id="s1"),
        Subtask(required_capability="writer", id="s2"),
    )
    memory = InMemoryMemoryStore()
    orchestrator = Orchestrator(
        registry={"screener": config_for("screener"), "writer": config_for("writer")},
        deps=make_deps([text_response("a"), text_response("b")]),
        memory=memory,
    )

    orchestrator.run_task(envelope)

    assert [item["capability"] for item in memory.items] == ["screener", "writer"]
    assert [item["summary"] for item in memory.items] == ["a", "b"]


def test_result_sink_observes_every_run() -> None:
    envelope = make_envelope(Subtask(required_capability="screener", id="s1"))
    sink = CollectingSink()
    orchestrator = Orchestrator(
        registry={"screener": config_for("screener")},
        deps=make_deps([text_response("done")]),
        result_sink=sink,
    )

    orchestrator.run_task(envelope)

    assert len(sink.results) == 1


def test_agent_chain_records_who_touched_the_task() -> None:
    envelope = make_envelope(
        Subtask(required_capability="screener", id="s1"),
        Subtask(required_capability="writer", id="s2"),
    )
    orchestrator = Orchestrator(
        registry={"screener": config_for("screener"), "writer": config_for("writer")},
        deps=make_deps([text_response("a"), text_response("b")]),
    )

    orchestrator.run_task(envelope)

    assert envelope.agent_chain == ["screener", "writer"]


# ── Delegation ──────────────────────────────────────────────────────────────


def test_delegation_runs_the_child_and_returns_its_output() -> None:
    """Delegation is a re-entrant call, not message passing.

    The earlier bus-plus-busy-wait design deadlocks whenever dispatch is
    single-threaded: the parent occupies the only thread that could run the
    child. The invariant survives because the parent still never sees the
    child object.
    """
    envelope = make_envelope(Subtask(required_capability="screener", id="s1"))
    deps = make_deps(
        [
            tool_response(
                ("delegate", {"capability": "researcher", "input_data": {"q": "pay"}})
            ),
            text_response("salary band is 40-60k"),
            text_response("screening complete"),
        ]
    )
    orchestrator = Orchestrator(
        registry={
            "screener": config_for("screener", can_delegate_to=("researcher",)),
            "researcher": config_for("researcher"),
        },
        deps=deps,
    )

    outcome = orchestrator.run_task(envelope)

    assert outcome.results[0].output == "screening complete"
    assert envelope.budget.spawns_used == 1
    parent_turn = deps.llm.calls[-1]
    tool_msg = next(m for m in parent_turn if m.get("role") == "tool")
    assert "40-60k" in tool_msg["content"]


def test_delegation_charges_the_shared_task_budget() -> None:
    """A per-specialist cap does nothing about a fan-out that collectively
    overspends, so the ceiling lives on the task."""
    envelope = make_envelope(
        Subtask(required_capability="screener", id="s1"),
        budget=TaskBudget(max_total_spawns=0, max_spawn_depth=3),
    )
    deps = make_deps(
        [
            tool_response(
                ("delegate", {"capability": "researcher", "input_data": {}})
            ),
            text_response("carried on alone"),
        ]
    )
    orchestrator = Orchestrator(
        registry={
            "screener": config_for("screener", can_delegate_to=("researcher",)),
            "researcher": config_for("researcher"),
        },
        deps=deps,
    )

    outcome = orchestrator.run_task(envelope)

    # The refusal reaches the model as a tool error; the task still finishes.
    assert outcome.results[0].output == "carried on alone"
    assert envelope.budget.spawns_used == 0
    assert any(
        "budget" in m.get("content", "")
        for call in deps.llm.calls
        for m in call
        if m.get("role") == "tool"
    )


def test_delegation_cycle_is_refused() -> None:
    """A -> B -> A would recurse until the budget is gone."""
    envelope = make_envelope(Subtask(required_capability="a", id="s1"))
    deps = make_deps(
        [
            tool_response(("delegate", {"capability": "b", "input_data": {}})),
            tool_response(("delegate", {"capability": "a", "input_data": {}})),
            text_response("b finished without recursing"),
            text_response("a finished"),
        ]
    )
    orchestrator = Orchestrator(
        registry={
            "a": config_for("a", can_delegate_to=("b",)),
            "b": config_for("b", can_delegate_to=("a",)),
        },
        deps=deps,
    )

    outcome = orchestrator.run_task(envelope)

    assert outcome.results[0].output == "a finished"
    # The inner attempt to re-enter 'a' was refused and reported.
    assert any(
        "cycle" in m.get("content", "")
        for call in deps.llm.calls
        for m in call
        if m.get("role") == "tool"
    )


def test_delegation_depth_is_capped() -> None:
    envelope = make_envelope(
        Subtask(required_capability="a", id="s1"),
        budget=TaskBudget(max_spawn_depth=1),
    )
    deps = make_deps(
        [
            tool_response(("delegate", {"capability": "b", "input_data": {}})),
            tool_response(("delegate", {"capability": "c", "input_data": {}})),
            text_response("b done"),
            text_response("a done"),
        ]
    )
    orchestrator = Orchestrator(
        registry={
            "a": config_for("a", can_delegate_to=("b",)),
            "b": config_for("b", can_delegate_to=("c",)),
            "c": config_for("c"),
        },
        deps=deps,
    )

    outcome = orchestrator.run_task(envelope)

    assert outcome.results[0].output == "a done"
    assert any(
        "depth" in m.get("content", "")
        for call in deps.llm.calls
        for m in call
        if m.get("role") == "tool"
    )


# ── Suspension and resumption ───────────────────────────────────────────────


def test_a_suspended_subtask_parks_its_run_and_blocks_dependents() -> None:
    approver = Subtask(required_capability="sender", id="s1")
    downstream = Subtask(required_capability="writer", id="s2", depends_on=["s1"])
    envelope = make_envelope(approver, downstream)

    deps = make_deps(
        [tool_response(("send_email", {"to": "a@b.c"}))], specs=[write_tool()]
    )
    orchestrator = Orchestrator(
        registry={
            "sender": config_for("sender", tools=("send_email",)),
            "writer": config_for("writer"),
        },
        deps=deps,
    )

    outcome = orchestrator.run_task(envelope)

    assert outcome.suspended
    assert "s1" in outcome.awaiting_approval
    assert approver.status is SubtaskStatus.BLOCKED
    assert downstream.status is SubtaskStatus.SKIPPED
    assert "s1" in envelope.extensions[SUSPENDED_RUNS_KEY]
    assert deps.tools.invocations == []


def test_resume_task_completes_an_approved_subtask() -> None:
    subtask = Subtask(required_capability="sender", id="s1")
    envelope = make_envelope(subtask)

    deps = make_deps(
        [tool_response(("send_email", {"to": "a@b.c"})), text_response("sent")],
        specs=[write_tool()],
        results={"send_email": "message-1"},
    )
    orchestrator = Orchestrator(
        registry={"sender": config_for("sender", tools=("send_email",))}, deps=deps
    )

    orchestrator.run_task(envelope)
    outcome = orchestrator.resume_task(envelope, "s1", approved=True)

    assert outcome.results[0].status is SpecialistStatus.COMPLETED
    assert [i.tool_name for i in deps.tools.invocations] == ["send_email"]
    assert "s1" not in envelope.extensions[SUSPENDED_RUNS_KEY]


def test_resume_task_rejects_an_unknown_subtask() -> None:
    envelope = make_envelope(Subtask(required_capability="sender", id="s1"))
    orchestrator = Orchestrator(
        registry={"sender": config_for("sender")}, deps=make_deps([])
    )

    with pytest.raises(KeyError):
        orchestrator.resume_task(envelope, "nope", approved=True)


def test_unregistered_capability_is_reported_clearly() -> None:
    envelope = make_envelope(Subtask(required_capability="ghost", id="s1"))
    orchestrator = Orchestrator(registry={}, deps=make_deps([]))

    with pytest.raises(Exception) as exc:
        orchestrator.run_task(envelope)

    assert "ghost" in str(exc.value)
