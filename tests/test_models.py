"""Wire-format guarantees: round-trips, forward compatibility, slicing."""

from __future__ import annotations

import json

import pytest

from baton_protocol import (
    PROTOCOL_VERSION,
    BudgetExceeded,
    ContextEnvelope,
    DelegationDepthExceeded,
    GapKind,
    IncompatibleProtocolVersion,
    SpecialistConfig,
    Subtask,
    SubtaskStatus,
    TaskBudget,
    is_compatible,
)


def test_envelope_round_trips_through_json() -> None:
    envelope = ContextEnvelope(
        task_id="t1",
        tenant_id="acme",
        principal_id="u1",
        intent="screen candidates",
        context_data={"job_title": "Engineer"},
        subtasks=[Subtask(required_capability="screener")],
    )
    envelope.flag_gap(GapKind.MISSING_INPUT, "no salary band supplied")

    restored = ContextEnvelope.from_dict(json.loads(json.dumps(envelope.to_dict())))

    assert restored.task_id == envelope.task_id
    assert restored.intent == envelope.intent
    assert restored.context_data == envelope.context_data
    assert restored.subtasks[0].required_capability == "screener"
    assert restored.gap_flags[0].kind is GapKind.MISSING_INPUT


def test_unknown_fields_survive_a_round_trip() -> None:
    """Forward compatibility.

    A v1.0 reader handling a payload from a newer producer must not destroy the
    fields it does not understand — that is what makes a multi-implementation
    protocol safe to upgrade one service at a time.
    """
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "task_id": "t1",
        "tenant_id": "acme",
        "principal_id": "u1",
        "future_field": {"added": "in v1.7"},
    }

    restored = ContextEnvelope.from_dict(payload).to_dict()

    assert restored["future_field"] == {"added": "in v1.7"}


def test_unknown_fields_cannot_shadow_known_ones() -> None:
    """Preserved extras must never override a field we actually parse."""
    envelope = ContextEnvelope(task_id="t1", tenant_id="acme", principal_id="u1")
    envelope.unknown["task_id"] = "spoofed"

    assert envelope.to_dict()["task_id"] == "t1"


def test_incompatible_major_version_is_rejected() -> None:
    with pytest.raises(IncompatibleProtocolVersion):
        ContextEnvelope.from_dict(
            {
                "protocol_version": "99.0.0",
                "task_id": "t1",
                "tenant_id": "acme",
                "principal_id": "u1",
            }
        )


def test_newer_minor_version_is_readable() -> None:
    assert is_compatible("1.9.3")
    assert not is_compatible("2.0.0")


def test_slice_exposes_only_declared_context_keys() -> None:
    """Containment: a specialist cannot see context it was not granted."""
    envelope = ContextEnvelope(
        task_id="t1",
        tenant_id="acme",
        principal_id="u1",
        context_data={
            "job_title": "Engineer",
            "salary_band": "secret",
            "cv_text": "...",
        },
    )
    config = SpecialistConfig(capability="screener", context_keys=("job_title",))

    sliced = envelope.slice_for(config)

    assert sliced.data == {"job_title": "Engineer"}
    assert "salary_band" not in sliced.data


def test_slice_always_carries_identity() -> None:
    """Identity must travel or the specialist cannot meter or scope anything."""
    envelope = ContextEnvelope(task_id="t1", tenant_id="acme", principal_id="u1")
    sliced = envelope.slice_for(SpecialistConfig(capability="screener"))

    assert (sliced.tenant_id, sliced.principal_id) == ("acme", "u1")


def test_record_result_marks_subtask_complete_and_bumps_revision() -> None:
    subtask = Subtask(required_capability="screener")
    envelope = ContextEnvelope(
        task_id="t1", tenant_id="acme", principal_id="u1", subtasks=[subtask]
    )
    before = envelope.revision

    envelope.record_result(subtask.id, {"score": 87})

    assert subtask.status is SubtaskStatus.COMPLETED
    assert envelope.context_data["results"][subtask.id] == {"score": 87}
    assert envelope.revision > before


def test_subtask_requires_a_capability() -> None:
    with pytest.raises(ValueError):
        Subtask.from_dict({"parameters": {}})


def test_config_rejects_nonsense_iteration_limits() -> None:
    with pytest.raises(ValueError):
        SpecialistConfig(capability="x", max_iterations=0)


def test_config_requires_a_capability_name() -> None:
    with pytest.raises(ValueError):
        SpecialistConfig(capability="")


class TestTaskBudget:
    def test_token_ceiling_is_enforced(self) -> None:
        budget = TaskBudget(max_total_tokens=100)
        budget.charge_tokens(90)

        with pytest.raises(BudgetExceeded):
            budget.charge_tokens(20)

    def test_spawn_ceiling_is_enforced(self) -> None:
        budget = TaskBudget(max_total_spawns=2, max_spawn_depth=5)
        budget.charge_spawn(1)
        budget.charge_spawn(1)

        with pytest.raises(BudgetExceeded):
            budget.charge_spawn(1)

    def test_depth_is_checked_before_spend_is_recorded(self) -> None:
        """A rejected delegation must not consume budget."""
        budget = TaskBudget(max_spawn_depth=2)

        with pytest.raises(DelegationDepthExceeded):
            budget.charge_spawn(3)

        assert budget.spawns_used == 0

    def test_a_refused_spawn_does_not_consume_budget(self) -> None:
        """Spawn refusal is recoverable, so a caller that retries differently
        must not have been charged for the attempt that was rejected."""
        budget = TaskBudget(max_total_spawns=1)
        budget.charge_spawn(1)

        with pytest.raises(BudgetExceeded):
            budget.charge_spawn(1)

        assert budget.spawns_used == 1

    def test_would_exceed_allows_checking_before_spending(self) -> None:
        budget = TaskBudget(max_total_tokens=100)
        budget.charge_tokens(80)

        assert budget.would_exceed_tokens(30)
        assert not budget.would_exceed_tokens(10)

    def test_round_trips(self) -> None:
        budget = TaskBudget(max_total_tokens=999, tokens_used=42)
        assert TaskBudget.from_dict(budget.to_dict()).tokens_used == 42
<<<<<<< HEAD
=======


class TestUnlimitedTokens:
    """Zero means no token ceiling; other ceilings still apply."""

    def test_a_zero_task_ceiling_never_trips(self):
        budget = TaskBudget(max_total_tokens=0)
        budget.charge_tokens(10_000_000)
        assert not budget.would_exceed_tokens(10**9)
        assert budget.tokens_used == 10_000_000

    def test_a_positive_ceiling_still_trips(self):
        budget = TaskBudget(max_total_tokens=10)
        with pytest.raises(BudgetExceeded):
            budget.charge_tokens(11)

    def test_unlimited_tokens_leave_tool_calls_bounded(self):
        budget = TaskBudget(max_total_tokens=0, max_tool_calls=1)
        budget.charge_tool_call()
        with pytest.raises(BudgetExceeded):
            budget.charge_tool_call()
>>>>>>> 4d653be (BATON 1.0: specification, reference library, JSON Schemas and conformance suite)
