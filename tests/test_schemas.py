"""The normative JSON Schemas must describe what the reference library emits.

The schemas are the standard; the Python models are one implementation of it.
These tests keep the two from drifting: every wire type the library serializes
is validated against its schema, and documents that are malformed on purpose
must be rejected.

Skipped when the optional ``jsonschema`` dependency is not installed.
"""

from __future__ import annotations

import dataclasses
import json
from importlib import resources

import pytest

jsonschema = pytest.importorskip("jsonschema")
referencing = pytest.importorskip("referencing")

from referencing import Registry, Resource  # noqa: E402
from referencing.jsonschema import DRAFT202012  # noqa: E402

from baton_protocol import (  # noqa: E402
    Blackboard,
    ContextEnvelope,
    DelegationRequest,
    GapKind,
    PendingApproval,
    SpecialistResult,
    SpecialistRun,
    SpecialistStatus,
    Subtask,
    TaskBudget,
)

SCHEMA_NAMES = [
    "budget",
    "subtask",
    "gap-flag",
    "blackboard-entry",
    "slice",
    "pending-approval",
    "specialist-run",
    "specialist-result",
    "delegation-request",
    "envelope",
]


def _load(name: str) -> dict:
    path = resources.files("baton_protocol") / "schemas" / f"{name}.schema.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def registry() -> Registry:
    reg: Registry = Registry()
    for name in SCHEMA_NAMES:
        doc = _load(name)
        reg = reg.with_resource(doc["$id"], Resource(doc, DRAFT202012))
    return reg


def validate(name: str, instance, registry: Registry) -> list[str]:
    validator = jsonschema.Draft202012Validator(_load(name), registry=registry)
    return [e.message for e in validator.iter_errors(instance)]


def assert_valid(name: str, instance, registry: Registry) -> None:
    errors = validate(name, instance, registry)
    assert not errors, f"{name} schema rejected reference output: {errors}"


def assert_invalid(name: str, instance, registry: Registry) -> None:
    assert validate(name, instance, registry), f"{name} schema accepted a bad document"


# ── The schemas themselves ──────────────────────────────────────────────────


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_every_schema_is_a_valid_draft_2020_12_schema(name: str) -> None:
    jsonschema.Draft202012Validator.check_schema(_load(name))


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_schema_ids_are_unique_and_versioned(name: str) -> None:
    doc = _load(name)
    assert doc["$id"].endswith(f"/1.0/{name}.schema.json")


def test_schemas_do_not_forbid_unknown_fields() -> None:
    """Forward compatibility: a v1.0 reader must accept a v1.x document."""
    for name in SCHEMA_NAMES:
        assert _load(name).get("additionalProperties", True) is not False, name


# ── Reference output validates ──────────────────────────────────────────────


def _full_envelope() -> ContextEnvelope:
    envelope = ContextEnvelope(
        task_id="t1",
        tenant_id="acme",
        principal_id="u1",
        intent="screen the shortlist",
        session_id="s1",
        context_data={"job_title": "Engineer", "required_skills": ["python"]},
        subtasks=[
            Subtask(required_capability="screener", id="a", parameters={"cv": "..."}),
            Subtask(required_capability="writer", id="b", depends_on=["a"]),
        ],
        extensions={"host.note": {"anything": True}},
    )
    envelope.flag_gap(GapKind.MISSING_INPUT, "no CV attached", subtask_id="a")
    board = Blackboard("t1", "acme")
    board.post(specialist_id="sp1", capability="screener", key="output", value={"score": 7})
    board.post(
        specialist_id="sp1",
        capability="screener",
        key="private",
        value="x",
        visible_to=[],
    )
    envelope.blackboard_snapshot = board.snapshot()
    envelope.agent_chain.append("screener")
    return envelope


def test_a_full_envelope_validates(registry: Registry) -> None:
    assert_valid("envelope", _full_envelope().to_dict(), registry)


def test_an_envelope_survives_a_json_round_trip_and_still_validates(registry) -> None:
    wire = json.loads(json.dumps(_full_envelope().to_dict()))
    assert_valid("envelope", wire, registry)
    assert_valid("envelope", ContextEnvelope.from_dict(wire).to_dict(), registry)


def test_each_nested_type_validates_on_its_own(registry: Registry) -> None:
    envelope = _full_envelope().to_dict()
    assert_valid("budget", envelope["budget"], registry)
    for sub in envelope["subtasks"]:
        assert_valid("subtask", sub, registry)
    for gap in envelope["gap_flags"]:
        assert_valid("gap-flag", gap, registry)
    for entry in envelope["blackboard_snapshot"]:
        assert_valid("blackboard-entry", entry, registry)


def test_a_slice_validates(registry: Registry) -> None:
    from baton_protocol import SpecialistConfig

    config = SpecialistConfig(capability="screener", context_keys=("job_title",))
    sliced = _full_envelope().slice_for(config, depth=1, ancestry=("root",))
    assert_valid("slice", sliced.to_dict(), registry)


def test_a_suspended_run_validates(registry: Registry) -> None:
    run = SpecialistRun(
        specialist_id="sp1",
        capability="sender",
        subtask_id="a",
        task_id="t1",
        messages=[{"role": "system", "content": "hi"}],
        status=SpecialistStatus.AWAITING_APPROVAL,
        pending_approval=PendingApproval(
            tool_name="send_email",
            arguments={"to": "a@b.c"},
            call_id="call-1",
            iteration=1,
            idempotency_key="k",
        ),
    )
    assert_valid("specialist-run", run.to_dict(), registry)
    assert_valid("pending-approval", run.pending_approval.to_dict(), registry)


def test_a_result_validates(registry: Registry) -> None:
    result = SpecialistResult(
        specialist_id="sp1",
        subtask_id="a",
        capability="screener",
        status=SpecialistStatus.COMPLETED,
        output={"score": 7},
        confidence=1.0,
        tokens_used=20,
        duration_ms=5,
        metadata={"iterations": 1},
    )
    assert_valid("specialist-result", result.to_dict(), registry)


def test_a_delegation_request_validates(registry: Registry) -> None:
    request = DelegationRequest(
        requesting_specialist_id="screener",
        requesting_capability="screener",
        required_capability="researcher",
        input_data={"query": "q"},
        depth=1,
        ancestry=("screener",),
    )
    wire = json.loads(json.dumps(dataclasses.asdict(request)))
    assert_valid("delegation-request", wire, registry)


def test_documents_with_extra_fields_still_validate(registry: Registry) -> None:
    doc = _full_envelope().to_dict()
    doc["x_future_field"] = {"added_in": "1.4"}
    doc["subtasks"][0]["x_priority"] = 3
    assert_valid("envelope", doc, registry)


# ── Malformed documents are rejected ────────────────────────────────────────


def test_an_envelope_missing_its_identity_is_rejected(registry: Registry) -> None:
    for field in ("task_id", "tenant_id", "principal_id"):
        doc = _full_envelope().to_dict()
        del doc[field]
        assert_invalid("envelope", doc, registry)


def test_a_malformed_protocol_version_is_rejected(registry: Registry) -> None:
    for bad in ("1", "1.0", "v1.0.0", "1.0.0-beta", ""):
        doc = _full_envelope().to_dict()
        doc["protocol_version"] = bad
        assert_invalid("envelope", doc, registry)


def test_unknown_enum_values_are_rejected(registry: Registry) -> None:
    assert_invalid("subtask", {"required_capability": "a", "status": "paused"}, registry)
    assert_invalid("gap-flag", {"kind": "bored"}, registry)
    assert_invalid(
        "specialist-result",
        {"specialist_id": "s", "subtask_id": "a", "capability": "c", "status": "meh"},
        registry,
    )


def test_a_subtask_needs_a_capability(registry: Registry) -> None:
    assert_invalid("subtask", {"id": "a"}, registry)
    assert_invalid("subtask", {"required_capability": ""}, registry)


def test_a_blackboard_entry_needs_an_address(registry: Registry) -> None:
    assert_invalid("blackboard-entry", {"capability": "a"}, registry)
    assert_invalid("blackboard-entry", {"capability": "a", "key": ""}, registry)


def test_negative_counters_and_revisions_are_rejected(registry: Registry) -> None:
    assert_invalid("budget", {"tokens_used": -1}, registry)
    doc = _full_envelope().to_dict()
    doc["revision"] = 0
    assert_invalid("envelope", doc, registry)


def test_nested_documents_are_checked_through_the_envelope(registry: Registry) -> None:
    doc = _full_envelope().to_dict()
    doc["subtasks"][0]["status"] = "paused"
    assert_invalid("envelope", doc, registry)
    doc = _full_envelope().to_dict()
    del doc["blackboard_snapshot"][0]["key"]
    assert_invalid("envelope", doc, registry)
