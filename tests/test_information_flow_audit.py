"""The audit must answer "who can learn what, and through which channel"."""

from __future__ import annotations

from baton_protocol import Blackboard, audit_information_flow
from baton_protocol.audit import CHANNEL_DELEGATION, CHANNEL_ENVELOPE

from conftest import make_config, make_envelope


def _registry(*configs):
    return {c.capability: c for c in configs}


def test_envelope_keys_follow_context_keys() -> None:
    envelope = make_envelope(context_data={"salary": 1, "notes": 2})
    registry = _registry(
        make_config("hr", context_keys=("salary",)),
        make_config("scribe", context_keys=("notes",)),
    )

    report = audit_information_flow(envelope, registry)

    assert report["hr"].envelope_keys == {"salary"}
    assert report["scribe"].envelope_keys == {"notes"}
    assert report.readers_of_envelope_key("salary") == {"hr": {CHANNEL_ENVELOPE}}


def test_declared_key_missing_from_envelope_is_surfaced() -> None:
    envelope = make_envelope(context_data={"notes": 1})
    registry = _registry(make_config("scribe", context_keys=("notes", "typo_key")))

    flow = audit_information_flow(envelope, registry)["scribe"]

    assert flow.envelope_keys == {"notes"}
    assert flow.envelope_keys_missing == {"typo_key"}


def test_unlisted_capability_cannot_read_envelope_key() -> None:
    envelope = make_envelope(context_data={"salary": 1})
    registry = _registry(make_config("hr", context_keys=("salary",)), make_config("other"))

    report = audit_information_flow(envelope, registry)

    assert "other" not in report.readers_of_envelope_key("salary")


def test_output_visibility_none_means_everyone_can_read() -> None:
    registry = _registry(
        make_config("producer", output_visible_to=None),
        make_config("reader"),
    )

    report = audit_information_flow(make_envelope(), registry)

    assert "producer.output" in report["reader"].blackboard_addresses


def test_output_visibility_empty_means_nobody_else() -> None:
    registry = _registry(
        make_config("producer", output_visible_to=()),
        make_config("reader"),
    )

    report = audit_information_flow(make_envelope(), registry)

    assert "producer.output" not in report["reader"].blackboard_addresses
    # The producer can still read its own output.
    assert "producer.output" in report["producer"].blackboard_addresses


def test_output_visibility_is_per_reader() -> None:
    registry = _registry(
        make_config("producer", output_visible_to=("friend",)),
        make_config("friend"),
        make_config("stranger"),
    )

    report = audit_information_flow(make_envelope(), registry)

    assert report.readers_of_address("producer.output") == {"producer", "friend"}


def test_wildcard_output_visibility_reaches_everyone() -> None:
    registry = _registry(
        make_config("producer", output_visible_to=("*",)),
        make_config("a"),
        make_config("b"),
    )

    report = audit_information_flow(make_envelope(), registry)

    assert report.readers_of_address("producer.output") == {"producer", "a", "b"}


def test_existing_blackboard_entries_respect_their_own_visible_to() -> None:
    board = Blackboard("task-1", "tenant-a")
    board.post(
        specialist_id="s",
        capability="producer",
        key="secret",
        value=1,
        visible_to=["friend"],
    )
    envelope = make_envelope(blackboard_snapshot=board.snapshot())
    registry = _registry(
        make_config("producer", output_visible_to=()),
        make_config("friend"),
        make_config("stranger"),
    )

    report = audit_information_flow(envelope, registry)

    assert "producer.secret" in report["friend"].blackboard_addresses
    assert "producer.secret" not in report["stranger"].blackboard_addresses


def test_delegation_confers_the_delegates_envelope_keys() -> None:
    """The easy-to-miss channel: delegating to HR lets you act on HR's data."""
    envelope = make_envelope(context_data={"salary": 1, "notes": 2})
    registry = _registry(
        make_config("hr", context_keys=("salary",)),
        make_config("manager", context_keys=("notes",), can_delegate_to=("hr",)),
    )

    flow = audit_information_flow(envelope, registry)["manager"]

    assert flow.envelope_keys == {"notes"}
    assert dict(flow.delegated_envelope_keys) == {"salary": {"hr"}}
    assert flow.channels_for_envelope_key("salary") == {CHANNEL_DELEGATION}
    assert audit_information_flow(envelope, registry).readers_of_envelope_key(
        "salary"
    ) == {"hr": {CHANNEL_ENVELOPE}, "manager": {CHANNEL_DELEGATION}}


def test_delegation_reach_is_transitive() -> None:
    envelope = make_envelope(context_data={"salary": 1})
    registry = _registry(
        make_config("hr", context_keys=("salary",)),
        make_config("middle", can_delegate_to=("hr",)),
        make_config("top", can_delegate_to=("middle",)),
    )

    flow = audit_information_flow(envelope, registry)["top"]

    assert flow.delegates == {"middle", "hr"}
    assert dict(flow.delegated_envelope_keys) == {"salary": {"hr"}}


def test_delegation_cycle_terminates() -> None:
    envelope = make_envelope(context_data={"k": 1})
    registry = _registry(
        make_config("a", context_keys=("k",), can_delegate_to=("b",)),
        make_config("b", can_delegate_to=("a",)),
    )

    report = audit_information_flow(envelope, registry)

    assert report["a"].delegates == {"b"}
    assert report["b"].delegates == {"a"}
    assert dict(report["b"].delegated_envelope_keys) == {"k": {"a"}}


def test_unregistered_delegate_confers_nothing() -> None:
    envelope = make_envelope(context_data={"k": 1})
    registry = _registry(make_config("a", can_delegate_to=("ghost",)))

    flow = audit_information_flow(envelope, registry)["a"]

    assert flow.delegates == frozenset()
    assert dict(flow.delegated_envelope_keys) == {}


def test_key_already_held_is_not_reported_as_delegated() -> None:
    envelope = make_envelope(context_data={"k": 1})
    registry = _registry(
        make_config("a", context_keys=("k",), can_delegate_to=("b",)),
        make_config("b", context_keys=("k",)),
    )

    flow = audit_information_flow(envelope, registry)["a"]

    assert dict(flow.delegated_envelope_keys) == {}


def test_report_is_json_serializable() -> None:
    import json

    envelope = make_envelope(context_data={"salary": 1})
    registry = _registry(
        make_config("hr", context_keys=("salary",)),
        make_config("manager", can_delegate_to=("hr",)),
    )

    report = audit_information_flow(envelope, registry)

    assert json.loads(json.dumps(report.to_dict()))["manager"]["delegated_envelope_keys"] == {
        "salary": ["hr"]
    }


def test_audit_does_not_mutate_inputs() -> None:
    envelope = make_envelope(context_data={"k": 1})
    registry = _registry(make_config("a", context_keys=("k",)))
    before = envelope.to_dict()

    audit_information_flow(envelope, registry)

    assert envelope.to_dict() == before
