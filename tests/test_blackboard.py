"""Blackboard semantics: visibility, ordering, tenancy, durability."""

from __future__ import annotations

import pytest

from baton_protocol import Blackboard, TenantMismatch

TENANT = "tenant-a"
TASK = "task-1"


def test_read_returns_latest_post_for_an_address(blackboard: Blackboard) -> None:
    for score in (10, 20, 30):
        blackboard.post(
            specialist_id="s1", capability="screener", key="score", value=score
        )

    assert (
        blackboard.read(
            capability="screener", key="score", requesting_capability="writer"
        )
        == 30
    )


def test_history_preserves_superseded_values(blackboard: Blackboard) -> None:
    """Append-only means a correction never erases what it replaced."""
    blackboard.post(specialist_id="s1", capability="screener", key="score", value=10)
    blackboard.post(specialist_id="s1", capability="screener", key="score", value=30)

    history = blackboard.history("writer", capability="screener", key="score")
    assert [e.value for e in history] == [10, 30]


def test_empty_visible_to_means_nobody(blackboard: Blackboard) -> None:
    """The fail-closed case.

    ``visible_to=[]`` must not be coerced into "visible to all" by a falsy
    default, which is how an access check silently inverts.
    """
    blackboard.post(
        specialist_id="s1",
        capability="screener",
        key="secret",
        value="classified",
        visible_to=[],
    )

    assert (
        blackboard.read(
            capability="screener", key="secret", requesting_capability="writer"
        )
        is None
    )


def test_none_visible_to_means_everybody(blackboard: Blackboard) -> None:
    blackboard.post(
        specialist_id="s1", capability="screener", key="open", value="public"
    )

    assert (
        blackboard.read(
            capability="screener", key="open", requesting_capability="anyone"
        )
        == "public"
    )


def test_explicit_grant_is_honoured(blackboard: Blackboard) -> None:
    blackboard.post(
        specialist_id="s1",
        capability="screener",
        key="shortlist",
        value=["ada"],
        visible_to=["writer"],
    )

    assert blackboard.read(
        capability="screener", key="shortlist", requesting_capability="writer"
    ) == ["ada"]
    assert (
        blackboard.read(
            capability="screener", key="shortlist", requesting_capability="auditor"
        )
        is None
    )


def test_a_capability_can_always_read_its_own_posts(blackboard: Blackboard) -> None:
    blackboard.post(
        specialist_id="s1",
        capability="screener",
        key="scratch",
        value=1,
        visible_to=[],
    )

    assert (
        blackboard.read(
            capability="screener", key="scratch", requesting_capability="screener"
        )
        == 1
    )


def test_read_all_filters_by_visibility(blackboard: Blackboard) -> None:
    blackboard.post(specialist_id="s1", capability="a", key="open", value=1)
    blackboard.post(
        specialist_id="s1", capability="a", key="closed", value=2, visible_to=["other"]
    )

    assert blackboard.read_all("writer") == {"a.open": 1}


def test_cross_tenant_post_raises(blackboard: Blackboard) -> None:
    """Raise rather than filter: this can only be a host wiring bug."""
    with pytest.raises(TenantMismatch):
        blackboard.post(
            specialist_id="s1",
            capability="screener",
            key="k",
            value="v",
            tenant_id="tenant-b",
        )


def test_sequence_numbers_are_monotonic(blackboard: Blackboard) -> None:
    entries = [
        blackboard.post(
            specialist_id="s1", capability="a", key=f"k{i}", value=i
        )
        for i in range(5)
    ]
    assert [e.seq for e in entries] == [1, 2, 3, 4, 5]


def test_rehydration_continues_the_sequence(blackboard: Blackboard) -> None:
    """A resumed task must not reuse sequence numbers, or ordering breaks."""
    for i in range(3):
        blackboard.post(specialist_id="s1", capability="a", key=f"k{i}", value=i)

    revived = Blackboard(
        task_id=TASK, tenant_id=TENANT, entries=blackboard.snapshot()
    )
    assert revived.post(specialist_id="s2", capability="b", key="k", value=1).seq == 4


def test_on_append_hook_fires_for_durability(blackboard: Blackboard) -> None:
    seen = []
    board = Blackboard(
        task_id=TASK, tenant_id=TENANT, on_append=seen.append
    )
    board.post(specialist_id="s1", capability="a", key="k", value="v")
    assert len(seen) == 1
    assert seen[0].key == "k"


def test_empty_key_is_rejected(blackboard: Blackboard) -> None:
    with pytest.raises(ValueError):
        blackboard.post(specialist_id="s1", capability="a", key="", value=1)
