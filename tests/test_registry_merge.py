"""Registry merging must never let one capability silently replace another."""

from __future__ import annotations

import pytest

from baton_protocol import CapabilityCollision, BatonError, merge_registries

from conftest import make_config


def test_disjoint_registries_merge() -> None:
    a = {"researcher": make_config("researcher")}
    b = {"writer": make_config("writer")}

    merged = merge_registries(a, b)

    assert set(merged) == {"researcher", "writer"}


def test_identical_duplicate_is_allowed() -> None:
    """A resumed run rebuilds a built-in it already had; that is not a conflict."""
    a = {"researcher": make_config("researcher")}
    b = {"researcher": make_config("researcher")}

    assert merge_registries(a, b)["researcher"] == a["researcher"]


def test_differing_configs_for_one_name_raise() -> None:
    builtin = {"researcher": make_config("researcher", tools=("web_search",))}
    impostor = {"researcher": make_config("researcher", tools=("*",))}

    with pytest.raises(CapabilityCollision) as exc:
        merge_registries(builtin, impostor)

    assert exc.value.capabilities == ["researcher"]


def test_every_conflicting_name_is_reported_at_once() -> None:
    a = {
        "x": make_config("x", tools=("t1",)),
        "y": make_config("y", tools=("t1",)),
        "ok": make_config("ok"),
    }
    b = {
        "x": make_config("x", tools=("t2",)),
        "y": make_config("y", tools=("t2",)),
        "ok": make_config("ok"),
    }

    with pytest.raises(CapabilityCollision) as exc:
        merge_registries(a, b)

    assert exc.value.capabilities == ["x", "y"]


def test_collision_is_a_protocol_error() -> None:
    assert issubclass(CapabilityCollision, BatonError)


def test_inputs_are_not_mutated() -> None:
    a = {"researcher": make_config("researcher")}
    b = {"writer": make_config("writer")}

    merge_registries(a, b)

    assert set(a) == {"researcher"}
    assert set(b) == {"writer"}


def test_three_way_merge_detects_a_late_conflict() -> None:
    a = {"x": make_config("x")}
    b = {"y": make_config("y")}
    c = {"x": make_config("x", tools=("other",))}

    with pytest.raises(CapabilityCollision):
        merge_registries(a, b, c)
