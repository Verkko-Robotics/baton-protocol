"""Executes every conformance vector against the Python reference implementation.

A vector tagged ``known_gap`` documents a place where the reference does not yet
meet the specification. It is marked as a *strict* expected failure: the suite
stays green today, and turns red the moment the gap is fixed, which is the
prompt to delete the tag.
"""

from __future__ import annotations

import pytest

from conformance_runner import load_all, run

_CASES = []
for _operation, _vector, _suite in load_all():
    _marks = []
    if "known_gap" in _vector:
        _marks.append(pytest.mark.xfail(strict=True, reason=_vector["known_gap"]))
    _CASES.append(
        pytest.param(
            _operation, _vector, id=f"{_suite['suite']}::{_vector['id']}", marks=_marks
        )
    )


@pytest.mark.parametrize("operation,vector", _CASES)
def test_conformance(operation: str, vector: dict) -> None:
    run(operation, vector)


def test_every_vector_has_an_id_and_they_are_unique() -> None:
    ids = [(s["suite"], v["id"]) for _op, v, s in load_all()]
    assert ids and len(ids) == len(set(ids))


def test_every_suite_operation_has_a_handler() -> None:
    from conformance_runner import HANDLERS

    assert {op for op, _v, _s in load_all()} <= set(HANDLERS)
