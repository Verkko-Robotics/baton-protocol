"""Runs the language-neutral conformance vectors against the Python reference.

The vectors under ``conformance/vectors/`` are plain JSON. They are the
executable half of the specification: another implementation, in any language,
proves it conforms by running the same vectors through its own runner. This file
is the Python runner, and doubles as documentation of how a vector is read.

Each suite file declares one ``operation``; each operation has one handler below.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import baton_protocol as baton
from baton_protocol import errors as baton_errors
from baton_protocol.testing import (
    RecordingApprovalGateway,
    RecordingMeter,
    ScriptedToolGateway,
    text_response,
    tool_response,
)

VECTOR_DIR = Path(__file__).resolve().parent.parent / "conformance" / "vectors"


def load_all() -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """``(suite_operation, vector, suite_doc)`` for every vector on disk."""
    out = []
    for path in sorted(VECTOR_DIR.glob("*.json")):
        suite = json.loads(path.read_text(encoding="utf-8"))
        for vector in suite["vectors"]:
            out.append((suite["operation"], vector, suite))
    return out


def run(operation: str, vector: dict[str, Any]) -> None:
    HANDLERS[operation](vector)


def _exc(name: str) -> type[BaseException]:
    if name == "ValueError":
        return ValueError
    return getattr(baton_errors, name)


# ── slice ───────────────────────────────────────────────────────────────────


def _slice(v: dict[str, Any]) -> None:
    envelope = baton.ContextEnvelope.from_dict(v["envelope"])
    config = baton.SpecialistConfig(
        capability=v["config"]["capability"],
        context_keys=tuple(v["config"].get("context_keys", ())),
    )
    sliced = envelope.slice_for(
        config, depth=v.get("depth", 0), ancestry=tuple(v.get("ancestry", ()))
    ).to_dict()
    expect = v["expect"]

    if "data" in expect:
        assert sliced["data"] == expect["data"]
    for key, value in expect.get("identity", {}).items():
        assert sliced[key] == value, key
    if "depth" in expect:
        assert sliced["depth"] == expect["depth"]
    if "ancestry" in expect:
        assert sliced["ancestry"] == expect["ancestry"]
    wire = json.dumps(sliced)
    for forbidden in expect.get("never_present", ()):
        assert forbidden not in wire, f"{forbidden!r} leaked into the slice"


# ── blackboard ──────────────────────────────────────────────────────────────


def _blackboard(v: dict[str, Any]) -> None:
    board = baton.Blackboard(v["task_id"], v["tenant_id"])

    def post(step: dict[str, Any]):
        return board.post(
            specialist_id=f"sp-{step['capability']}",
            capability=step["capability"],
            key=step["key"],
            value=step["value"],
            tenant_id=step.get("tenant_id"),
            visible_to=step.get("visible_to"),
        )

    for step in v["steps"]:
        op = step["op"]
        if op == "post":
            post(step)
        elif op == "post_raises":
            try:
                post(step)
            except _exc(step["error"]):
                pass
            else:
                raise AssertionError(f"expected {step['error']}")
        elif op == "read":
            got = board.read(
                capability=step["capability"],
                key=step["key"],
                requesting_capability=step["requester"],
            )
            assert got == step["expect"], step
        elif op == "read_all":
            assert board.read_all(step["requester"]) == step["expect"], step
        elif op == "history_count":
            hist = board.history(
                step["requester"], capability=step.get("capability"), key=step.get("key")
            )
            assert len(hist) == step["expect"], step
        elif op == "seq_strictly_increasing":
            seqs = [e.seq for e in board.snapshot()]
            assert seqs == sorted(set(seqs)) and len(seqs) == len(set(seqs)), seqs
        else:
            raise AssertionError(f"unknown blackboard op {op!r}")


# ── roundtrip ───────────────────────────────────────────────────────────────

_MODELS = {
    "envelope": baton.ContextEnvelope,
    "subtask": baton.Subtask,
    "gap_flag": baton.GapFlag,
    "blackboard_entry": baton.BlackboardEntry,
    "slice": baton.ContextSlice,
    "pending_approval": baton.PendingApproval,
    "budget": baton.TaskBudget,
    "run": baton.SpecialistRun,
}


def _pointer(doc: Any, pointer: str) -> Any:
    node = doc
    for part in pointer.strip("/").split("/"):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def _roundtrip(v: dict[str, Any]) -> None:
    model = _MODELS[v["type"]]
    wire_in = json.loads(json.dumps(v["document"]))
    wire_out = json.loads(json.dumps(model.from_dict(wire_in).to_dict()))
    for pointer in v["expect_preserved"]:
        assert _pointer(wire_out, pointer) == _pointer(wire_in, pointer), pointer


# ── version ─────────────────────────────────────────────────────────────────


def _version(v: dict[str, Any]) -> None:
    from baton_protocol.version import is_compatible

    if "envelope_version" in v:
        doc = {"task_id": "t", "tenant_id": "a", "principal_id": "u",
               "protocol_version": v["envelope_version"]}
        try:
            baton.ContextEnvelope.from_dict(doc)
            read = True
        except baton_errors.IncompatibleProtocolVersion:
            read = False
        assert read is v["reads"]
        return
    if "error" in v:
        try:
            is_compatible(v["version"])
        except _exc(v["error"]):
            return
        raise AssertionError(f"expected {v['error']}")
    assert is_compatible(v["version"]) is v["compatible"]


# ── ordering ────────────────────────────────────────────────────────────────


def _ordering(v: dict[str, Any]) -> None:
    subtasks = [
        baton.Subtask(
            required_capability=s["required_capability"],
            id=s["id"],
            depends_on=list(s.get("depends_on", ())),
        )
        for s in v["subtasks"]
    ]
    if "expect_error" in v:
        try:
            baton.topological_order(subtasks)
        except _exc(v["expect_error"]):
            return
        raise AssertionError(f"expected {v['expect_error']}")
    assert [s.id for s in baton.topological_order(subtasks)] == v["expect_order"]


# ── shared builders for delegation and flow ─────────────────────────────────


class _VectorLLM(baton.LLMGateway):
    """Replays the vector's scripted model turns."""

    def __init__(self, script: list[dict[str, Any]]) -> None:
        self._turns = iter(script)

    def complete(self, *, messages, tools, model, temperature, max_tokens=None):
        try:
            turn = next(self._turns)
        except StopIteration as exc:
            raise AssertionError("the vector scripted fewer model turns than were used") from exc
        if "error" in turn:
            raise RuntimeError(turn["error"])
        if "text" in turn:
            return text_response(turn["text"])
        return tool_response(*[(name, args) for name, args in turn["tool_calls"]])


def _registry(spec: dict[str, dict[str, Any]]) -> dict[str, baton.SpecialistConfig]:
    out = {}
    for capability, cfg in spec.items():
        visible = cfg.get("output_visible_to")
        out[capability] = baton.SpecialistConfig(
            capability=capability,
            model="test-model",
            tools=tuple(cfg.get("tools", ())),
            can_delegate_to=tuple(cfg.get("can_delegate_to", ())),
            output_visible_to=tuple(visible) if visible is not None else None,
            output_schema=cfg.get("output_schema"),
        )
    return out


def _deps(llm: baton.LLMGateway, specs, results) -> tuple[baton.LoopDependencies, ScriptedToolGateway]:
    tools = ScriptedToolGateway(
        [
            baton.ToolSpec(
                name=s["name"],
                description="vector tool",
                parameters={"type": "object", "properties": {}},
                side_effect=bool(s["side_effect"]),
            )
            for s in specs
        ],
        results,
    )
    deps = baton.LoopDependencies(
        llm=llm,
        tools=tools,
        approvals=RecordingApprovalGateway(),
        meter=RecordingMeter(),
    )
    return deps, tools


# ── delegation ──────────────────────────────────────────────────────────────


def _delegation(v: dict[str, Any]) -> None:
    deps, _ = _deps(_VectorLLM([]), [], {})
    orchestrator = baton.Orchestrator(registry=_registry(v["registry"]), deps=deps)
    envelope = baton.ContextEnvelope(
        task_id="t1",
        tenant_id="acme",
        principal_id="u1",
        budget=baton.TaskBudget(**v.get("budget", {})),
    )
    board = baton.Blackboard("t1", "acme")
    req = v["request"]
    request = baton.DelegationRequest(
        requesting_specialist_id=req["requesting_capability"],
        requesting_capability=req["requesting_capability"],
        required_capability=req["required_capability"],
        depth=req["depth"],
        ancestry=tuple(req["ancestry"]),
    )
    try:
        orchestrator.run_child(envelope, board, request)
    except _exc(v["expect_error"]):
        pass
    else:
        raise AssertionError(f"expected {v['expect_error']}")
    assert envelope.budget.spawns_used == v["expect_spawns_used"]


# ── flow ────────────────────────────────────────────────────────────────────


def _flow(v: dict[str, Any]) -> None:
    deps, tools = _deps(_VectorLLM(v["llm"]), v.get("tool_specs", []), v.get("tool_results", {}))
    if v.get("validate_output"):
        try:
            from baton_protocol.validation import JsonSchemaValidator

            deps.validator = JsonSchemaValidator()
        except RuntimeError:
            import pytest

            pytest.skip("output validation vectors need the 'validation' extra (jsonschema)")
    orchestrator = baton.Orchestrator(registry=_registry(v["registry"]), deps=deps)
    envelope = baton.ContextEnvelope(
        task_id="t1",
        tenant_id="acme",
        principal_id="u1",
        subtasks=[
            baton.Subtask(
                required_capability=s["required_capability"],
                id=s["id"],
                depends_on=list(s.get("depends_on", ())),
                consumes=list(s.get("consumes", ())),
            )
            for s in v["subtasks"]
        ],
    )

    outcome = orchestrator.run_task(envelope)
    if "expect_first" in v:
        assert outcome.suspended is v["expect_first"]["suspended"]

    after = v.get("after_run")
    if after:
        if after.get("serialize"):
            # Invariant: run state survives a process restart.
            envelope = baton.ContextEnvelope.from_dict(
                json.loads(json.dumps(envelope.to_dict()))
            )
        resume = after.get("resume")
        if resume:
            outcome = orchestrator.resume_task(
                envelope,
                resume["subtask_id"],
                approved=resume["approved"],
                denial_reason=resume.get("denial_reason"),
            )

    expect = v["expect"]
    assert outcome.suspended is expect["suspended"]
    assert [i.tool_name for i in tools.invocations] == expect["invocations"]
    statuses = {s.id: s.status.value for s in envelope.subtasks}
    for sid, status in expect["subtask_status"].items():
        assert statuses[sid] == status, (sid, statuses)
    if "gap_kinds" in expect:
        assert sorted({g.kind.value for g in envelope.gap_flags}) == sorted(expect["gap_kinds"])
    if "metering" in expect:
        records = deps.meter.records
        want = expect["metering"]
        assert len(records) == want["calls"]
        keys = [r["idempotency_key"] for r in records]
        assert all(keys), "every metered call must carry an idempotency key"
        assert len(set(keys)) == len(keys), "idempotency keys must be unique per call"
        assert {r["principal_id"] for r in records} == {want["principal"]}
    if "blackboard" in expect:
        board = baton.Blackboard(
            envelope.task_id, envelope.tenant_id, entries=envelope.blackboard_snapshot
        )
        for check in expect["blackboard"]:
            got = board.read(
                capability=check["capability"],
                key=check["key"],
                requesting_capability=check["requester"],
            )
            assert got == check["value"], check
    for check in expect.get("provenance", ()):
        board = baton.Blackboard(
            envelope.task_id, envelope.tenant_id, entries=envelope.blackboard_snapshot
        )
        entry = board.read_address(check["address"], check["requester"])
        assert entry is not None, check
        assert list(entry.derived_from) == check["derived_from"], check


HANDLERS = {
    "slice": _slice,
    "blackboard": _blackboard,
    "roundtrip": _roundtrip,
    "version": _version,
    "ordering": _ordering,
    "delegation": _delegation,
    "flow": _flow,
}
