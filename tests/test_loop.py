"""Loop behaviour: message shape, authorization, approval suspension, limits."""

from __future__ import annotations

import json

from baton_protocol import (
    AgenticLoop,
    Blackboard,
    SpecialistRun,
    SpecialistStatus,
    TaskBudget,
    ToolSpec,
)
from baton_protocol.testing import (
    RecordingApprovalGateway,
    RecordingMeter,
    text_response,
    tool_response,
)

from conftest import (
    PRINCIPAL,
    make_config,
    make_deps,
    make_slice,
    read_tool,
    write_tool,
)
from baton_protocol.models import Subtask


def build_loop(
    responses,
    *,
    config=None,
    specs=(),
    results=None,
    failures=None,
    budget=None,
    blackboard=None,
    deps=None,
    subtask=None,
):
    config = config or make_config()
    deps = deps or make_deps(
        responses, specs=specs, results=results, failures=failures
    )
    return AgenticLoop(
        config=config,
        context=make_slice(config),
        subtask=subtask or Subtask(required_capability=config.capability),
        blackboard=blackboard or Blackboard(task_id="task-1", tenant_id="tenant-a"),
        budget=budget or TaskBudget(),
        deps=deps,
    ), deps


# ── Happy paths ─────────────────────────────────────────────────────────────


def test_final_answer_completes_immediately() -> None:
    loop, _ = build_loop([text_response("done")])

    outcome = loop.start("system prompt")

    assert outcome.result is not None
    assert outcome.result.status is SpecialistStatus.COMPLETED
    assert outcome.result.output == "done"
    assert outcome.result.confidence == 1.0


def test_tool_call_then_final_answer() -> None:
    loop, deps = build_loop(
        [tool_response(("web_search", {"q": "x"})), text_response("found it")],
        specs=[read_tool()],
        results={"web_search": "a result"},
    )

    outcome = loop.start("system prompt")

    assert outcome.result.output == "found it"
    assert [i.tool_name for i in deps.tools.invocations] == ["web_search"]


def test_assistant_tool_call_message_precedes_tool_results() -> None:
    """Regression guard for a total-failure bug.

    OpenAI-compatible APIs reject a ``role: "tool"`` message that is not preceded
    by an assistant turn declaring the matching ``tool_call_id``. Omitting that
    assistant message makes every multi-turn tool run fail with a 400.
    """
    loop, deps = build_loop(
        [tool_response(("web_search", {"q": "x"})), text_response("ok")],
        specs=[read_tool()],
    )

    loop.start("system prompt")

    second_turn = deps.llm.calls[1]
    assistant = next(m for m in second_turn if m["role"] == "assistant")
    tool_msg = next(m for m in second_turn if m["role"] == "tool")

    assert second_turn.index(assistant) < second_turn.index(tool_msg)
    declared = {tc["id"] for tc in assistant["tool_calls"]}
    assert tool_msg["tool_call_id"] in declared
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"q": "x"}


def test_usage_is_metered_per_turn() -> None:
    meter = RecordingMeter()
    deps = make_deps(
        [tool_response(("web_search", {})), text_response("ok")],
        specs=[read_tool()],
        meter=meter,
    )
    loop, _ = build_loop(None, deps=deps, specs=[read_tool()])

    outcome = loop.start("system prompt")

    assert len(meter.records) == 2
    assert meter.total_tokens == outcome.result.tokens_used
    assert meter.records[0]["principal_id"] == PRINCIPAL


# ── Tool authorization ──────────────────────────────────────────────────────


def test_hallucinated_tool_is_reported_not_fatal() -> None:
    """A made-up tool name is normal model behaviour, not a system failure."""
    loop, _ = build_loop(
        [tool_response(("no_such_tool", {})), text_response("recovered")],
        specs=[read_tool()],
    )

    outcome = loop.start("system prompt")

    assert outcome.result.status is SpecialistStatus.COMPLETED
    assert outcome.result.output == "recovered"


def test_declared_but_unconnected_tools_are_flagged() -> None:
    """Authorization is config ∩ what the principal actually connected.

    Offering a tool whose credentials are missing guarantees a failure several
    seconds and several thousand tokens later, so it is never offered and the
    misconfiguration is surfaced in metadata.
    """
    config = make_config(tools=("web_search", "send_email"))
    loop, _ = build_loop([text_response("ok")], config=config, specs=[read_tool()])

    outcome = loop.start("system prompt")

    assert loop.unavailable_tools == ["send_email"]
    assert outcome.result.metadata["unavailable_tools"] == ["send_email"]


def test_only_authorized_tools_are_offered_to_the_model() -> None:
    config = make_config(tools=("web_search",))
    loop, deps = build_loop(
        [text_response("ok")],
        config=config,
        specs=[read_tool(), read_tool("other_tool")],
    )

    loop.start("system prompt")

    assert "other_tool" not in deps.llm.offered_tools[0]
    assert "web_search" in deps.llm.offered_tools[0]


def test_wildcard_grants_every_available_tool() -> None:
    """`tools=("*",)` offers the principal's whole connected catalog.

    For hosts whose gateway already scopes discovery per principal, a config
    should not have to enumerate (and chase) every tool name to say "use
    anything you have". Nothing is flagged unavailable: the wildcard cannot
    name a missing tool.
    """
    from baton_protocol import TOOL_WILDCARD

    config = make_config(tools=(TOOL_WILDCARD,))
    loop, deps = build_loop(
        [text_response("ok")],
        config=config,
        specs=[read_tool(), read_tool("other_tool")],
    )

    outcome = loop.start("system prompt")

    assert "web_search" in deps.llm.offered_tools[0]
    assert "other_tool" in deps.llm.offered_tools[0]
    assert loop.unavailable_tools == []
    assert outcome.result.metadata["unavailable_tools"] == []


def test_tool_failure_is_fed_back_to_the_model() -> None:
    loop, _ = build_loop(
        [tool_response(("web_search", {})), text_response("handled")],
        specs=[read_tool()],
        failures={"web_search": "upstream timeout"},
    )

    outcome = loop.start("system prompt")

    assert outcome.result.status is SpecialistStatus.COMPLETED


# ── Approval suspension ─────────────────────────────────────────────────────


def test_side_effecting_tool_suspends_before_executing() -> None:
    approvals = RecordingApprovalGateway()
    deps = make_deps(
        [tool_response(("send_email", {"to": "a@b.c"}))],
        specs=[write_tool()],
        approvals=approvals,
    )
    config = make_config(tools=("send_email",))
    loop, _ = build_loop(None, config=config, deps=deps)

    outcome = loop.start("system prompt")

    assert outcome.suspended
    assert outcome.approval_id == "approval-1"
    assert outcome.run.status is SpecialistStatus.AWAITING_APPROVAL
    assert outcome.run.pending_approval.tool_name == "send_email"
    # Critically: nothing was sent.
    assert deps.tools.invocations == []
    assert len(approvals.requests) == 1


def test_suspended_run_survives_serialization() -> None:
    """The run must be storable, since a human may take hours to respond."""
    deps = make_deps(
        [tool_response(("send_email", {"to": "a@b.c"}))], specs=[write_tool()]
    )
    config = make_config(tools=("send_email",))
    loop, _ = build_loop(None, config=config, deps=deps)

    outcome = loop.start("system prompt")
    revived = SpecialistRun.from_dict(json.loads(json.dumps(outcome.run.to_dict())))

    assert revived.pending_approval.tool_name == "send_email"
    assert revived.messages == outcome.run.messages
    assert revived.status is SpecialistStatus.AWAITING_APPROVAL


def test_resume_after_approval_executes_the_tool() -> None:
    deps = make_deps(
        [tool_response(("send_email", {"to": "a@b.c"})), text_response("sent")],
        specs=[write_tool()],
        results={"send_email": "message-id-1"},
    )
    config = make_config(tools=("send_email",))
    loop, _ = build_loop(None, config=config, deps=deps)

    suspended = loop.start("system prompt")
    outcome = loop.resume(suspended.run, approved=True)

    assert outcome.result.status is SpecialistStatus.COMPLETED
    assert [i.tool_name for i in deps.tools.invocations] == ["send_email"]
    assert deps.tools.invocations[0].idempotency_key is not None


def test_history_compaction_elides_old_tool_results_keeps_recent() -> None:
    """Tool results are re-billed every turn they sit in the history, so the
    cost of a browsing member grows quadratically unless older results are
    collapsed. Only the newest few stay full; stubs keep the history well-formed.
    """
    from baton_protocol.loop import (
        ELIDED_TOOL_RESULT_CHARS,
        KEEP_RECENT_TOOL_RESULTS,
        _compact_history,
    )

    big = "y" * 10_000
    messages = [{"role": "system", "content": "sys"}]
    for i in range(6):
        messages.append({"role": "assistant", "tool_calls": [{"id": f"c{i}"}]})
        messages.append(
            {"role": "tool", "tool_call_id": f"c{i}", "name": "browse", "content": big}
        )

    _compact_history(messages)

    tool_msgs = [m for m in messages if m["role"] == "tool"]
    compacted = [m for m in tool_msgs if "COMPACTED" in m["content"]]
    full = [m for m in tool_msgs if m["content"] == big]
    assert len(full) == KEEP_RECENT_TOOL_RESULTS
    assert len(compacted) == len(tool_msgs) - KEEP_RECENT_TOOL_RESULTS
    # The newest results are the untouched ones.
    assert tool_msgs[-1]["content"] == big
    for m in compacted:
        assert len(m["content"]) < ELIDED_TOOL_RESULT_CHARS + 300
    # Small results are never touched, and compaction is idempotent.
    before = [m["content"] for m in tool_msgs]
    _compact_history(messages)
    assert [m["content"] for m in tool_msgs] == before


def test_giant_tool_results_are_truncated_before_entering_history() -> None:
    """A tool result is re-billed on every later iteration it sits in the
    history, so an unbounded web page dump quietly burns the whole run budget.
    Oversized results are cut and the model is told to extract facts now."""
    from baton_protocol.loop import MAX_TOOL_RESULT_CHARS, _tool_message
    from baton_protocol.models import ToolResult

    huge = ToolResult(success=True, data="x" * (MAX_TOOL_RESULT_CHARS * 3))
    message = _tool_message("call-1", "web_operator_execute", huge)
    assert len(message["content"]) < MAX_TOOL_RESULT_CHARS + 400
    assert "TRUNCATED" in message["content"]

    small = ToolResult(success=True, data="all good")
    assert _tool_message("call-2", "web_search", small)["content"] == "all good"


def test_idempotency_keys_are_per_call_not_per_tool() -> None:
    """Two different invocations of one tool must not share a replay key.

    A researcher legitimately browses several sites with the same web-operator
    tool. With a per-tool key the second mission replayed the first mission's
    cached result: the specialist never received new data, asked again, and
    the run looped through approval gates forever.
    """
    deps = make_deps([], specs=[write_tool()])
    config = make_config(tools=("send_email",))
    loop, _ = build_loop(None, config=config, deps=deps)

    first = loop._idempotency_key("send_email", {"to": "a@b.c"})
    second = loop._idempotency_key("send_email", {"to": "d@e.f"})
    repeat = loop._idempotency_key("send_email", {"to": "a@b.c"})

    assert first != second  # new arguments → a real execution, not a replay
    assert first == repeat  # crash-retry of the same call still replays


def test_resume_recomputes_the_idempotency_key_instead_of_trusting_the_parked_one() -> None:
    """A run parked under a stale key must still execute the approved action.

    The key stored on the approval may come from an older key format. Replaying
    it would hit a cached result for a *different* call and the approved action
    would never run. Resume must derive the key from tool + arguments, and must
    do so identically in a fresh loop (a different process in production).
    """
    args = {"to": "a@b.c"}
    responses = [tool_response(("send_email", args)), text_response("sent")]
    config = make_config(tools=("send_email",))

    # The key is scoped to the subtask, so both loops act for the same one —
    # as the orchestrator does when it rebuilds a loop to resume a parked run.
    subtask = Subtask(required_capability=config.capability)

    deps = make_deps(responses, specs=[write_tool()], results={"send_email": "ok"})
    loop, _ = build_loop(None, config=config, deps=deps, subtask=subtask)
    suspended = loop.start("system prompt")

    # Simulate a run parked by an older build, then stored and reloaded.
    suspended.run.pending_approval.idempotency_key = "legacy-key-format"
    revived = SpecialistRun.from_dict(json.loads(json.dumps(suspended.run.to_dict())))
    assert revived.pending_approval.idempotency_key == "legacy-key-format"

    fresh_deps = make_deps(
        [text_response("sent")], specs=[write_tool()], results={"send_email": "ok"}
    )
    fresh_loop, _ = build_loop(None, config=config, deps=fresh_deps, subtask=subtask)
    outcome = fresh_loop.resume(revived, approved=True)

    assert outcome.result.status is SpecialistStatus.COMPLETED
    assert len(fresh_deps.tools.invocations) == 1
    used = fresh_deps.tools.invocations[0].idempotency_key
    assert used != "legacy-key-format"
    assert used == fresh_loop._idempotency_key("send_email", args)
    # Stable across loops, so a crash-retry of the resume still replays.
    assert used == loop._idempotency_key("send_email", args)


def test_denial_is_information_not_failure() -> None:
    """A refused action is something the specialist can work around."""
    deps = make_deps(
        [
            tool_response(("send_email", {"to": "a@b.c"})),
            text_response("understood, not sending"),
        ],
        specs=[write_tool()],
    )
    config = make_config(tools=("send_email",))
    loop, _ = build_loop(None, config=config, deps=deps)

    suspended = loop.start("system prompt")
    outcome = loop.resume(suspended.run, approved=False, denial_reason="not authorized")

    assert outcome.result.status is SpecialistStatus.COMPLETED
    assert deps.tools.invocations == []
    tool_msg = next(
        m for m in deps.llm.calls[-1] if m.get("role") == "tool"
    )
    assert "not authorized" in tool_msg["content"]


def test_suspension_mid_batch_resumes_only_the_outstanding_call() -> None:
    """A single assistant turn can request several tools at once.

    Suspending on the second must leave the first answered and the second
    outstanding, then complete only the remainder on resume — otherwise the
    message history is malformed or a tool runs twice.
    """
    deps = make_deps(
        [
            tool_response(("web_search", {"q": "x"}), ("send_email", {"to": "a@b.c"})),
            text_response("all done"),
        ],
        specs=[read_tool(), write_tool()],
    )
    config = make_config(tools=("web_search", "send_email"))
    loop, _ = build_loop(None, config=config, deps=deps)

    suspended = loop.start("system prompt")
    assert suspended.suspended
    assert [i.tool_name for i in deps.tools.invocations] == ["web_search"]

    outcome = loop.resume(suspended.run, approved=True)

    assert outcome.result.status is SpecialistStatus.COMPLETED
    assert [i.tool_name for i in deps.tools.invocations] == ["web_search", "send_email"]
    final_turn = deps.llm.calls[-1]
    tool_ids = [m["tool_call_id"] for m in final_turn if m.get("role") == "tool"]
    assert tool_ids == ["call-1", "call-2"]


def test_auto_approve_skips_the_gate() -> None:
    deps = make_deps(
        [tool_response(("send_email", {})), text_response("sent")],
        specs=[write_tool()],
    )
    config = make_config(tools=("send_email",), auto_approve_side_effects=True)
    loop, _ = build_loop(None, config=config, deps=deps)

    outcome = loop.start("system prompt")

    assert outcome.result.status is SpecialistStatus.COMPLETED
    assert [i.tool_name for i in deps.tools.invocations] == ["send_email"]


# ── Limits ──────────────────────────────────────────────────────────────────


def test_iteration_exhaustion_returns_partial_output() -> None:
    """Discarding a run's whole output because it needed one more turn wastes
    real money. Return it degraded and let the caller judge."""
    config = make_config(max_iterations=1)
    loop, _ = build_loop(
        [tool_response(("web_search", {}), content="partial finding")],
        config=config,
        specs=[read_tool()],
    )

    outcome = loop.start("system prompt")

    assert outcome.result.status is SpecialistStatus.TRUNCATED
    assert outcome.result.output == "partial finding"
    assert outcome.result.metadata["truncated"] is True
    assert outcome.result.confidence < 1.0


def test_task_token_budget_is_shared_and_fatal() -> None:
    budget = TaskBudget(max_total_tokens=15)
    loop, _ = build_loop([text_response("hi", tokens=10)], budget=budget)

    outcome = loop.start("system prompt")

    assert outcome.result.status is SpecialistStatus.FAILED
    assert "budget" in outcome.result.error


def test_per_specialist_token_ceiling_is_enforced() -> None:
    config = make_config(max_tokens=5)
    loop, _ = build_loop([text_response("hi", tokens=10)], config=config)

    outcome = loop.start("system prompt")

    assert outcome.result.status is SpecialistStatus.FAILED


def test_tool_call_budget_is_enforced() -> None:
    budget = TaskBudget(max_tool_calls=1)
    loop, _ = build_loop(
        [
            tool_response(("web_search", {}), ("web_search", {})),
            text_response("ok"),
        ],
        specs=[read_tool()],
        budget=budget,
    )

    outcome = loop.start("system prompt")

    assert outcome.result.status is SpecialistStatus.FAILED


# ── Pseudo-tools ────────────────────────────────────────────────────────────


def test_blackboard_read_pseudo_tool() -> None:
    board = Blackboard(task_id="task-1", tenant_id="tenant-a")
    board.post(
        specialist_id="s1", capability="screener", key="top", value={"name": "ada"}
    )
    loop, deps = build_loop(
        [
            tool_response(
                ("blackboard_read", {"capability": "screener", "key": "top"})
            ),
            text_response("read it"),
        ],
        blackboard=board,
    )

    loop.start("system prompt")

    tool_msg = next(m for m in deps.llm.calls[-1] if m.get("role") == "tool")
    assert "ada" in tool_msg["content"]


def test_blackboard_pseudo_tools_are_always_offered() -> None:
    loop, deps = build_loop([text_response("ok")])

    loop.start("system prompt")

    assert "blackboard_read" in deps.llm.offered_tools[0]
    assert "blackboard_list" in deps.llm.offered_tools[0]


def test_delegate_tool_is_hidden_when_no_grant_exists() -> None:
    loop, deps = build_loop([text_response("ok")], config=make_config())

    loop.start("system prompt")

    assert "delegate" not in deps.llm.offered_tools[0]


def test_delegate_is_offered_when_granted() -> None:
    config = make_config(can_delegate_to=("helper",))
    loop, deps = build_loop([text_response("ok")], config=config)

    loop.start("system prompt")

    assert "delegate" in deps.llm.offered_tools[0]


def test_delegating_without_a_grant_is_reported_to_the_model() -> None:
    """Refusing authority should redirect the model, not crash the task."""
    config = make_config(can_delegate_to=("helper",))
    deps = make_deps(
        [
            tool_response(("delegate", {"capability": "forbidden", "input_data": {}})),
            text_response("took another route"),
        ]
    )
    deps.delegate = lambda request: (_ for _ in ()).throw(
        AssertionError("must not be reached")
    )
    loop, _ = build_loop(None, config=config, deps=deps)

    outcome = loop.start("system prompt")

    assert outcome.result.status is SpecialistStatus.COMPLETED
    tool_msg = next(m for m in deps.llm.calls[-1] if m.get("role") == "tool")
    assert "may not delegate" in tool_msg["content"]
