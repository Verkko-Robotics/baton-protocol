"""The agentic loop — suspendable, resumable, provider-neutral.

This is the execution core. It differs from a typical tool-calling loop in one
structural way: it can **stop in the middle and be resumed later from storage**.

That requirement comes from human approval. A side-effecting tool call may wait
hours for a person to authorize it, which is far longer than any HTTP request or
worker thread lives. So the loop's position is never held on the Python stack —
it lives in a :class:`SpecialistRun`, which serializes to JSON. Suspending means
persisting the run; resuming means loading it and calling :meth:`resume`.

The awkward consequence is that the loop cannot be a simple ``for`` statement,
because a ``for`` statement's position dies with its frame. Instead
:meth:`_drive` reconstructs where it was from the message history each time it
is entered, by finding tool calls that have not yet been answered. That makes
starting and resuming the same code path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ._codec import utcnow
from .blackboard import Blackboard
from .budget import TaskBudget
from .errors import (
    ApprovalRequired,
    BudgetExceeded,
    DelegationDenied,
    DelegationDepthExceeded,
    DelegationCycle,
    RecoverableError,
    SchemaViolation,
    SpecialistTimeout,
    ToolExecutionFailed,
    ToolNotAuthorized,
)
from .interfaces import (
    ApprovalGateway,
    Clock,
    LLMGateway,
    SchemaValidator,
    SystemClock,
    ToolGateway,
    UsageMeter,
)
from .models import (
    ContextSlice,
    DelegationRequest,
    PendingApproval,
    SpecialistConfig,
    SpecialistResult,
    SpecialistRun,
    SpecialistStatus,
    Subtask,
    ToolCall,
    ToolResult,
    ToolSpec,
)

#: Pseudo-tool: read a fact another specialist posted to the blackboard.
TOOL_BLACKBOARD_READ = "blackboard_read"
#: Pseudo-tool: list every blackboard address visible to this specialist.
TOOL_BLACKBOARD_LIST = "blackboard_list"
#: Pseudo-tool: ask the orchestrator to run another capability.
TOOL_DELEGATE = "delegate"

PSEUDO_TOOLS = frozenset({TOOL_BLACKBOARD_READ, TOOL_BLACKBOARD_LIST, TOOL_DELEGATE})

#: Declaring this in ``SpecialistConfig.tools`` grants every tool the
#: principal's gateway reports as available, instead of a fixed allowlist.
TOOL_WILDCARD = "*"

#: Consecutive schema failures tolerated before the run is failed. Bounded so a
#: model that cannot produce valid output stops burning budget.
MAX_SCHEMA_RETRIES = 2


@dataclass
class RunOutcome:
    """What came back from starting or resuming a run.

    Exactly one of ``result`` and ``approval_id`` is set. A suspended run has an
    ``approval_id`` and must be persisted; a finished run has a ``result``.
    """

    run: SpecialistRun
    result: SpecialistResult | None = None
    approval_id: str | None = None

    @property
    def suspended(self) -> bool:
        return self.result is None


@dataclass
class LoopDependencies:
    """Injected collaborators for a loop.

    Grouped into one object because a loop needs six of them and threading six
    keyword arguments through every call site is how they drift out of sync.
    """

    llm: LLMGateway
    tools: ToolGateway
    approvals: ApprovalGateway
    meter: UsageMeter
    validator: SchemaValidator | None = None
    clock: Clock = field(default_factory=SystemClock)
    #: Set by the orchestrator. Absent means delegation is unavailable, which
    #: is reported to the model rather than crashing.
    delegate: Callable[[DelegationRequest], SpecialistResult] | None = None


def _pseudo_tool_specs(config: SpecialistConfig) -> list[ToolSpec]:
    """Protocol-provided tools, offered on top of the host's own.

    None have side effects: reading the blackboard is inert, and a delegated
    specialist runs its own approval gate for anything it does.
    """
    specs = [
        ToolSpec(
            name=TOOL_BLACKBOARD_READ,
            description=(
                "Read a fact posted by another specialist on this task. "
                "Returns null if the fact does not exist or is not visible to you."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "capability": {
                        "type": "string",
                        "description": "Capability that posted the fact.",
                    },
                    "key": {"type": "string", "description": "Fact key."},
                },
                "required": ["capability", "key"],
            },
            side_effect=False,
        ),
        ToolSpec(
            name=TOOL_BLACKBOARD_LIST,
            description=(
                "List every fact address visible to you on this task, "
                "formatted as capability.key."
            ),
            parameters={"type": "object", "properties": {}},
            side_effect=False,
        ),
    ]
    if config.can_delegate_to:
        specs.append(
            ToolSpec(
                name=TOOL_DELEGATE,
                description=(
                    "Ask another specialist to do work you are not authorized "
                    "to do yourself. Allowed capabilities: "
                    + ", ".join(sorted(config.can_delegate_to))
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "capability": {
                            "type": "string",
                            "enum": sorted(config.can_delegate_to),
                        },
                        "input_data": {
                            "type": "object",
                            "description": "Parameters for the delegated subtask.",
                        },
                    },
                    "required": ["capability", "input_data"],
                },
                side_effect=False,
            )
        )
    return specs


class AgenticLoop:
    """Executes one specialist's reasoning and tool use.

    One instance drives exactly one specialist run. It is not thread-safe and is
    not meant to be reused across runs.
    """

    def __init__(
        self,
        *,
        config: SpecialistConfig,
        context: ContextSlice,
        subtask: Subtask,
        blackboard: Blackboard,
        budget: TaskBudget,
        deps: LoopDependencies,
        output_parser: Callable[[str], Any] | None = None,
        confidence_scorer: Callable[[Any, SpecialistStatus], float] | None = None,
    ) -> None:
        self.config = config
        self.context = context
        self.subtask = subtask
        self.blackboard = blackboard
        self.budget = budget
        self.deps = deps
        self._parse_output = output_parser or (lambda raw: raw)
        self._score = confidence_scorer or _default_confidence

        # Tool authorization is the intersection of the static declaration and
        # what the principal has actually connected. Offering a tool whose
        # credentials are missing guarantees a failure later in the run.
        # Declaring "*" grants every tool the principal has: hosts whose
        # gateway already scopes discovery to the principal
        # can hand a specialist the full catalog without enumerating it — the
        # side-effect approval gate still applies per call regardless.
        host_tools = {t.name: t for t in self.deps.tools.available_tools(
            context.principal_id
        )}
        declared = set(config.tools)
        if TOOL_WILDCARD in declared:
            self._host_tools: dict[str, ToolSpec] = dict(host_tools)
        else:
            self._host_tools = {
                name: spec for name, spec in host_tools.items() if name in declared
            }
        self._pseudo_tools = {t.name: t for t in _pseudo_tool_specs(config)}
        self._offered: list[ToolSpec] = [
            *self._host_tools.values(),
            *self._pseudo_tools.values(),
        ]
        #: Declared but unavailable — surfaced in result metadata so a
        #: misconfiguration is visible rather than looking like model failure.
        self.unavailable_tools = sorted(declared - {TOOL_WILDCARD} - set(host_tools))

    # ── Entry points ────────────────────────────────────────────────────────

    def start(self, system_prompt: str, user_prompt: str | None = None) -> RunOutcome:
        """Begin a fresh run."""
        run = SpecialistRun(
            specialist_id=_new_specialist_id(),
            capability=self.config.capability,
            subtask_id=self.subtask.id,
            task_id=self.context.task_id,
            messages=[{"role": "system", "content": system_prompt}],
            status=SpecialistStatus.RUNNING,
            depth=self.context.depth,
            ancestry=self.context.ancestry,
            started_at=self.deps.clock.now(),
        )
        if user_prompt:
            run.messages.append({"role": "user", "content": user_prompt})
        return self._drive(run)

    def resume(
        self,
        run: SpecialistRun,
        *,
        approved: bool,
        denial_reason: str | None = None,
    ) -> RunOutcome:
        """Continue a run that was suspended awaiting approval.

        A denial is fed back to the model as a tool error rather than failing the
        run. "You may not send that email" is information the specialist can act
        on — by revising the draft or by finishing without sending.
        """
        approval = run.pending_approval
        if approval is None:
            raise ValueError("run is not awaiting approval")

        run.pending_approval = None
        run.status = SpecialistStatus.RUNNING

        if approved:
            # Recomputed, not read from the parked approval: the key derives
            # deterministically from tool + arguments anyway, and a run parked
            # under an older key format would otherwise replay a stale cached
            # result instead of executing the approved action.
            result = self._execute_host_tool(
                approval.tool_name,
                approval.arguments,
                idempotency_key=self._idempotency_key(
                    approval.tool_name, approval.arguments
                ),
            )
        else:
            reason = denial_reason or "a human declined this action"
            result = ToolResult(
                success=False,
                error=f"not performed: {reason}. Do not retry this action.",
            )

        run.messages.append(
            _tool_message(approval.call_id, approval.tool_name, result)
        )
        return self._drive(run)

    # ── Core driver ─────────────────────────────────────────────────────────

    def _drive(self, run: SpecialistRun) -> RunOutcome:
        """Advance the run until it finishes or suspends.

        Re-derives its position from ``run.messages`` on entry, which is what
        makes start and resume identical after their first step.
        """
        while True:
            self._check_deadline(run)

            unanswered = _unanswered_tool_calls(run.messages)
            if unanswered:
                suspended = self._dispatch(run, unanswered)
                if suspended is not None:
                    return suspended
                continue

            if run.iteration >= self.config.max_iterations:
                return self._finish_truncated(run)

            # Keep the prompt from growing quadratically: a tool result that
            # stays in the history at full size is re-billed on EVERY later
            # turn, so a member that browsed a dozen pages pays for all twelve
            # dumps again each iteration (~900k tokens measured on one member).
            # Older results collapse to stubs — the model was told at delivery
            # to extract what it needed immediately.
            _compact_history(run.messages)

            try:
                response = self.deps.llm.complete(
                    messages=run.messages,
                    tools=self._offered,
                    model=self.config.model,
                    temperature=self.config.temperature,
                    max_tokens=self.config.completion_tokens or None,
                )
            except Exception as exc:
                return self._finish_failed(run, f"model call failed: {exc}")

            run.iteration += 1

            try:
                self._charge(run, response.prompt_tokens, response.completion_tokens)
            except BudgetExceeded as exc:
                return self._finish_failed(run, str(exc))

            if response.is_final:
                return self._finish_final(run, response.content or "")

            run.messages.append(_assistant_tool_call_message(response))

    # ── Tool dispatch ───────────────────────────────────────────────────────

    def _dispatch(
        self,
        run: SpecialistRun,
        calls: Sequence[ToolCall],
    ) -> RunOutcome | None:
        """Answer outstanding tool calls.

        Returns a suspended :class:`RunOutcome` if one needs approval, otherwise
        ``None`` to mean "keep going". Calls are answered in order, so a
        suspension leaves earlier results in the history and later calls
        unanswered — which is exactly what :func:`_unanswered_tool_calls`
        reconstructs on resume.
        """
        for call in calls:
            try:
                self.budget.charge_tool_call()
            except BudgetExceeded as exc:
                run.messages.append(
                    _tool_message(
                        call.id, call.name, ToolResult(success=False, error=str(exc))
                    )
                )
                # Fatal, but only after the history stays well-formed: every
                # tool call must have a matching tool message or the next
                # model call is rejected outright.
                return self._finish_failed(run, str(exc))

            try:
                result = self._invoke(call)
            except ApprovalRequired as pending:
                return self._suspend(run, pending)
            except RecoverableError as exc:
                result = ToolResult(success=False, error=str(exc))
            except (DelegationDenied, DelegationCycle, DelegationDepthExceeded) as exc:
                # Authority and recursion refusals are told to the model rather
                # than crashing the task: it can choose another route or finish
                # without the delegate.
                result = ToolResult(success=False, error=str(exc))
            except BudgetExceeded as exc:
                run.messages.append(
                    _tool_message(
                        call.id, call.name, ToolResult(success=False, error=str(exc))
                    )
                )
                return self._finish_failed(run, str(exc))
            except Exception as exc:
                result = ToolResult(
                    success=False, error=f"tool raised unexpectedly: {exc}"
                )

            run.messages.append(_tool_message(call.id, call.name, result))
        return None

    def _invoke(self, call: ToolCall) -> ToolResult:
        """Route one tool call to a pseudo-tool or the host gateway."""
        if call.name == TOOL_BLACKBOARD_READ:
            value = self.blackboard.read(
                capability=str(call.arguments.get("capability", "")),
                key=str(call.arguments.get("key", "")),
                requesting_capability=self.config.capability,
            )
            return ToolResult(success=True, data=value)

        if call.name == TOOL_BLACKBOARD_LIST:
            return ToolResult(
                success=True,
                data=sorted(self.blackboard.read_all(self.config.capability)),
            )

        if call.name == TOOL_DELEGATE:
            return self._delegate(call)

        return self._authorized_host_call(call)

    def _authorized_host_call(self, call: ToolCall) -> ToolResult:
        if call.name not in self._host_tools:
            # Covers both hallucinated names and tools declared in config but
            # not connected by the principal. Recoverable either way.
            raise ToolNotAuthorized(call.name, sorted(self._host_tools))

        spec = self._host_tools[call.name]
        needs_approval = spec.side_effect and not self.config.auto_approve_side_effects
        if needs_approval:
            raise ApprovalRequired(call.name, call.arguments, call.id)

        return self._execute_host_tool(
            call.name,
            call.arguments,
            idempotency_key=self._idempotency_key(call.name, call.arguments),
        )

    def _execute_host_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        idempotency_key: str | None,
    ) -> ToolResult:
        result = self.deps.tools.execute(
            tool_name=tool_name,
            principal_id=self.context.principal_id,
            arguments=arguments,
            idempotency_key=idempotency_key,
        )
        if not result.success:
            raise ToolExecutionFailed(tool_name, result.error or "unknown error")
        return result

    def _delegate(self, call: ToolCall) -> ToolResult:
        if self.deps.delegate is None:
            raise ToolNotAuthorized(TOOL_DELEGATE, sorted(self._host_tools))

        capability = str(call.arguments.get("capability", ""))
        if capability not in self.config.can_delegate_to:
            raise DelegationDenied(
                self.config.capability, capability, list(self.config.can_delegate_to)
            )

        request = DelegationRequest(
            requesting_specialist_id=self.config.capability,
            requesting_capability=self.config.capability,
            required_capability=capability,
            input_data=dict(call.arguments.get("input_data") or {}),
            depth=self.context.depth + 1,
            ancestry=(*self.context.ancestry, self.config.capability),
        )

        try:
            result = self.deps.delegate(request)
        except BudgetExceeded as exc:
            # Running out of *spawns* only means no more help is available; the
            # caller can still finish its own work, and it has already been paid
            # for. Running out of tokens is different — no further model call is
            # possible — so that stays fatal.
            if exc.kind == "spawn":
                return ToolResult(
                    success=False,
                    error=f"{exc}; no further delegation is available, "
                    "continue without it",
                )
            raise

        if not result.succeeded:
            return ToolResult(
                success=False,
                error=f"delegate {capability!r} failed: {result.error}",
            )
        return ToolResult(success=True, data=result.output)

    def _suspend(self, run: SpecialistRun, pending: ApprovalRequired) -> RunOutcome:
        approval = PendingApproval(
            tool_name=pending.tool_name,
            arguments=pending.arguments,
            call_id=pending.call_id,
            iteration=run.iteration,
            idempotency_key=self._idempotency_key(pending.tool_name, pending.arguments),
            requested_at=self.deps.clock.now(),
        )
        run.pending_approval = approval
        run.status = SpecialistStatus.AWAITING_APPROVAL
        approval_id = self.deps.approvals.request(
            context=self.context, approval=approval
        )
        return RunOutcome(run=run, approval_id=approval_id)

    # ── Termination ─────────────────────────────────────────────────────────

    def _finish_final(self, run: SpecialistRun, content: str) -> RunOutcome:
        """Handle a model response with no tool calls.

        A schema violation is retried in-loop rather than failed: the model is
        told what was wrong and given another turn, bounded by
        :data:`MAX_SCHEMA_RETRIES`.
        """
        try:
            output = self._parse_output(content)
        except Exception as exc:
            return self._retry_or_fail(run, f"output could not be parsed: {exc}")

        violations = self._validate(output)
        if violations:
            return self._retry_or_fail(
                run,
                "output did not match the required schema: " + "; ".join(violations),
            )

        run.status = SpecialistStatus.COMPLETED
        return RunOutcome(run=run, result=self._result(run, output))

    def _retry_or_fail(self, run: SpecialistRun, complaint: str) -> RunOutcome:
        run.schema_retries += 1
        if run.schema_retries > MAX_SCHEMA_RETRIES:
            return self._finish_failed(run, complaint)
        run.messages.append(
            {
                "role": "user",
                "content": f"{complaint}. Return corrected output only.",
            }
        )
        return self._drive(run)

    def _finish_truncated(self, run: SpecialistRun) -> RunOutcome:
        """Hit the iteration ceiling.

        Returns the last assistant text as degraded output instead of raising.
        Discarding a run's entire work because it needed one more turn wastes
        real money, and a caller with a confidence score can decide for itself.
        """
        partial = _last_assistant_text(run.messages)
        run.status = SpecialistStatus.TRUNCATED
        result = self._result(run, partial)
        result.metadata["truncated"] = True
        result.metadata["reason"] = (
            f"reached max_iterations ({self.config.max_iterations})"
        )
        return RunOutcome(run=run, result=result)

    def _finish_failed(self, run: SpecialistRun, error: str) -> RunOutcome:
        run.status = SpecialistStatus.FAILED
        result = self._result(run, None)
        result.error = error
        result.confidence = 0.0
        return RunOutcome(run=run, result=result)

    def _result(self, run: SpecialistRun, output: Any) -> SpecialistResult:
        elapsed = (self.deps.clock.now() - run.started_at).total_seconds()
        return SpecialistResult(
            specialist_id=run.specialist_id,
            subtask_id=run.subtask_id,
            capability=self.config.capability,
            status=run.status,
            output=output,
            confidence=self._score(output, run.status),
            tokens_used=run.tokens_used,
            duration_ms=int(elapsed * 1000),
            metadata={
                "iterations": run.iteration,
                "depth": run.depth,
                "ancestry": list(run.ancestry),
                "unavailable_tools": self.unavailable_tools,
            },
        )

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _validate(self, output: Any) -> list[str]:
        schema = self.subtask.expected_output_schema or self.config.output_schema
        if not schema:
            return []
        if self.deps.validator is None:
            # No validator wired: do not silently pretend the contract held.
            return []
        return self.deps.validator.validate(output, schema)

    def _charge(self, run: SpecialistRun, prompt: int, completion: int) -> None:
        total = prompt + completion
        run.tokens_used += total
        self.budget.charge_tokens(total)

        self.deps.meter.record(
            principal_id=self.context.principal_id,
            capability=self.config.capability,
            model=self.config.model,
            prompt_tokens=prompt,
            completion_tokens=completion,
            idempotency_key=f"{run.specialist_id}:{run.iteration}",
            metadata={"task_id": run.task_id, "subtask_id": run.subtask_id},
        )

<<<<<<< HEAD
        if run.tokens_used > self.config.max_tokens:
=======
        if self.config.max_tokens and run.tokens_used > self.config.max_tokens:
>>>>>>> 4d653be (BATON 1.0: specification, reference library, JSON Schemas and conformance suite)
            raise BudgetExceeded(
                "specialist token", run.tokens_used, self.config.max_tokens
            )

    def _check_deadline(self, run: SpecialistRun) -> None:
        """Enforce the wall-clock allowance.

        Time spent suspended awaiting approval is excluded — a human's response
        time is not the specialist's fault.
        """
        if not self.config.timeout_seconds:
            return
        elapsed = (self.deps.clock.now() - run.started_at).total_seconds()
        if elapsed > self.config.timeout_seconds:
            raise SpecialistTimeout(self.config.capability, self.config.timeout_seconds)

    def _idempotency_key(self, tool_name: str, arguments: dict[str, Any] | None = None) -> str:
        """Stable across retries of the same subtask, distinct per tool CALL.

        Derived from the subtask's key so replaying a task after a crash does
        not re-send an email that already went out. The arguments are part of
        the key: the same specialist legitimately invokes one tool several
        times with different inputs (browse two sites, send two emails), and a
        per-tool key would replay the first call's cached result for every
        later one — the specialist never gets new data and spins forever.
        """
        base = self.subtask.idempotency_key or self.subtask.id
        key = f"{base}:{self.config.capability}:{tool_name}"
        if arguments:
            import hashlib
            import json

            digest = hashlib.sha1(
                json.dumps(arguments, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()[:12]
            key = f"{key}:{digest}"
        return key


# ── Module helpers ──────────────────────────────────────────────────────────


def _new_specialist_id() -> str:
    import uuid

    return str(uuid.uuid4())


def _default_confidence(output: Any, status: SpecialistStatus) -> float:
    if status is SpecialistStatus.COMPLETED:
        return 1.0 if output not in (None, "", [], {}) else 0.5
    if status is SpecialistStatus.TRUNCATED:
        return 0.4
    return 0.0


def _assistant_tool_call_message(response: Any) -> dict[str, Any]:
    """Build the assistant turn that *precedes* its tool results.

    Omitting this message is a subtle and total failure: OpenAI-compatible APIs
    reject a ``role: "tool"`` message that is not preceded by the assistant
    message declaring the matching ``tool_call_id``.
    """
    return {
        "role": "assistant",
        "content": response.content or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments),
                },
            }
            for call in response.tool_calls
        ],
    }


#: How many of the most recent tool results stay in the history at full size.
#: Everything older collapses to a stub: the specialist consumed those results
#: on the turn they arrived, and carrying the raw dumps forward only re-bills
#: them on every subsequent model call.
KEEP_RECENT_TOOL_RESULTS = 3

#: Size a compacted (older) tool result is reduced to.
ELIDED_TOOL_RESULT_CHARS = 1_500


def _compact_history(messages: list[dict[str, Any]]) -> None:
    """Collapse all but the newest tool results to stubs, in place.

    In-place and permanent on purpose: the persisted run resumes with the same
    compacted history, so a run parked at an approval gate does not resurrect
    the full dumps it already paid to read.
    """
    tool_indexes = [
        i for i, m in enumerate(messages) if m.get("role") == "tool"
    ]
    for i in tool_indexes[: max(0, len(tool_indexes) - KEEP_RECENT_TOOL_RESULTS)]:
        content = messages[i].get("content") or ""
        if len(content) <= ELIDED_TOOL_RESULT_CHARS or "[COMPACTED:" in content:
            continue
        messages[i]["content"] = (
            content[:ELIDED_TOOL_RESULT_CHARS]
            + f"\n\n[COMPACTED: this older result ({len(content):,} chars) was "
            "trimmed from your context. You already extracted its facts on the "
            "turn it arrived; re-request narrowly if something is missing.]"
        )


#: Ceiling for one tool result entering the message history. A web-operator
#: page dump can be hundreds of thousands of characters, and once it is in the
#: history it is re-billed on EVERY later iteration — one giant result burned
#: 800k tokens on a single member. Beyond this size the specialist gets the
#: head of the result plus an explicit instruction to extract what it needs
#: now, because the tail is gone for good.
MAX_TOOL_RESULT_CHARS = 24_000


def _tool_message(call_id: str, name: str, result: ToolResult) -> dict[str, Any]:
    content = result.as_message_content()
    if len(content) > MAX_TOOL_RESULT_CHARS:
        content = (
            content[:MAX_TOOL_RESULT_CHARS]
            + f"\n\n[TRUNCATED: the full result was {len(content):,} characters. "
            "Summarize or extract the facts you need from the text above NOW — "
            "the removed remainder will not be shown again. If something "
            "essential is missing, make a narrower request.]"
        )
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": content,
    }


def _unanswered_tool_calls(messages: Sequence[dict[str, Any]]) -> list[ToolCall]:
    """Tool calls from the latest assistant turn that have no result yet.

    This is how the loop recovers its position after being rehydrated from
    storage, and how it finishes a batch that was interrupted part-way by an
    approval suspension.
    """
    last_index = -1
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") == "assistant" and message.get("tool_calls"):
            last_index = index
            break
    if last_index < 0:
        return []

    answered = {
        m.get("tool_call_id")
        for m in messages[last_index + 1 :]
        if m.get("role") == "tool"
    }

    pending: list[ToolCall] = []
    for raw in messages[last_index]["tool_calls"]:
        if raw.get("id") in answered:
            continue
        function = raw.get("function") or {}
        raw_args = function.get("arguments") or "{}"
        try:
            arguments = (
                raw_args if isinstance(raw_args, dict) else json.loads(raw_args)
            )
        except json.JSONDecodeError:
            # Malformed arguments are the model's problem to fix, so keep the
            # call and let the tool report the error back to it.
            arguments = {}
        pending.append(
            ToolCall(
                id=str(raw.get("id", "")),
                name=str(function.get("name", "")),
                arguments=arguments if isinstance(arguments, dict) else {},
            )
        )
    return pending


def _last_assistant_text(messages: Sequence[dict[str, Any]]) -> str | None:
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content"):
            return str(message["content"])
    return None
