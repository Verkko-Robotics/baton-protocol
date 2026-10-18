"""The orchestrator — spawns specialists, routes delegation, owns the task.

Two responsibilities sit here and nowhere else.

**Specialist construction.** Nothing else may build a specialist. That is what
guarantees the protocol's central invariant: no specialist ever holds a
reference to another, so every inter-specialist interaction is either a
blackboard entry or a delegation request, and therefore recorded.

**Authority.** Tool grants, delegation grants, recursion depth, cycles and spend
are all checked here, before any work or money is committed.

Delegation is implemented as a **re-entrant call**, not message passing:
:meth:`Orchestrator.run_child` builds the child and runs it on the *caller's*
stack. Earlier designs used an event bus with the parent blocking on a
``sleep``-polled result, which deadlocks whenever the bus dispatches on a single
thread — the parent occupies the only thread that could run the child. Running
on the caller's stack removes the scheduler entirely, and the invariant survives
because the parent still never sees the child object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .blackboard import Blackboard
from .errors import (
    BudgetExceeded,
    CapabilityCollision,
    CapabilityNotRegistered,
    DelegationCycle,
    DelegationDenied,
    DelegationDepthExceeded,
    SpecialistTimeout,
    UnresolvableDependencies,
)
from .interfaces import (
    ContextStore,
    MemoryStore,
    ResultSink,
    TemplateRenderer,
)
from .loop import AgenticLoop, LoopDependencies, RunOutcome
from .models import (
    ContextEnvelope,
    ContextSlice,
    DelegationRequest,
    GapKind,
    SpecialistConfig,
    SpecialistResult,
    SpecialistRun,
    SpecialistStatus,
    Subtask,
    SubtaskStatus,
)

#: Envelope extension key under which suspended runs are parked.
SUSPENDED_RUNS_KEY = "cpp.suspended_runs"


@dataclass
class TaskOutcome:
    """Result of running one task to completion or to a stopping point."""

    envelope: ContextEnvelope
    results: list[SpecialistResult] = field(default_factory=list)
    #: Subtask id → approval id, for runs parked awaiting a human.
    awaiting_approval: dict[str, str] = field(default_factory=dict)

    @property
    def suspended(self) -> bool:
        return bool(self.awaiting_approval)

    @property
    def failed(self) -> list[SpecialistResult]:
        return [r for r in self.results if not r.succeeded]


class Orchestrator:
    """Runs a task's subtask graph and mediates every specialist interaction."""

    def __init__(
        self,
        *,
        registry: Mapping[str, SpecialistConfig],
        deps: LoopDependencies,
        context_store: ContextStore | None = None,
        memory: MemoryStore | None = None,
        renderer: TemplateRenderer | None = None,
        result_sink: ResultSink | None = None,
        output_parsers: Mapping[str, Callable[[str], Any]] | None = None,
    ) -> None:
        self.registry = dict(registry)
        self.deps = deps
        self.context_store = context_store
        self.memory = memory
        self.renderer = renderer
        self.result_sink = result_sink
        self.output_parsers = dict(output_parsers or {})

    # ── Task execution ──────────────────────────────────────────────────────

    def run_task(self, envelope: ContextEnvelope) -> TaskOutcome:
        """Execute a task's subtasks in dependency order.

        Stops descending a branch whose prerequisite failed: running a specialist
        whose input never arrived produces a confidently wrong answer, which is
        worse than an unfinished task.

        A subtask already :attr:`SubtaskStatus.COMPLETED` on entry is *not*
        re-executed — its output already sits on the blackboard and in
        ``context_data["results"]``, and satisfies its dependents exactly as it
        did the first time. A fresh task has no completed subtasks, so this
        changes nothing there; it is what lets a caller retry a task that
        failed partway through by resetting only the failed subtask and its dependents to ``PENDING``
        and calling this method again on the same envelope — upstream work is
        not redone, and is not re-billed.
        """
        blackboard = self._blackboard_for(envelope)
        outcome = TaskOutcome(envelope=envelope)
        order = topological_order(envelope.subtasks)
        satisfied: set[str] = {
            s.id for s in order if s.status == SubtaskStatus.COMPLETED
        }

        for subtask in order:
            if subtask.id in satisfied:
                continue

            unmet = [d for d in subtask.depends_on if d not in satisfied]
            if unmet:
                subtask.status = SubtaskStatus.SKIPPED
                envelope.flag_gap(
                    GapKind.EXTERNAL_DEPENDENCY,
                    f"subtask {subtask.id!r} skipped; unmet dependencies: {unmet}",
                    subtask_id=subtask.id,
                    blocking=False,
                )
                continue

            missing = self._missing_inputs(subtask, blackboard)
            if missing:
                # A declared input that is absent or unreadable. Running anyway
                # would invite the specialist to invent it.
                subtask.status = SubtaskStatus.SKIPPED
                envelope.flag_gap(
                    GapKind.MISSING_INPUT,
                    f"subtask {subtask.id!r} skipped; missing inputs: {missing}",
                    subtask_id=subtask.id,
                )
                continue

            run_outcome = self._execute_subtask(envelope, subtask, blackboard)

            if run_outcome.suspended:
                subtask.status = SubtaskStatus.BLOCKED
                self._park(envelope, subtask.id, run_outcome.run)
                outcome.awaiting_approval[subtask.id] = run_outcome.approval_id or ""
                continue

            result = run_outcome.result
            assert result is not None  # not suspended, so a result exists
            outcome.results.append(result)
            self._publish(envelope, blackboard, subtask, result)

            if result.succeeded:
                satisfied.add(subtask.id)
                envelope.record_result(subtask.id, result.output)
            else:
                subtask.status = SubtaskStatus.FAILED

        self._finalize(envelope, blackboard)
        return outcome

    def resume_task(
        self,
        envelope: ContextEnvelope,
        subtask_id: str,
        *,
        approved: bool,
        denial_reason: str | None = None,
    ) -> TaskOutcome:
        """Resume a subtask that was parked awaiting approval."""
        parked = envelope.extensions.get(SUSPENDED_RUNS_KEY, {})
        raw = parked.get(subtask_id)
        if raw is None:
            raise KeyError(f"no suspended run for subtask {subtask_id!r}")

        subtask = envelope.subtask(subtask_id)
        if subtask is None:
            raise KeyError(f"unknown subtask {subtask_id!r}")

        run = SpecialistRun.from_dict(raw)
        blackboard = self._blackboard_for(envelope)
        config = self._config_for(subtask.required_capability)
        loop = self._build_loop(envelope, config, subtask, blackboard, run.ancestry, run.depth)

        run_outcome = loop.resume(run, approved=approved, denial_reason=denial_reason)
        outcome = TaskOutcome(envelope=envelope)

        if run_outcome.suspended:
            self._park(envelope, subtask_id, run_outcome.run)
            outcome.awaiting_approval[subtask_id] = run_outcome.approval_id or ""
            return outcome

        parked.pop(subtask_id, None)
        result = run_outcome.result
        assert result is not None
        outcome.results.append(result)
        self._publish(envelope, blackboard, subtask, result)
        if result.succeeded:
            envelope.record_result(subtask_id, result.output)
        else:
            subtask.status = SubtaskStatus.FAILED

        self._finalize(envelope, blackboard)
        return outcome

    # ── Delegation ──────────────────────────────────────────────────────────

    def run_child(
        self,
        envelope: ContextEnvelope,
        blackboard: Blackboard,
        request: DelegationRequest,
    ) -> SpecialistResult:
        """Run a delegated specialist on the caller's stack.

        Every guard fires before the child is built, so a rejected delegation
        costs nothing.

        A delegate that suspends for approval returns a failed result rather than
        parking. Its parent is mid-flight on the stack and cannot itself be
        serialized, so there is nothing to resume into. Configure capabilities
        reachable by delegation with read-only tools, or with
        ``auto_approve_side_effects`` where that is genuinely intended.
        """
        parent = request.requesting_capability
        parent_config = self.registry.get(parent)
        allowed = list(parent_config.can_delegate_to) if parent_config else []
        if request.required_capability not in allowed:
            raise DelegationDenied(parent, request.required_capability, allowed)

        if request.required_capability in request.ancestry:
            raise DelegationCycle(list(request.ancestry), request.required_capability)

        if request.depth > envelope.budget.max_spawn_depth:
            raise DelegationDepthExceeded(
                request.depth, envelope.budget.max_spawn_depth
            )

        # An unregistered target is refused before anything is charged.
        config = self._config_for(request.required_capability)

        # Raises BudgetExceeded / DelegationDepthExceeded before any spend.
        envelope.budget.charge_spawn(request.depth)

        subtask = Subtask(
            required_capability=request.required_capability,
            parameters=dict(request.input_data),
            expected_output_schema=request.expected_output_schema,
            description=f"delegated by {parent}",
            idempotency_key=f"{envelope.task_id}:{request.request_id}",
        )
        envelope.subtasks.append(subtask)

        loop = self._build_loop(
            envelope,
            config,
            subtask,
            blackboard,
            ancestry=request.ancestry,
            depth=request.depth,
        )
        run_outcome = self._start(loop, envelope, config, subtask, blackboard)

        if run_outcome.suspended:
            self._park(envelope, subtask.id, run_outcome.run)
            return SpecialistResult(
                specialist_id=run_outcome.run.specialist_id,
                subtask_id=subtask.id,
                capability=config.capability,
                status=SpecialistStatus.AWAITING_APPROVAL,
                error=(
                    f"delegate {config.capability!r} needs human approval for tool "
                    f"{run_outcome.run.pending_approval.tool_name!r} and cannot "
                    "complete inline"
                ),
            )

        result = run_outcome.result
        assert result is not None
        self._publish(envelope, blackboard, subtask, result)
        if result.succeeded:
            subtask.status = SubtaskStatus.COMPLETED
        return result

    # ── Internals ───────────────────────────────────────────────────────────

    def _execute_subtask(
        self,
        envelope: ContextEnvelope,
        subtask: Subtask,
        blackboard: Blackboard,
    ) -> RunOutcome:
        config = self._config_for(subtask.required_capability)
        subtask.status = SubtaskStatus.RUNNING
        loop = self._build_loop(envelope, config, subtask, blackboard, (), 0)
        return self._start(loop, envelope, config, subtask, blackboard)

    def _start(
        self,
        loop: AgenticLoop,
        envelope: ContextEnvelope,
        config: SpecialistConfig,
        subtask: Subtask,
        blackboard: Blackboard,
    ) -> RunOutcome:
        prompt = self._render_prompt(envelope, config, subtask, blackboard)
        try:
            return loop.start(prompt)
        except (BudgetExceeded, SpecialistTimeout) as exc:
            return RunOutcome(
                run=SpecialistRun(
                    specialist_id="",
                    capability=config.capability,
                    subtask_id=subtask.id,
                    task_id=envelope.task_id,
                    status=SpecialistStatus.FAILED,
                ),
                result=SpecialistResult(
                    specialist_id="",
                    subtask_id=subtask.id,
                    capability=config.capability,
                    status=SpecialistStatus.FAILED,
                    error=str(exc),
                ),
            )

    def _build_loop(
        self,
        envelope: ContextEnvelope,
        config: SpecialistConfig,
        subtask: Subtask,
        blackboard: Blackboard,
        ancestry: Sequence[str],
        depth: int,
    ) -> AgenticLoop:
        context = envelope.slice_for(config, depth=depth, ancestry=ancestry)

        # The delegate hook is bound per-loop so the child inherits this task's
        # envelope, blackboard and budget without the specialist ever seeing
        # them directly.
        deps = LoopDependencies(
            llm=self.deps.llm,
            tools=self.deps.tools,
            approvals=self.deps.approvals,
            meter=self.deps.meter,
            validator=self.deps.validator,
            clock=self.deps.clock,
            delegate=lambda request: self.run_child(envelope, blackboard, request),
        )

        if config.capability not in envelope.agent_chain:
            envelope.agent_chain.append(config.capability)

        return AgenticLoop(
            config=config,
            context=context,
            subtask=subtask,
            blackboard=blackboard,
            budget=envelope.budget,
            deps=deps,
            output_parser=self.output_parsers.get(config.capability),
        )

    def _config_for(self, capability: str) -> SpecialistConfig:
        config = self.registry.get(capability)
        if config is None:
            raise CapabilityNotRegistered(capability, list(self.registry))
        return config

    def _blackboard_for(self, envelope: ContextEnvelope) -> Blackboard:
        """Build a blackboard, rehydrating any persisted entries.

        One blackboard per task, always. Sharing one across tasks would leak
        facts between unrelated work, and a stored "current blackboard" on the
        orchestrator would break the moment two tasks ran at once.
        """
        append = None
        if self.context_store is not None:
            store = self.context_store
            task_id = envelope.task_id
            append = lambda entry: store.append_blackboard(task_id, entry)  # noqa: E731
        return Blackboard(
            task_id=envelope.task_id,
            tenant_id=envelope.tenant_id,
            entries=envelope.blackboard_snapshot,
            on_append=append,
        )

    def _render_prompt(
        self,
        envelope: ContextEnvelope,
        config: SpecialistConfig,
        subtask: Subtask,
        blackboard: Blackboard,
    ) -> str:
        variables = {
            "context": envelope.slice_for(config).to_dict(),
            "subtask": subtask.to_dict(),
            "blackboard": blackboard.read_all(config.capability),
            "memory": self._recall(envelope, config, subtask),
            "capability": config.capability,
            "tools": list(config.tools),
            "can_delegate_to": list(config.can_delegate_to),
            "output_schema": subtask.expected_output_schema or config.output_schema,
        }
        if self.renderer is not None and config.prompt_template:
            return self.renderer.render(config.prompt_template, variables)
        return _default_prompt(variables)

    @staticmethod
    def _missing_inputs(subtask: Subtask, blackboard: Blackboard) -> list[str]:
        """Declared inputs the subtask's capability cannot currently read."""
        return [
            addr
            for addr in subtask.consumes
            if blackboard.read_address(addr, subtask.required_capability) is None
        ]

    @staticmethod
    def _provenance(subtask: Subtask, blackboard: Blackboard) -> list[int]:
        """``seq`` of each declared input, as the subtask's capability sees it."""
        seqs: list[int] = []
        for addr in subtask.consumes:
            entry = blackboard.read_address(addr, subtask.required_capability)
            if entry is not None:
                seqs.append(entry.seq)
        return seqs

    def _recall(
        self,
        envelope: ContextEnvelope,
        config: SpecialistConfig,
        subtask: Subtask,
    ) -> list[dict[str, Any]]:
        if self.memory is None:
            return []
        query = subtask.description or envelope.intent or config.capability
        try:
            return self.memory.recall(
                principal_id=envelope.principal_id,
                capability=config.capability,
                query=query,
            )
        except Exception:
            # Memory is an enhancement. Losing recall degrades answer quality;
            # failing the task over it would be worse.
            return []

    def _publish(
        self,
        envelope: ContextEnvelope,
        blackboard: Blackboard,
        subtask: Subtask,
        result: SpecialistResult,
    ) -> None:
        """Post a successful result to the blackboard and consolidate memory.

        Each specialist consolidates only its *own* summary. Writing the whole
        blackboard per specialist, as earlier drafts did, makes memory writes
        quadratic in the number of specialists and stores the same facts N times.
        The blackboard is consolidated once, by :meth:`_finalize`.
        """
        if self.result_sink is not None:
            try:
                self.result_sink.on_result(result)
            except Exception:
                pass  # observability must never fail a task

        if not result.succeeded or result.output is None:
            return

        config = self.registry.get(result.capability)
        blackboard.post(
            specialist_id=result.specialist_id,
            capability=result.capability,
            key="output",
            value=result.output,
            tenant_id=envelope.tenant_id,
            visible_to=config.output_visible_to if config else None,
            derived_from=self._provenance(subtask, blackboard),
        )

        if self.memory is not None:
            try:
                self.memory.consolidate(
                    principal_id=envelope.principal_id,
                    capability=result.capability,
                    summary=str(result.output)[:2000],
                    metadata={
                        "task_id": envelope.task_id,
                        "subtask_id": subtask.id,
                        "confidence": result.confidence,
                        "specialist_id": result.specialist_id,
                    },
                )
            except Exception:
                pass

    def _park(
        self, envelope: ContextEnvelope, subtask_id: str, run: SpecialistRun
    ) -> None:
        """Persist a suspended run so it can be resumed after approval."""
        envelope.extensions.setdefault(SUSPENDED_RUNS_KEY, {})[subtask_id] = (
            run.to_dict()
        )
        envelope.touch()

    def _finalize(self, envelope: ContextEnvelope, blackboard: Blackboard) -> None:
        envelope.blackboard_snapshot = blackboard.snapshot()
        envelope.touch()
        if self.context_store is not None:
            self.context_store.save(envelope)


# ── Registry merging ────────────────────────────────────────────────────────


def merge_registries(
    *registries: Mapping[str, SpecialistConfig],
) -> dict[str, SpecialistConfig]:
    """Merge specialist registries, refusing silent overrides.

    The same capability appearing in several registries is fine when the configs
    are identical (a resumed run rebuilding a built-in it already had). When the
    configs differ, one side would win and the other side's tool and delegation
    grants would vanish without a trace, so every conflicting name is collected
    and :class:`CapabilityCollision` is raised.
    """
    merged: dict[str, SpecialistConfig] = {}
    conflicts: list[str] = []
    for registry in registries:
        for capability, config in registry.items():
            existing = merged.get(capability)
            if existing is not None and existing != config:
                conflicts.append(capability)
                continue
            merged[capability] = config
    if conflicts:
        raise CapabilityCollision(sorted(set(conflicts)))
    return merged


# ── Dependency ordering ─────────────────────────────────────────────────────


def topological_order(subtasks: Sequence[Subtask]) -> list[Subtask]:
    """Order subtasks so every dependency precedes its dependents.

    Ties are broken by the original list order, which keeps runs deterministic
    and reproducible. Grouping by dependency layer is also what would make
    parallel execution safe later, since the blackboard is already synchronized.

    Raises:
        UnresolvableDependencies: On a cycle or a reference to an unknown
            subtask. Both are authoring bugs and must surface loudly; a task
            that silently drops half its graph looks like it succeeded.
    """
    by_id = {s.id: s for s in subtasks}
    for subtask in subtasks:
        unknown = [d for d in subtask.depends_on if d not in by_id]
        if unknown:
            raise UnresolvableDependencies(
                f"subtask {subtask.id!r} depends on unknown subtasks: {unknown}",
                unknown,
            )

    ordered: list[Subtask] = []
    placed: set[str] = set()
    remaining = list(subtasks)

    while remaining:
        ready = [s for s in remaining if all(d in placed for d in s.depends_on)]
        if not ready:
            stuck = sorted(s.id for s in remaining)
            raise UnresolvableDependencies(
                f"dependency cycle among subtasks: {stuck}", stuck
            )
        for subtask in ready:
            ordered.append(subtask)
            placed.add(subtask.id)
        remaining = [s for s in remaining if s.id not in placed]

    return ordered


def _default_prompt(variables: Mapping[str, Any]) -> str:
    """Minimal built-in prompt, so the package is usable with no renderer.

    Hosts are expected to supply a :class:`TemplateRenderer` for anything real;
    this exists to keep the core dependency-free and testable.
    """
    lines = [
        f"You are the '{variables['capability']}' specialist.",
        "Complete only your assigned subtask. Do not attempt work outside it.",
        "",
        f"Intent: {variables['context'].get('intent', '')}",
        f"Subtask parameters: {variables['subtask'].get('parameters', {})}",
    ]
    if variables["blackboard"]:
        lines += ["", "Facts posted by other specialists on this task:"]
        lines += [f"- {k}: {v}" for k, v in variables["blackboard"].items()]
    if variables["memory"]:
        lines += ["", "Relevant prior work:"]
        lines += [f"- {item}" for item in variables["memory"]]
    if variables["can_delegate_to"]:
        lines += [
            "",
            "You may delegate to: " + ", ".join(variables["can_delegate_to"]),
        ]
    if variables["output_schema"]:
        lines += [
            "",
            "Return output matching this JSON Schema exactly:",
            str(variables["output_schema"]),
        ]
    return "\n".join(lines)
