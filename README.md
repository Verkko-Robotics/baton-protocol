# BATON — Bounded Authority for Traceable Orchestrated Networks

A vendor-neutral standard for building agent systems where every unit of work is
**scoped**, every interaction between agents is **recorded**, and every action
that touches the outside world is **authorized and metered**.

BATON is not a framework you build your product inside. It is a protocol plus a
reference implementation, with a hard rule: **the core performs no I/O**. It has
no dependency on any database, HTTP client, or model provider. Everything
external is an abstract interface you implement against your own stack.

```
pip install baton-protocol
```

## Why this exists

Most agent frameworks let agents call each other directly. That works until you
need to answer a question someone will eventually ask you: *why did the system
send that email?* Direct calls leave no trace, so the answer has to be
reconstructed from logs, if it can be reconstructed at all.

BATON makes the trace structural rather than optional. Specialists cannot reach one
another, so every piece of data that moved between them is a durable, attributed,
ordered blackboard entry, and every request for another capability is a recorded
delegation. Auditability is a consequence of the architecture, not a logging
discipline someone has to maintain.

## Core concepts

| Concept | What it is |
|---|---|
| `ContextEnvelope` | The versioned, persistent record of one task: intent, decomposition, known gaps, shared facts, spend |
| `ContextSlice` | The bounded view of that context a single specialist receives |
| `Blackboard` | Append-only, visibility-controlled fact store — the only channel between specialists |
| `Orchestrator` | The sole constructor of specialists; owns authority and dependency ordering |
| `AgenticLoop` | Suspendable tool-calling loop that can pause for human approval and resume from storage |
| `TaskBudget` | Token, tool-call, spawn and depth ceilings shared across a whole task tree |

## The invariants

These are enforced in code, not documented as guidance.

1. A specialist never holds a reference to another specialist.
2. All data between specialists flows through the blackboard.
3. Delegation is restricted to declared capabilities, is never cyclic, and never exceeds the depth limit.
4. Every specialist receives a context slice, never the full envelope.
5. Token, tool-call and spawn budgets are shared across the task, not per specialist.
6. Side-effecting tools pause for human approval, mid-loop and resumably.
7. A specialist's usable tools are its declared set intersected with what the principal actually authorized.
8. Every model call is metered against the principal, with idempotency keys.
9. A blackboard belongs to exactly one task and one tenant.
10. Output is validated against its declared schema before it becomes visible to anyone.
11. Blackboard and run state survive process restarts.

## Quick start

```python
from baton_protocol import (
    ContextEnvelope, LoopDependencies, Orchestrator, SpecialistConfig, Subtask,
)

registry = {
    "screener": SpecialistConfig(
        capability="screener",
        tools=("cv_parse",),
        can_delegate_to=("researcher",),
        context_keys=("job_title", "required_skills"),
        model="gpt-4o",
        output_schema={
            "type": "object",
            "required": ["candidate_name", "score"],
            "properties": {
                "candidate_name": {"type": "string"},
                "score": {"type": "integer", "minimum": 0, "maximum": 100},
            },
        },
    ),
}

deps = LoopDependencies(
    llm=MyLLMGateway(),          # implement baton_protocol.LLMGateway
    tools=MyToolGateway(),       # implement baton_protocol.ToolGateway
    approvals=MyApprovals(),     # implement baton_protocol.ApprovalGateway
    meter=MyUsageMeter(),        # implement baton_protocol.UsageMeter
)

envelope = ContextEnvelope(
    task_id="task-1",
    tenant_id="acme",
    principal_id="user-1",
    intent="Screen the shortlist",
    context_data={"job_title": "Engineer", "required_skills": ["python"]},
    subtasks=[Subtask(required_capability="screener", parameters={"cv_text": "..."})],
)

outcome = Orchestrator(registry=registry, deps=deps).run_task(envelope)
```

## Human approval, and why the loop looks unusual

A side-effecting tool call may wait hours for a person to authorize it — far
longer than any HTTP request or worker thread lives. So the loop's position is
never held on the Python stack. It lives in a `SpecialistRun`, which serializes
to JSON:

```python
outcome = orchestrator.run_task(envelope)

if outcome.suspended:
    store.save(envelope)               # the parked run travels with the envelope
    # ... hours later, once a human decides ...
    envelope = store.load("task-1")
    outcome = orchestrator.resume_task(envelope, subtask_id, approved=True)
```

A denial is not a failure. It is fed back to the specialist as a tool result, so
it can revise its approach or finish without the action.

## Implementing the interfaces

| Interface | Responsibility |
|---|---|
| `LLMGateway` | Normalize your provider's response into `LLMResponse` |
| `ToolGateway` | Advertise the principal's authorized tools; execute them idempotently |
| `ApprovalGateway` | Record an approval request and report its status |
| `UsageMeter` | Record token spend against the principal |
| `ContextStore` | Persist envelopes and append blackboard entries |
| `MemoryStore` | Consolidate and recall long-term memory |
| `TemplateRenderer` | Render prompt templates (optional) |
| `SchemaValidator` | Validate output against JSON Schema (optional) |

`baton_protocol.testing` ships in-memory implementations of all of them. They
define the expected semantics more precisely than prose can, and they let you
test your own adapters against the same behaviour.

## Versioning

`PROTOCOL_VERSION` versions the wire format; `__version__` versions this library.
A minor bump only ever adds fields, and readers preserve fields they do not
recognize, so a v1.0 service can round-trip a v1.4 envelope without losing data.
That is what makes it safe to upgrade one service at a time.

The JSON Schemas under `baton_protocol/schemas/` are the normative definition. This
Python package is the reference implementation, not the standard itself.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

Apache-2.0.
