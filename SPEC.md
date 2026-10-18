# BATON Specification

**Version 1.0.0 (draft)** · Bounded Authority for Traceable Orchestrated Networks

BATON defines how agents share context, hand work to one another, and act on the
outside world, so that every piece of information that moved between agents is
scoped, attributed, ordered and recorded.

This document is normative. The key words MUST, MUST NOT, SHOULD and MAY are used
as in RFC 2119. Where this document and the reference library disagree, the
disagreement is a bug in one of them and is listed in [Known gaps](#9-known-gaps).

The machine-readable definition of every wire type is the set of JSON Schemas in
[`src/baton_protocol/schemas/`](src/baton_protocol/schemas). The executable
definition of the behaviour is the set of conformance vectors in
[`conformance/vectors/`](conformance/vectors).

---

## 1. Scope

BATON specifies:

- the data model for a **task** and its context (§3);
- how context **flows** between agents (§4);
- what an agent is **allowed to do**, and how humans gate it (§5);
- how a task **pauses and resumes** (§6);
- how the format **evolves** (§7).

BATON does not specify: model providers, storage engines, prompt wording, user
interfaces, or the transport used between processes. Those are supplied by the
implementer through interfaces (see the reference library's `interfaces` module).
The core MUST perform no I/O.

## 2. Terminology

| Term | Meaning |
|---|---|
| **Task** | One goal, decomposed into subtasks, executed under one tenant for one principal. |
| **Tenant** | The owner boundary. An opaque string; never interpreted. |
| **Principal** | The identity the task acts for. An opaque string; never interpreted. |
| **Capability** | A named kind of work, and the unit of authority. |
| **Specialist** | One running instance of a capability. |
| **Orchestrator** | The only party that constructs specialists and mediates between them. |
| **Envelope** | The persistent record of a task. |
| **Slice** | The bounded view of the envelope given to one specialist. |
| **Blackboard** | The append-only fact store scoped to one task. |

## 3. Data model

Every wire type is a JSON object. Identifiers are opaque strings. Timestamps are
ISO 8601 in UTC with an explicit offset. A producer MUST omit a field rather than
send `null`; a reader MUST treat an absent field and a `null` field identically.

| Type | Schema | Purpose |
|---|---|---|
| ContextEnvelope | `envelope` | The persistent, versioned record of one task |
| Subtask | `subtask` | One unit of work for one capability |
| GapFlag | `gap-flag` | A recorded hole in the task's information or authority |
| BlackboardEntry | `blackboard-entry` | One immutable fact |
| ContextSlice | `slice` | What one specialist is allowed to see |
| DelegationRequest | `delegation-request` | One specialist's request for another capability's work |
| SpecialistRun | `specialist-run` | Serializable state of a paused or running specialist |
| PendingApproval | `pending-approval` | A side-effecting call frozen awaiting a human |
| SpecialistResult | `specialist-result` | The outcome of one specialist run |
| TaskBudget | `budget` | Shared resource ceilings and usage for a task tree |

### 3.1 The envelope

An envelope MUST carry `task_id`, `tenant_id` and `principal_id`. It carries the
goal (`intent`), the shared input (`context_data`), the subtask graph, recorded
gaps, the capabilities that have touched the task (`agent_chain`), a snapshot of
the blackboard, and a `revision` that increases on every mutation. Implementations
MAY use `revision` for optimistic concurrency.

`extensions` is a namespace for host-specific data that is not part of the
protocol. A reader MUST preserve it.

## 4. Context flow

### 4.1 The two channels

Specialists MUST NOT hold a reference to one another. Information crosses from one
capability to another in exactly two ways, and both are recorded:

1. **A blackboard post** (§4.3), which carries data.
2. **A delegation** (§5.2), which carries a request for a capability.

There is no third channel. The orchestrator is the only party that constructs
specialists, which is what makes this enforceable.

### 4.2 Slicing

A specialist MUST NOT receive the full envelope. It receives a **slice** made of:

- identity that always travels: `task_id`, `tenant_id`, `principal_id`,
  `session_id`, `intent`, and its own `capability`;
- `data`: only the keys of `context_data` that the capability declared in
  `context_keys`. A declared key the envelope does not hold is omitted, not set to
  null. An undeclared key MUST NOT appear anywhere in the slice;
- `depth` and `ancestry`, used for delegation checks and attribution.

Slicing is a containment boundary, not an optimisation.
*Vectors: `slicing::*`.*

### 4.3 The blackboard

A blackboard belongs to exactly one task and one tenant.

- **Append-only.** Entries are immutable and are never deleted. Correcting a fact
  is a new post.
- **Addressed.** A fact is addressed as `capability.key`. `capability` and `key`
  MUST be non-empty.
- **Ordered.** Every post gets a strictly increasing `seq`. Wall-clock time MUST
  NOT be used to decide which post is latest.
- **Latest wins.** A read returns the highest-`seq` entry the requester is
  permitted to read. Superseded entries remain in history.
- **Fail-closed visibility.** `visible_to` lists the capabilities that may read an
  entry; `["*"]` means all. An empty list means **nobody** except the author, and
  MUST be honoured literally. An absent `visible_to` means all.
- **Unreadable equals absent.** A read of an entry the requester may not see MUST
  return the same result as a read of an entry that does not exist.
- **Tenant-scoped.** A post made under a different tenant MUST raise an error. It
  MUST NOT be silently dropped.
- **Authors can always read their own entries.**

The blackboard's complete snapshot ignores visibility and is for the orchestrator
and for audit only. It MUST NOT be exposed to a specialist.
*Vectors: `blackboard::*`.*

### 4.4 Subtasks and ordering

A subtask names one capability and the subtasks it `depends_on`. The orchestrator
MUST run a subtask only after everything it depends on has completed.

- Order is by dependency layer; independent subtasks keep their authored order, so
  that runs are reproducible.
- A dependency cycle (including a subtask that depends on itself) or a reference to
  an unknown subtask is an authoring error. It MUST be raised, never silently
  dropped.
- If a prerequisite fails, its dependents MUST NOT run. They are marked `skipped`
  and a non-blocking gap of kind `external_dependency` is recorded. Running a
  specialist whose input never arrived produces a confidently wrong answer, which
  is worse than an unfinished task.
- A subtask already `completed` MUST NOT be executed again when the task is re-run.
  Its output is already on the blackboard.

*Vectors: `ordering::*`, `flow::flow-failed-prerequisite-skips-dependents`.*

### 4.4.1 Declared inputs

A subtask MAY list the blackboard addresses it needs in `consumes`. Before running
it, the orchestrator MUST check that each address has an entry the subtask's
capability can read. If any does not, the subtask MUST NOT run: it is marked
`skipped` and a blocking gap of kind `missing_input` is recorded. An entry the
capability may not see counts as absent (§4.3). A subtask with no `consumes`
behaves exactly as before.

Declaring inputs turns "the specialist noticed something was missing" into "the
protocol knew before it spent anything".
*Vectors: `flow::flow-missing-declared-input-*`, `flow::flow-unreadable-declared-input-*`.*

### 4.4.2 Provenance

When a specialist's output is published, the entry MUST record in `derived_from`
the `seq` of each entry the subtask declared in `consumes`, as that capability saw
them at publication. An auditor can then walk from any conclusion back to the facts
it rested on. A subtask with no `consumes` produces an entry with no `derived_from`.

`derived_from` records what was **declared**, not what the model actually used; it
is an upper bound on influence, not proof of it.
*Vectors: `flow::flow-satisfied-input-runs-and-is-recorded-as-provenance`, `roundtrip::rt-blackboard-entry-derived-from`.*

### 4.5 Publishing results

When a specialist succeeds, the orchestrator posts its output to the blackboard
under the key `output` for that capability, with visibility set from the
capability's `output_visible_to` (absent means all), and records it in
`context_data.results` under the subtask id.

A result counts as successful when its status is `completed` or `truncated`. A
truncated run (it reached its iteration ceiling) returns its last text as
degraded output rather than discarding the work.
*Vectors: `flow::flow-output-is-published-with-declared-readers`.*

### 4.6 Output contracts

If a subtask or capability declares an output schema, the output MUST be validated
against it **before** it becomes visible. A violation is returned to the specialist
for correction, up to a bounded number of attempts; after that the run fails and
nothing is published. An implementation without a validator MUST say so; it MUST
NOT pretend the contract held.
*Vectors: `flow::flow-invalid-output-*`, `flow::flow-output-that-never-validates-*`.*

### 4.7 Gaps

When a task cannot proceed on the information it has, it SHOULD record a gap
(`missing_input`, `missing_capability`, `missing_authorization`,
`ambiguous_intent`, `external_dependency`) rather than invent an answer. A gap is
`blocking` or not, and `resolved` or not. The envelope exposes the unresolved
blocking gaps; what a host does with them is not yet specified (see §10).

### 4.8 Memory

Long-term memory is a separate channel from slices and the blackboard. Recall is
scoped to a principal and a capability. Memory MUST be treated as an enhancement:
a failure to recall MUST NOT fail the task.

## 5. Authority

### 5.1 Tools

A specialist's usable tools are the intersection of the tools its capability
declares and the tools the principal has actually authorized. A tool it did not
declare MUST be refused even if the principal could use it. A declaration of `*`
grants everything the principal has authorized.

Each tool is classified `side_effect`. An unclassified tool MUST be treated as
side-effecting.
*Vectors: `flow::flow-undeclared-tool-is-refused`, `flow::flow-read-only-tool-runs-without-approval`.*

### 5.2 Delegation

A capability may delegate only to capabilities it declares in `can_delegate_to`.
Delegation transitively confers every tool the delegate holds, so this grant is as
important as the tool grant. The orchestrator MUST check, in order, **before
creating the delegate or committing any resource**:

1. the requester is permitted to delegate to the target;
2. the target is not already in the ancestry (no cycles);
3. the delegation depth does not exceed the task's depth limit;
4. the task's spawn ceiling is not exceeded.

A refused delegation MUST NOT consume any resource.
*Vectors: `delegation::*`.*

A delegate that needs human approval cannot be paused (its parent is mid-flight)
and MUST return a failed result. Capabilities reachable by delegation SHOULD hold
read-only tools.

### 5.3 Approval

A call to a side-effecting tool MUST pause the specialist for human approval and
MUST NOT execute until approved. A **denial is not a failure**: it is returned to
the specialist as the result of the call, so the specialist can revise its
approach or finish without the action.
*Vectors: `flow::flow-side-effect-suspends-*`, `flow::flow-approval-then-resume-*`, `flow::flow-denial-is-not-a-failure`.*

### 5.4 Idempotency

Every tool call SHOULD carry an idempotency key that is stable across retries of
the same subtask and distinct for each distinct call (derived from the subtask,
the capability, the tool and the arguments), so a replayed task does not repeat a
side effect.

### 5.5 Metering

Every model call MUST be recorded against the principal with an idempotency key
that is unique per call.
*Vectors: `flow::flow-every-model-call-is-metered`.*

### 5.6 Resource ceilings

A task carries ceilings shared by every specialist in its tree, including those
created by delegation: total tokens, total tool calls, total spawns and maximum
delegation depth. They are properties of the task, not of a specialist, because a
per-specialist limit does nothing about a fan-out of specialists that each stay
under their own. Exceeding a ceiling fails the task rather than degrading quietly.

<<<<<<< HEAD
=======
A token ceiling of **zero means no ceiling**, both for a task (`max_total_tokens`)
and for a specialist (`max_tokens`). A host that prefers never to cut work off
mid-task can set zero and rely on the other ceilings (iterations, tool calls,
spawns, depth), which bound a runaway loop without bounding useful work.

>>>>>>> 4d653be (BATON 1.0: specification, reference library, JSON Schemas and conformance suite)
## 6. Execution and durability

A pause for approval can outlast any request or process, so a run's position MUST
NOT live on a call stack. It lives in a **SpecialistRun**, which serializes to JSON
and contains the message history, iteration count and the pending call. Pausing
means persisting the run (parked in the envelope's `extensions`); resuming means
loading it and continuing from the unanswered call.

A task MUST survive a process restart: an envelope serialized and read back MUST
resume to the same outcome.
*Vectors: `flow::flow-approval-then-resume-executes`, `roundtrip::*`.*

## 7. Versioning and compatibility

`protocol_version` is `MAJOR.MINOR.PATCH`.

- **MAJOR:** a field was removed or its meaning changed. Not backward compatible.
- **MINOR:** a field was added. Older readers MUST ignore it and MUST preserve it
  on round-trip.
- **PATCH:** clarification only.

A reader accepts any version whose major is between 1 and its own. A reader MUST
refuse a newer major, and MUST refuse a malformed version string. A producer MUST
stamp the version it writes.

**Preservation.** Every wire type MUST round-trip fields it does not recognize,
untouched. A reader that drops them destroys data when an older service sits
between two newer ones. A known field MUST win over an unknown field of the same
name.
*Vectors: `roundtrip::*`, `version::*`.*

## 8. Conformance

An implementation conforms to BATON 1.0 if it:

1. reads and writes documents that validate against the schemas in
   `src/baton_protocol/schemas/`;
2. passes every vector in `conformance/vectors/`.

### 8.1 Vector format

Each file in `conformance/vectors/` is a suite:

```json
{ "suite": "slicing", "operation": "slice", "spec": "SPEC.md#4.2", "vectors": [ ... ] }
```

`operation` selects how the vectors in the file are interpreted: `slice`,
`blackboard`, `roundtrip`, `version`, `ordering`, `delegation` or `flow`. Every
vector has a unique `id` and a `description` stating the rule it checks. The
Python runner (`tests/conformance_runner.py`) shows the exact reading of each
operation; another language's runner should reproduce it.

A vector with a `known_gap` field marks a place the reference implementation does
not yet meet the specification.

### 8.2 Rule coverage

| # | Invariant | Checked by |
|---|---|---|
| 1 | A specialist never holds a reference to another | Structural (§4.1); not testable as data |
| 2 | All data between specialists flows through the blackboard | Structural; `flow::*output-is-published*` |
| 3 | Delegation is declared, acyclic and depth-limited | `delegation::*` |
| 4 | Every specialist gets a slice, never the envelope | `slicing::*` |
| 5 | Resource ceilings are shared across the task | `delegation::deleg-*-limit-*` (spawn, depth); token and tool-call sharing covered by the library's own tests, vector pending |
| 6 | Side-effecting tools pause for approval, resumably | `flow::*suspends*`, `*approval-then-resume*`, `*denial*` |
| 7 | Usable tools = declared ∩ authorized | `flow::flow-undeclared-tool-is-refused` |
| 8 | Every model call is metered with idempotency keys | `flow::flow-every-model-call-is-metered` |
| 9 | A blackboard belongs to one task and one tenant | `blackboard::bb-cross-tenant-post-rejected` |
| 10 | Output is validated before it is visible | `flow::flow-invalid-output-*`, `*never-validates*` |
| 11 | State survives restarts | `flow::*approval-then-resume*`, `roundtrip::*` |

## 9. Known gaps

None. The reference library meets every rule in this document, and every vector
passes. A vector carrying a `known_gap` field marks a rule the implementation does
not yet meet, and is expected to fail until fixed.

Two gaps found while writing this specification were fixed before 1.0:
`SpecialistRun` now preserves unknown fields (§7), and a delegation to an
unregistered target is refused before any spawn is charged (§5.2).

## 10. Open questions for 1.x

Behaviours the reference implements today that this version records but has not
yet decided to standardise:

- **Supersession and visibility.** If a newer post is hidden from a reader, the
  reader sees the newest post it *can* read, which may be an older, superseded
  value (`blackboard::bb-superseding-post-hidden-from-reader-leaves-older-value`).
  Whether a hidden correction should instead hide the address entirely is
  undecided.
- **Capability configuration is not on the wire.** `SpecialistConfig` (tools,
  delegation grants, context keys, output visibility) has no schema; it is
  declared by the host. A remote or cross-implementation orchestrator needs one.
- **No transport.** Message framing, remote specialists and cross-process
  authority are not yet specified.
- **Blocking gaps are not enforced.** The orchestrator records gaps and exposes
  `blocking_gaps`, but does not halt a task on one. Whether it should is undecided.
- **Planning-time handoff checking.** `consumes` is checked at run time. The
  orchestrator does not yet verify before execution that some earlier subtask
  produces each declared address.
- **Named outputs.** A specialist still publishes exactly one entry, `output`.
  Multiple named outputs, and `produces` declarations to check against, are not
  yet specified.
- **Version stamp.** `consumes` and `derived_from` are additive and optional, so
  they are a 1.x minor change, but the stamped `protocol_version` has not been
  bumped.
