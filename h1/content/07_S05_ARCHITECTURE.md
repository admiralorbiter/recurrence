# S05 · The State Machine
## Turning a memory snapshot into a persistent process

S04 hands S05 a typed state object.

S05 has to build everything around it.

The central question is:

> **Can the system maintain an event stream, working memory, goals, provenance, and bounded state over time without uncontrolled drift?**

## The Level-1 loop

```text
logical clock advances
        ↓
events scheduled for tick t are released
        ↓
updater proposes a state transition
        ↓
StateManager validates and applies the transition
        ↓
immutable audit log records what happened
        ↓
next tick
```

## The components

### SimulatedClock

A discrete logical clock separates experimental time from wall-clock time.

This makes it possible to create:

- active ticks with incoming events;
- quiet ticks with no new event;
- sparse 100-tick scenarios;
- precisely timed goal interruptions and resumptions.

### EventQueue

Events are dispatched in deterministic order.

This matters because replay later depends on the exact sequence.

### ImmutableEventLog

The event log is append-only and hash chained.

The system can verify that history has not silently changed.

### StateManager

The manager enforces:

- working-memory capacity;
- recency order;
- LRU eviction;
- legal goal transitions;
- state versions;
- audit snapshots.

### StructuredSelfState

The state carries:

- `working_memory`;
- goals and statuses;
- source ledger;
- unresolved items;
- update metadata.

### Updaters

S05 tests three relevant paths:

1. **Deterministic/oracle updater** — exact ground-truth transition baseline.
2. **Model delta updater** — the model emits only changed fields.
3. **Model full-state updater** — the model rewrites the entire state each active tick.

## Why include both model update styles?

Full-state rewriting risks deleting anything the model forgets to regenerate.

Delta writing risks preserving a bad update forever once it enters the state.

That creates a real tradeoff:

- forgetting can remove errors;
- persistence can inherit errors.

## The six scenarios

S05 does not rely on one toy stream.

It includes:

- three standard 15-tick scenarios;
- a 16-tick full goal-lifecycle scenario;
- a 28-tick capacity-overflow scenario with 24 entities and `Kmax = 16`;
- a 100-tick sparse-event stress scenario.

Across each condition:

- 189 logical ticks;
- 69 active inference ticks;
- 120 quiet ticks;
- 756 total evaluated ticks across conditions.

## Quiet ticks in S05

A critical detail:

S05 quiet ticks are deterministic identity no-ops.

They use:

- zero model calls;
- zero prompt tokens;
- zero latency;
- no state change except clock bookkeeping.

They demonstrate **scaffold stability**, not hidden thought.

This distinction will matter when S07 asks a different question about actual null-interval computation.

<div class="interactive-lab" data-widget="s05-loop">
<div class="kicker">Architecture viewer</div>
<h2>Follow one event through the state machine</h2>
<div id="s05-loop"></div>
</div>
