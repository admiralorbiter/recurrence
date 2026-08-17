# S05 · What Changed
## The split gate

S05 does not produce one PASS or FAIL.

It produces two.

### Scaffold gate: PASS

The Level-1 architecture is:

- deterministic;
- auditable;
- bounded;
- recoverable;
- stable across quiet ticks;
- protected by legal goal transitions;
- traceable through an immutable event log.

### Model-autonomous maintenance gate: FAIL

Under the tested protocol, Qwen2.5-3B does not reliably maintain multi-slot state.

The model delta updater retains 13.2% at the scenario-macro level. The full rewrite retains 6.3% and reaches 0% terminal retention.

## Why the fallback matters scientifically

The roadmap anticipated failure:

> Use deterministic state-transition constraints or smaller state until the system is stable.

H1 activates that fallback.

From this point onward, the canonical Level-1 state is maintained deterministically.

That decision prevents later experiments from confusing:

- “persistence itself has no effect”

with:

- “the model forgot or corrupted the state before the test.”

## A permanent distinction

S05 creates a three-part decomposition:

### Representation capacity

Can the explicit schema hold the relevant information?

Yes.

### Transition reliability

Can a process update it correctly?

The deterministic process can. The model updater cannot.

### Behavioral use

Can the model later answer from the maintained state?

That remains an empirical question for S06–S08.

## The systems lesson

A persistent AI system needs at least two governance mechanisms:

1. **Write validity:** Is this proposed update grounded and correctly routed?
2. **Memory hygiene:** If an error enters, how is it corrected, downgraded, or removed?

Persistence without those mechanisms can turn one hallucination into a durable fact.

## Why S06 becomes necessary

By the end of S05, the project has a reliable state machine.

But a reliable state machine can still be conceptually trivial.

If its final state is just a deterministic function of the event log, then perhaps the system never needed to maintain it online. Perhaps the same state can be rebuilt at the end.

That is the replay challenge.

<div class="handoff-card">
<span>S05 → S06</span>
<strong>From stable persistence to temporal necessity</strong>
<p>Now that state corruption is controlled, the project can finally ask whether carrying state through time adds anything beyond replaying the same history later.</p>
</div>
