# S05 · The Failure Atlas
## Schema compliance is not semantic correctness

All active model transitions satisfy the JSON schema.

That sounds impressive until we inspect what the valid JSON contains.

| Updater | Schema validity | Macro retention | Micro retention | Terminal retention | Omission | Mutation | Phantom ticks / unique | Goal coherence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Deterministic | 100% | **100%** | **100%** | **100%** | 0% | 0% | 0 / 0 | **100%** |
| Model delta | 100% | **13.2%** | 9.0% | 11.1% | 80.6% | 6.2% | **452 / 45** | 42.8% |
| Full-state rewrite | 100% | **6.3%** | 5.4% | **0%** | 92.0% | 1.7% | 56 / 25 | 16.7% |

The state can be syntactically perfect and semantically devastated.

## Failure mode 1 · Omission

The full-state updater rewrites the world from scratch every active tick.

Anything omitted disappears.

By the terminal step, exact retention falls to 0%.

The delta updater omits fewer things because unmentioned fields remain in place—but macro omission is still 80.6% against the oracle state.

## Failure mode 2 · Category routing

Earlier scouts revealed that the model sometimes places factual assertions into prose-like unresolved lists rather than the typed working-memory dictionary.

This is not a malformed schema. It is a semantic slot-allocation failure.

## Failure mode 3 · Phantom state

The delta updater creates 45 unique never-seen keys, appearing across 452 evaluated tick instances.

Once a phantom enters persistent state, later ticks inherit it.

That is the **error inheritance phenomenon**:

> **Continuity preserves errors as well as truths.**

## Failure mode 4 · Goal-state confusion

The goal lifecycle includes transitions such as:

```text
pending → active → suspended → active → completed
```

The model attempts 14 illegal demotions, including active or suspended goals returning to pending.

StateManager rejects them.

This is a crucial architectural lesson: deterministic constraints can protect a model-maintained system from some classes of invalid transition, even when the model's proposed state is unreliable.

## Failure mode 5 · Broken memory looks cheap

Token cost per active inference:

- model delta: **848.5** prompt tokens;
- full-state rewrite: **338.4** prompt tokens.

The full rewrite appears cheaper partly because it forgets almost everything, causing its future prompts to shrink.

That means prompt size alone can reward catastrophic forgetting.

> **A memory system can look efficient because it has stopped remembering.**

## Capacity overflow

The state receives 24 entities with a maximum capacity of 16.

The deterministic manager:

- keeps the bound exactly at 16;
- evicts 8 least-recently-updated entries;
- updates recency only on explicit writes.

This is not a cognitive result, but it is an essential baseline. Later state comparisons are meaningless if the state manager itself is unstable.

## Quiet-span stability

Across the 100-tick sparse scenario, the deterministic scaffold remains stable during long idle spans.

Again, this proves:

- identity preservation;
- clock integrity;
- state-manager stability.

It does not prove internal consolidation.

<div class="interactive-lab" data-widget="s05-updater">
<div class="kicker">Failure explorer</div>
<h2>Compare updater behavior</h2>
<div id="s05-updater"></div>
</div>
