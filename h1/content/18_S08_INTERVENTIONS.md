# S08 · Surgery, Clones, and Reconvergence
## The headline needs qualifications

“History wins” is useful shorthand.

It is not the full result.

S08 includes several interventions that reveal when structured state can matter.

## Surgical single-slot inversion

The experiment changes one target key in state:

```text
S_A[target] = value_red
        ↓ intervention
S_A'[target] = value_blue
```

The memory transcript remains `M_A`, which supports red.

A control key is left untouched.

### Results

- target follows injected state: **12.5%**;
- control slot remains correct: **93.8%**;
- joint local causal precision: **12.5%**.

The intervention is physically local in the data structure.

Behavior is also mostly local in the sense that the control survives.

But the counterfactual target write rarely overrides the conflicting history.

## State-only calibration

When state is the only relevant representation:

- target state allegiance reaches 75% for State A and 75% for State B;
- goal-state readout reaches 81.2% and 100%;
- empty-state calibration falls near the uninformed floor.

Therefore the model can read the state.

The S08 issue is not incapacity.

It is **authority under conflict**.

## Memory-only calibration

With only the transcript:

- target memory allegiance reaches 100%;
- control-key correctness remains high;
- goal readout is lower and more variable.

This echoes S04 and S06: raw chronological history is often an excellent reasoning surface.

## The full condition matrix

<details>
<summary>Open the disaggregated confirmatory condition table</summary>

| Condition | Order | Target state | Target memory | Goal state | Goal memory | Control correct |
|---|---|---:|---:|---:|---:|---:|
| Congruent A | memory first | 93.8% | 93.8% | 75.0% | 75.0% | 100.0% |
| Congruent A | state first | 100.0% | 100.0% | 62.5% | 62.5% | 100.0% |
| Congruent B | memory first | 93.8% | 93.8% | 100.0% | 100.0% | 93.8% |
| Congruent B | state first | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Conflict M_A + S_B | memory first | 6.2% | 87.5% | 100.0% | 0.0% | 93.8% |
| Conflict M_A + S_B | state first | 0.0% | 93.8% | 93.8% | 6.2% | 93.8% |
| Conflict M_B + S_A | memory first | 12.5% | 87.5% | 31.2% | 62.5% | 100.0% |
| Conflict M_B + S_A | state first | 6.2% | 93.8% | 6.2% | 81.2% | 100.0% |
| Reset M_A + S_0 | memory first | 0.0% | 100.0% | 0.0% | 37.5% | 93.8% |
| Surgical M_A + S_A' | memory first | 12.5% | 87.5% | 12.5% | 68.8% | 93.8% |
| State-only S_A | state only | 75.0% | — | 81.2% | — | 81.2% |
| State-only S_B | state only | 75.0% | — | 100.0% | — | 87.5% |
| Memory-only M_A | memory only | — | 100.0% | — | 62.5% | 87.5% |
| Memory-only M_B | memory only | — | 100.0% | — | 87.5% | 87.5% |

The allegiance columns deliberately mean different things under congruent and conflict conditions. They report whether the selected answer follows the corresponding channel, not ordinary world-truth accuracy in every row.

</details>

## The clone cross-swap exception

The clone experiment creates branch-specific values.

Branch A history contains `fork_A`.

State B introduces `fork_B`, which does **not** occur in Branch A history.

When State B is transplanted into Branch A:

- state allegiance reaches **75%**;
- memory allegiance is 25%.

This does not contradict the balanced twin result.

It reveals a condition under which state becomes influential:

> **The state contributes distinctive information that the episodic history does not contain.**

The calibrated conclusion is therefore:

> Structured state **can** steer behavior when it introduces distinctive information. It usually does not override a rich matched history that already contains both competing values.

## Clone and fork

Before branching:

- clone state hashes are equal.

After different continuation events:

- branch states diverge.

This creates clean lineage for later intervention.

## Reconvergence

A synchronizing event updates the relevant branch state to the same final value.

After state reconvergence:

- pairwise behavioral concordance is **93.8%**.

This supports the idea that explicit branch differences are behaviorally meaningful when present and mostly disappear once the relevant explicit states reconverge.

It does not prove no hidden lineage effect exists, because H1 has no persistent hidden recurrent channel to carry one.

That is exactly the H2 question.

## What S08 changes about the word “self-state”

A field named `StructuredSelfState` can be:

- readable;
- useful;
- editable;
- cloneable;
- causally active in some contexts;

without being the model's privileged current belief or authoritative inner perspective.

That distinction is the conceptual payoff of S08.

<div class="handoff-card">
<span>S08 → S09</span>
<strong>From “what controls behavior?” to “who owns the evidence?”</strong>
<p>If the declared current self-state is not authoritative when it conflicts with history, S09 must ask what the model actually treats as Self, Peer, Environment, or Observer.</p>
</div>
