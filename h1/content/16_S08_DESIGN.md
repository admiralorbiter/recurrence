# S08 · State × Memory
## Does the declared current state actually govern behavior?

By S08, `StructuredSelfState` is:

- readable;
- stable when maintained deterministically;
- replayable from history;
- easy to reset, clone, and edit.

That still does not tell us whether the model treats it as authoritative.

A weak test would show the state by itself and observe that the model answers from it.

S08 does something harder.

## Matched twin worlds

Create World A and World B with the same structure and candidate vocabulary.

Example:

```text
World A
key_target → value_red
goal_beta → active

World B
key_target → value_blue
goal_beta → suspended
```

Both `value_red` and `value_blue` appear somewhere in both histories.

This prevents candidate familiarity from solving the conflict.

## The 2 × 2 matrix

| | State A | State B |
|---|---|---|
| **Memory A** | `M_A + S_A` congruent | `M_A + S_B` conflict |
| **Memory B** | `M_B + S_A` conflict | `M_B + S_B` congruent |

This lets the experiment estimate two causal effects.

### Change state while holding memory fixed

```text
M_A + S_A  versus  M_A + S_B
```

If behavior changes, the state has independent leverage beyond the history.

### Change memory while holding state fixed

```text
M_A + S_A  versus  M_B + S_A
```

If behavior changes, the episodic record has leverage beyond current state.

## State and memory allegiance

In a conflict trial, there are three meaningful outcomes:

- answer follows state;
- answer follows memory;
- answer follows neither / a foil.

This is better than ordinary accuracy because the intervention intentionally creates incompatible “correct” answers under two representations.

## Presentation-order control

The prompt is shown in both orders:

- Memory → State;
- State → Memory.

This checks whether simple recency explains which channel wins.

## The intervention battery

### Reset with memory preserved

Compare:

```text
M_A + S_A
M_A + S_empty
```

If reset causes no impairment, the transcript compensates for the removed state.

### Surgical slot inversion

Change exactly one target binding in state while leaving a control key untouched.

A precise state controller should:

- flip the target answer;
- preserve the control answer.

### State-only calibration

Show only state. This verifies that the model can read the state when no history competes with it.

### Memory-only calibration

Show only history. This measures direct episodic readout.

### Clone, fork, and cross-swap

Clone one state, give branches different events, then transplant one branch's state into the other.

### Reconvergence

Apply a synchronizing event that makes relevant branch states identical again. If current state fully governs behavior, the branches should reconverge.

## Why S08 is the causal centerpiece

Earlier sprints compare memory formats or update procedures.

S08 performs targeted interventions while matching competing channels.

It asks not only whether the state contains information, but whether manipulating that state **causes** a selective behavioral change.

<div class="interactive-lab" data-widget="s08-matrix">
<div class="kicker">Factorial explorer</div>
<h2>Cross memory and state</h2>
<div id="s08-matrix"></div>
</div>
