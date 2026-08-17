# S09 · The Closure Experiments
## Why H1 did not stop immediately

The core S04–S09 battery already satisfies the Horizon gate.

Two ambiguities remain important enough to test before freezing the public synthesis.

They do not reopen H1 broadly.

They target the exact remaining alternative explanations.

<div class="live-banner">
<strong>Live evidence boundary</strong>
<p>The values on this page are exploratory N=4 results supplied while the confirmatory runs are still in progress. They are included because they change the question the final site must teach, but they are not part of the frozen claim ledger yet.</p>
</div>

## E08c · Does the attribution attractor follow the Self role?

Canonical E08 fixes:

```text
agent_alpha = Self
agent_beta = Peer
```

That makes three explanations indistinguishable:

1. the model prefers Self;
2. the model prefers the primary actor;
3. the model prefers the token `agent_alpha`.

E08c creates matched role assignments.

### Role A

```text
agent_alpha = Self / primary
agent_beta = Peer
```

### Role B

```text
agent_beta = Self / primary
agent_alpha = Peer
```

Everything else is structurally matched.

### Exploratory N=4 result

- role-reversal shift: **+40.0pp**;
- exploratory CI `[+12.5,+62.5]`;
- residual Alpha token bias: **+5.0pp**;
- Alpha-as-Self attribution: 55.0%;
- Beta-as-Peer attribution: 10.0%;
- Beta-as-Self attribution: 50.0%;
- Alpha-as-Peer attribution: 15.0%.

The attractor appears to move with the designated Self/primary role rather than staying attached to `agent_alpha`.

<div class="interactive-lab" data-widget="e08c-role-explorer">
<div class="kicker">Exploratory role counterbalance</div>
<h2>Flip the Self assignment</h2>
<div id="e08c-role-explorer"></div>
</div>

### Instrument ceiling

An isolated, explicit provenance-lookup positive control reaches only **30.0%** exploratory accuracy.

That is surprisingly low for a 5AFC task with minimal memory load.

It suggests the instrument itself is hard for Qwen2.5-3B and motivates careful interpretation of ownership deficits.

### What the confirmatory result will decide

If the attractor again moves with the role assignment:

> H1 has evidence for a **prompt-role anchoring effect**, not merely an Alpha token prior.

If it stays attached to Alpha:

> The canonical “Self” pattern is substantially lexical.

If it weakens:

> The original arrangement contributed more than either simple explanation.

## E09c · Does the metacognitive reversal survive a fixed target?

Canonical E09 generates separate first-order choices under transcript and scaffolded contexts.

E09c freezes the first-order target decision per item across:

- transcript assessment;
- scaffolded assessment;
- Self evaluator;
- Observer evaluator.

Now the only intended change is the assessment format and framing—not the decision being evaluated.

### Exploratory N=4 result

- first-order accuracy fixed at **25.0%** in all four cells;
- Brier difference-in-differences: **−0.0921**, `p = .8824`;
- AUROC difference-in-differences: **+0.053**, `p = .75`.

No resolved interaction appears when the target choice is strictly fixed.

### What confirmation will decide

If N=16 remains null:

> The earlier canonical format reversal belongs to the broader format-conditioned decision distribution, not an isolated change in metacognitive evaluation of the same choice.

If the interaction reappears:

> Explicit scaffolded state changes the relative Self/Observer calibration even when the target decision is identical.

## Why these extensions are worth doing

Both experiments are small because the H1 architecture is already complete.

They serve a different purpose:

- remove one remaining explanation from a headline result;
- simplify the public claim ledger before H2.

They are closure studies, not new horizons.
