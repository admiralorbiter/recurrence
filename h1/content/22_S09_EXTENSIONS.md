# S09 · The Closure Experiments
## Why H1 did not stop immediately

The core S04–S09 battery already satisfies the Horizon gate.

Two ambiguities remain important enough to test before freezing the public synthesis.

They do not reopen H1 broadly.

They target the exact remaining alternative explanations.

<div class="live-banner">
<strong>Confirmatory Evidence Frozen</strong>
<p>The values on this page report the completed $N=16$ confirmatory results for E08c (32 episodes, 800 trials, Seed 1337) and E09c (16 episodes, 80 fixed items, 320 evaluator probes, Seed 1337). All analyses carry cluster-bootstrapped confidence intervals and exact permutation tests.</p>
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

### Confirmatory N=16 Result

- role-reversal shift: **+28.1pp** ($95\%$ CI: $[+15.6\%, +41.2\%]$, exact sign-flip $p = \mathbf{0.0012}$);
- residual Alpha token bias: **+8.1pp** ($95\%$ CI: $[+1.2\%, +14.4\%]$);
- Role A (Alpha=Self): 47.5% Alpha (Self), 11.2% Beta (Peer);
- Role B (Beta=Self): 40.0% Beta (Self), 20.0% Alpha (Peer);
- True-Self accuracy: identical at **75.0%** across both roles.

Role designation is a strong causal contributor to attribution, dominating but not eliminating a smaller residual actor-token preference.

<div class="interactive-lab" data-widget="e08c-role-explorer">
<div class="kicker">Confirmatory role counterbalance</div>
<h2>Flip the Self assignment</h2>
<div id="e08c-role-explorer"></div>
</div>

### Failed Direct-Mention Positive Control

An isolated, explicit provenance-lookup positive control reaches only **21.2%** accuracy (5AFC chance: 20.0%), collapsing toward Self (**68.8%**) while external sources range from 3.1% to 15.6%.

This is not an instrument ceiling; it is a **failed positive control**. Even under direct explicit mention without memory load, the model collapses toward the designated primary Self role, establishing that prompt-level role packaging interferes with source lookup and motivating role-channel ablation (E08d).

## E09c · Does the metacognitive reversal survive a fixed target?

Canonical E09 generates separate first-order choices under transcript and scaffolded contexts.

E09c freezes the first-order target decision per item across:

- transcript assessment;
- scaffolded assessment;
- Self evaluator;
- Observer evaluator.

Now the only intended change is the assessment format and framing—not the decision being evaluated.

### Confirmatory N=16 Result

- first-order accuracy fixed at **47.5%** (38/80 items) in all four cells;
- Brier difference-in-differences: **+0.1880** ($95\%$ CI: $[-0.0232, +0.4242]$, exact sign-flip $p = \mathbf{0.1501}$);
- AUROC difference-in-differences: **−0.209** ($95\%$ clustered CI: $[-0.458, +0.021]$, exact format-block swap $p = \mathbf{0.1406}$).

**Epistemic takeaway:** Under strictly matched first-order target decisions, there is **no resolved format × framing interaction under the prespecified exact test** ($p > 0.14$), confirming that explicit Level-1 prompt scaffolding provides no privileged self-calibration channel.

## Why these extensions are worth doing

Both experiments are small because the H1 architecture is already complete.

They serve a different purpose:

- remove one remaining explanation from a headline result;
- simplify the public claim ledger before H2.

They are closure studies, not new horizons.
