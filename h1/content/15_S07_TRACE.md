# S07 · The Self-Polluting Loop
## The most important number is not the final accuracy

Across the available-inference selective condition:

- **274 derived writes**;
- **0 exact correct root → terminal derivations**;
- precision: **0%**;
- recall: **0%**.

This is not a case where the scorer rejects a mostly correct paraphrase.

The trace contains malformed opaque identifiers and corrupted associations such as:

```text
key_dist1_platinumatinum_bea
key_dist1_platinum_b_bea → val_dist1_c_v_vortex
```

## How the loop develops

<div class="interactive-lab" data-widget="s07-trace-player">
<div class="kicker">Mechanism trace</div>
<h2>Step through a syntheticized reflection failure</h2>
<div id="s07-trace-player"></div>
</div>

The mechanism is recursive:

1. The first reflection tick sees correct protected facts.
2. It writes a malformed or unsupported derived inference.
3. The second tick sees both the correct facts and the bad write.
4. It treats the bad write as part of current state.
5. New writes drift farther from the original symbolic structure.
6. The final query must reason through an increasingly polluted prompt.

## Protected facts remain intact

Selective evidence mutation is **0%**.

The working-memory and source fields are clamped.

Yet available multi-hop accuracy drops from:

- Strict Identity: **62.5%**
- Selective Reflection: **31.2%**

The effect estimate is −31.2pp. Bootstrap and trial-level evidence support a deficit, while the primary episode permutation narrowly misses the conventional threshold (`p = .0625`).

The safe interpretation is not “statistically proven cognitive damage.”

The safe interpretation is:

> The audited write channel fails completely to persist the target derivation and introduces a plausible interference mechanism.

## Factual integrity is not epistemic quality

This is one of H1's most useful new concepts.

### Factual integrity

Are the original protected facts still present and unchanged?

Yes.

### Epistemic-state quality

Is the entire state still organized so those facts can be used effectively?

Not necessarily.

Bad hypotheses, duplicated strings, stale unresolved items, and contradictory derived claims can reduce usability without deleting ground truth.

## Unconstrained rewriting

The unconstrained condition is more destructive:

- 98.4% evidence drift;
- stable working-memory readout falls to 18.8%;
- completion loops can reach 1,850 tokens per tick;
- overall accuracy falls to 42.7%.

The primary episode test for the overall −17.7pp difference is `p = .0707`, so H1 emphasizes the direct drift audit rather than overstating the aggregate performance contrast.

## Direct reasoning is not the same as persistent writing

Under Strict Identity, the model reaches 62.5% on the available multi-hop items.

The raw transcript reaches 78.1%.

Therefore S07 does not show that Qwen2.5-3B is incapable of relational inference.

It shows:

> **The tested iterative prompt-level reflection mechanism cannot reliably externalize and persist the valid inference into opaque symbolic derived state.**

## The S07 gate

Deterministic preservation remains the strongest tested Level-1 management strategy.

The model-generated quiet update loop does not provide a successful consolidation mechanism.

Whether a native recurrent latent process behaves differently remains open.

<div class="handoff-card">
<span>S07 → S08</span>
<strong>From state quality to state authority</strong>
<p>If the model is unreliable at writing state, perhaps the state can still control behavior when the experimenter edits it directly. S08 reaches into the declared current state and changes it.</p>
</div>

<div class="research-callout">
<strong>Research bridge: memory transformation and provenance</strong>
<p>Human reconsolidation research and recent agent-memory security work both warn, in very different systems, that revisiting or consolidating memory can transform provenance and content rather than merely preserve it. H1 does not equate these mechanisms; the bridge is the shared engineering question: when should generated material be allowed to become persistent state?</p>
<a href="https://arxiv.org/abs/2607.29167" target="_blank" rel="noopener">Open recent provenance-laundering research</a>
</div>
