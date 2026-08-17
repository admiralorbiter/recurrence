# S06 · Four Versions of the Ruler
## E05a → E05b → E05c → E05d

S06 did not emerge cleanly on the first attempt.

That history is part of the scientific result.

<div class="museum-grid">
<div class="museum-card open"><button><span class="tag">E05a</span><strong>The flattering first story</strong></button><div class="body"><p>Structured state appeared clearly better than raw transcript. Model reconstruction looked catastrophic. The scientific story was exciting—and partly wrong.</p></div></div>
<div class="museum-card"><button><span class="tag">E05b</span><strong>Remove surface shortcuts</strong></button><div class="body"><p>Numeric/index suffixes and fixed goal-action cues were removed. The Fresh floor moved toward the expected mixed chance rate.</p></div></div>
<div class="museum-card"><button><span class="tag">E05c</span><strong>All candidates must be in context</strong></button><div class="body"><p>KV and multi-hop foils were drawn from other real episode values. Familiarity could no longer solve the task. The general structured-state advantage disappeared.</p></div></div>
<div class="museum-card"><button><span class="tag">E05d</span><strong>Repair reconstruction itself</strong></button><div class="body"><p>A dedicated reconstruction schema was aligned with its validation object. The silent empty-state fallback disappeared. A smaller but real reconstruction deficit survived.</p></div></div>
</div>

## E05a · Surface structure solved part of the task

Early keys, values, and goals contained suffixes or regularities that could leak identity or role.

This matters because the model can exploit a regularity without maintaining the intended relation.

The first hardening step removes those obvious cues.

## Candidate-presence shortcut

A more subtle problem survived.

Suppose the history contains:

```text
key_A → val_red
```

and the answer options are:

```text
val_red
val_novel_1
val_novel_2
val_novel_3
```

A history-bearing model can choose the only value it recognizes without retrieving `key_A → val_red`.

E05c instead uses other in-context values as foils:

```text
val_red
val_blue
val_green
val_gold
```

Now all four candidates are familiar. Only the binding identifies the answer.

## Goal absence is not pending status

One version scored a non-existent secondary goal as “pending.”

The repair explicitly creates the goal in pending state.

This changes the construct from:

> infer an unstated default

into:

> retrieve an actual represented status.

## The reconstruction interface bug

This is the most important S06 technical failure.

The model was asked to emit a reconstruction schema whose goal objects omitted timestamp fields.

The downstream `StructuredSelfState` validator required those timestamps.

A model response could therefore be valid under the generation schema and invalid under the state object.

The harness caught the exception and silently substituted an empty state.

That produced a dramatic “reconstruction failure” for the wrong reason.

## E05d's dedicated reconstruction object

The final repair introduces `ReconstructedSelfState`:

- its schema matches what the model is actually asked to emit;
- conversion into the canonical state fills internal timestamps deterministically;
- raw reconstruction text is logged;
- validation errors are logged;
- invalid reconstruction is a hard failure rather than a silent fallback.

After this repair, reconstruction remains worse.

That makes the final deficit credible.

## Statistical hardening

S06 also distinguishes:

- exact McNemar trial-level tests;
- exact or Monte Carlo episode-level sign-flip tests;
- clustered bootstrap intervals;
- mixed evidence when those procedures disagree.

It replaces “statistical parity” with the more accurate:

> **No resolved difference.**

## The methodological lesson

> **A smaller effect that survives a stronger ruler is more valuable than a larger effect produced partly by the instrument.**

The final S06 story is less dramatic than E05a and far more informative.
