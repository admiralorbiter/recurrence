# The Memory That Couldn't Remember

## A journey beyond the attention window

### Prologue — Give the machine a past

Give two otherwise matched runs one historical difference.

> The marked object was **amber**.

versus

> The marked object was **cobalt**.

Then continue both runs with the same future input.

RecurrentGemma carries history through three physical stores: a four-token convolution buffer, a 2,048-token sliding attention KV cache, and a continuous RG-LRU recurrent state.

The first question sounds simple: when the local stores lose direct residency, is the past gone?

---

## Act I — The machine with three memories

At roughly three filler tokens, the original event no longer resides directly in the convolution buffer.

At the attention-window boundary, it no longer resides directly in the local KV cache either.

At 4,096 filler tokens — twice the local attention window — neither local store contains the historical event as directly resident tokens.

But Horizon 2 first establishes a methodological warning from S10:

> **Hidden ≠ privileged.**

The internal state is operationally hidden from the prompt text, but deterministic public-history replay reconstructs it. Hidden state alone is not evidence of informational privilege.

---

## Act II — The ghost remains

S11 follows paired historical trajectories across architectural boundaries.

As local residency disappears, the physical representations do not simply become identical. Branch-specific RG-LRU separation remains above the sham floor even at 4,096 tokens across all four filler regimes.

At the same time, the original paired factual cloze probe no longer resolves retrieval at 4,096 tokens in any tested filler regime.

The correct statement is therefore not “the model remembers the fact.”

It is narrower and stranger:

> **Branch-specific recurrent state remains physically differentiated while the original paired retrieval probe is unresolved.**

Physical persistence and behavioral reportability have come apart.

---

## Act III — Different is cheap

Two states being different does not establish that their difference matters.

Two files can differ by one irrelevant bit. Two neural trajectories can remain separated while later computation ignores the separating direction.

So S12 changes the question from observation to intervention:

> If we transplant one run's surviving state into another run, does the recipient's future move toward the donor?

---

## Act IV — Steal somebody else's past

At twice the attention window, transplant the matching RG-LRU state from donor A into recipient B while leaving the recipient's other stores intact.

The downstream logit distribution moves strongly in the donor direction.

The frozen S12b estimate is:

> **P_RGLRU(2W) = +74.10, 95% CI [+46.79, +106.72].**

The physical trace is therefore not merely residual geometry. It has causal leverage over later computation.

---

## Act V — The suspicious result

Matching history is not the only structured history that steers the recipient.

At 2W:

- matching RG-LRU donor: +74.10;
- unrelated cyclic donor: +54.42;
- other permuted donor: +44.46;
- matched Frobenius noise: +17.64.

Matching history adds a resolved selective increment over unrelated histories:

> **+19.68, 95% CI [+1.84, +39.12].**

But the large cross-history baseline matters scientifically. The frozen result establishes **partial specificity**, not a clean “only the correct memory works” story.

This is where S12c attaches as a live Specificity Microscope: same-template wrong values versus cross-template histories versus matched history.

---

## Act VI — The store race that wasn't

A normalized causal-share statistic tempts a simple ranking:

- KV share: 0.632;
- RG-LRU share: 0.368.

But those shares are algebraic complements whose denominator varies with total contrast. They do not establish that KV has greater absolute causal power.

The preregistered absolute contrast is:

> **P_KV − P_RGLRU = −11.65, 95% CI [−49.02, +20.11].**

The interval spans zero.

> **NO RESOLVED WINNER.**

This is measurement archaeology in miniature: a visually compelling normalized statistic answers a different question than the one we actually care about.

---

## Act VII — The memory begins leaking forward

After grafting recurrent state, continue the recipient with new filler and watch the newly written KV representation.

At N=512 tokens, the post-graft KV geometry remains strongly recipient-anchored.

At N=2048 — a full attention-window turnover — recipient anchoring is reduced, but the grand mean remains on the recipient side.

Something changes as the old local attention window turns over. The frozen S12 evidence does **not** justify calling that process confirmed mediation.

The thread remains open.

---

## Final room — What survives the core?

### RECONSTRUCTIBLE?

**YES.** Public-history replay reconstructs deterministic hidden state.

### PERSISTENT?

**YES.** Branch-specific recurrent state remains physically differentiated at 2W.

### CAUSALLY OPERATIVE?

**YES.** Surgical recurrent-state transplantation steers downstream logits.

### SPECIFIC?

**PARTIALLY.** Matching history has a resolved selective increment, but other structured histories also steer substantially. S12c is live.

### OWNED / PRIVILEGED?

**UNKNOWN.** The frozen core establishes neither introspective access nor source ownership.

---

# S13 — What happens when nothing meaningful happens?

The next major question is dynamical rather than mnemonic:

> The past survives. The past matters. What happens to that past when no new task-relevant information enters?
