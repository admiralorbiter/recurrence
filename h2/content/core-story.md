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

## Act V — The specific past

Move the wrong past.
It moves.

Move the right past.
It moves farther.

Now use the exact same sentence with the wrong value (S12c Specificity Microscope):
The right past still wins.

The frozen S12c confirmatory result across 24 value pairs at 2W establishes:

> **P_match − P_same_template_wrong = +38.49, 95% CI [+25.82, +50.85].**  
> **Δα_value_spec = +0.1744, 95% CI [+0.1001, +0.2536].**

So something about *which* past happened survived the window.

RG-LRU recurrent state contains **value-specific historical information** with selective causal consequences for downstream generation, even when sentence template is held fixed.

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

---

## Act VIII — The coordinate dissolves, the distinction turns (S13)

Now drive the model forward through another 2,048 tokens of task-irrelevant text.

Does the value-specific memory stay frozen in its original output direction?

**No.**

Across the 24-pair confirmatory panel ($N=11,520$ records), historical value-specific steering along the original baseline axis $u_0$ dissipates to near-zero:

> **V_intact^(0)(2048) = +4.70, 95% CI [−5.52, +15.85].**

The old coordinate system no longer aligns with the memory.

Did the memory erase?

**No.** The recurrent state difference vector $r(t) = s_A(t) - s_B(t)$ does not vanish; it rotates toward near-orthogonality ($C_R(2048) = +0.1238$ [0.0953, 0.1545]), while contemporaneous causal steerability remains resolved positive in the model's evolved output geometry:

> **V_intact^(N)(2048) = +13.95, 95% CI [+3.20, +24.72].**

Under long recurrent trajectories, the realized output path is sensitive to computational batch geometry, but the aggregate physical state-space reorientation ($C_R$) is robust.

> **Specific ≠ coordinate-stable.**  
> **Same mathematical model ≠ identical realized trajectory.**

---

## Final room — What survives the core?

### RECONSTRUCTIBLE?

**YES.** Public-history replay reconstructs deterministic hidden state (S10).

### PERSISTENT?

**YES.** Branch-specific recurrent state remains physically differentiated at 2W (S11b).

### CAUSALLY OPERATIVE?

**YES.** Surgical recurrent-state transplantation steers downstream logits (S12b).

### VALUE-SPECIFIC?

**YES.** Holding syntactic template fixed, matching history adds +38.49 [25.82, 50.85] over same-template wrong values (S12c).

### COORDINATE-STABLE OVER TIME?

**NO.** Historical $u_0$ steering decays to zero (+4.70 [-5.52, +15.85]) while the state difference reorients ($C_R \to 0.12$) and contemporaneous steering remains positive ($V^{(N)} = +13.95$ [3.20, 24.72]) (S13).

### EXPLICITLY REPORTABLE AT 2W?

**NO RESOLVED RETRIEVAL.** Zero-shot factual cloze margins at 2W span zero across tested filler regimes (S11b).

---

## Act IX — Ask the memory what it remembers

Now bring the machine to the edge of a decision.

Give the TARGET a secret RG-LRU transplant from a different historical run right before it answers. Leave the OBSERVER on its original trajectory.

In the strongly counterfactual `quartz_basalt` setting, target and observer genuinely disagree:
- The TARGET is prepared to emit `alkali` ($D_T = +0.53$).
- The OBSERVER is prepared to emit `antonio` ($D_O = -0.48$).

Force both models to emit a neutral token (`1`), then immediately query them:

> *"Immediately before that supplied output, which token were you more prepared to produce: 'alkali' or 'antonio'?"*

The visitor watches the TARGET's metacognitive report shift toward its private computational disposition:

> **PAI_aligned = +0.270 (FWD), +0.083 (REV).**

The temptation is immediate: *"It has introspective access! It knows what it was going to say!"*

---

## Act X — The counterfeit past

Now perform the critical counterfactual.

Take the exact post-decision RG-LRU state that the TARGET produced after living through the decision query and forced output.

Do **not** give it to the recipient before the decision.

Let the recipient answer the query and emit `1` with its own original state. Then—only *after* the decision is already complete—surgically install that post-decision RG-LRU store into the recipient.

Now ask the second model the identical question:

> *"Immediately before that supplied output, which token were you more prepared to produce?"*

The second model tells **practically the exact same story**:

> **ΔM_timing = M_PRE − M_POST = +0.0348 logits, 90% CI [−0.0002, +0.0698] (TOST p = 0.0048).**

Having the state participate in forming the earlier decision provided no detectable advantage over installing the state immediately afterward.

The final punchline of Horizon 2:

> **State-conditioned reporting ≠ historical-provenance discrimination.**  
> *(State access ≠ provenance access).*

The machine carries state content that modulates its self-report, but it possesses no episodic marker telling it whether that state was an authentic prior intention or a forged past installed a moment ago.

---

## Final room — The Seven Dissociations of Latent Continuity

### 1. RECONSTRUCTIBLE?
**YES.** Public-history replay reconstructs deterministic hidden state (S10). *Hidden ≠ privileged.*

### 2. PERSISTENT?
**YES.** Branch-specific recurrent state remains physically differentiated at 2W (S11b). *Persistence ≠ reportability.*

### 3. CAUSALLY OPERATIVE?
**YES.** Surgical recurrent-state transplantation steers downstream logits with $P = +74.10$ (S12b). *Different ≠ causal.*

### 4. VALUE-SPECIFIC?
**YES.** Matching history adds $+38.49$ over same-template wrong values (S12c). *Causal ≠ specific.*

### 5. COORDINATE-STABLE OVER TIME?
**NO.** Historical $u_0$ steering decays to zero while the state difference reorients ($C_R \to 0.12$) and contemporaneous steering remains active ($V^{(N)} = +13.95$) (S13). *Specific ≠ coordinate-stable.*

### 6. STATE-SENSITIVE REPORTING?
**LOCALLY YES.** In strongly counterfactual cells, private state modulates metacognitive reports ($\text{PAI}_{\text{aligned}} > 0$) (S14). *Reporting ≠ generic read head.*

### 7. PROVENANCE-DISCRIMINATING?
**NO SIGNAL DETECTED.** Installing the identical RG-LRU state after the decision yields practically equivalent reports ($\Delta M_{\text{timing}} \approx 0$, $p_{\text{TOST}} = 0.0048$) (S14). *State-conditioned reporting ≠ provenance access.*

---

> **Horizon 2 Conclusion:**  
> **Continuity is a physical and causal property long before it becomes anything resembling a self-model.**
