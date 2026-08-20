# Sprint S16: Monitor / Content Dissociation Scout Protocol

**Status:** PROPOSED MECHANISTIC CODA PROTOCOL (Horizon 2 Final Investigation)  
**Target Substrate:** `google/recurrentgemma-2b-it` (revision: `2766eb5d4264c6c0357803990791f9ab9cd50f8e`)  
**Dependency:** Horizon 2 Core (S10–S14) Frozen

---

## 1. Motivation & Core Scientific Question

In Sprint S14, we established that possessing a post-decision recurrent state ($R_{\text{POST}}$) is sufficient to reproduce the system's metacognitive report, proving that *state-conditioned reporting does not imply historical provenance discrimination*.

The next logical mechanistic question is:
> **Does the recurrent architecture contain a second-order latent variable whose computational function is to monitor, route, or evaluate first-order representations, rather than merely carry first-order content?**

If such a monitor exists in the frozen weights, Horizon 2 has a major discovery left to characterize. If it does not, Horizon 3 gains its clearest possible motivation: *under what developmental training pressures does an autonomous monitoring interface emerge?*

---

## 2. Two-Phase Experimental Design

Rather than assuming a monitor subspace $M$ exists, S16 proceeds in two sequential phases with an explicit stop rule.

```
                              SPRINT S16 TWO-STAGE PIPELINE
                              
  [Phase A: Discovery & Localization]
    │  - Search for internal variables predicting confidence/uncertainty after controlling for C
    │  - Check linear probes, Jacobian sensitivity, and store/layer localization
    ▼
  Candidate Found? ───( NO )───► [ Clean Null Exit: Proceed to Horizon 3 Development ]
    │ ( YES )
    ▼
  [Phase B: Causal 2x2 Factorial Dissociation]
    │  - Factorial intervention: Content (C_0 vs C_1) x Candidate Monitor (M_0 vs M_1)
    │  - Evaluate first-order preservation (|Delta C| < epsilon) and monitor steering
    ▼
  Double Dissociation Confirmed? ───( YES )───► [ Deeper Horizon 2 Characterization ]
    │ ( NO )
    ▼
  [ Clean Null Exit: Horizon 3 Developmental Bring-Up ]
```

---

## 3. Phase A: Discovery & Correlation Protocol

**Objective:** Identify whether any internal variable or subspace $M$ predicts confidence, conflict, or reporting *after controlling for the first-order decision margin $D(x,y)$*.

### Candidate Subspaces to Screen:
1. **Store Dissociation:** Slow RG-LRU recurrent memory vs. Fast Conv1D rolling buffer / Top-layer residual stream.
2. **Attention Entropy Direction:** The principal singular vector of attention weight dispersion during high-conflict prompts.
3. **Task-Conflict Contrast Vector:** The differential activation vector between high-margin ($|D| > 2.0$) and near-zero-margin ($|D| < 0.10$) prompt evaluations.

### Phase A Gate:
A candidate subspace $M$ passes Phase A if:
- Partial correlation $r(M, \text{Report} \mid C) > 0.30$ ($p < 0.01$).
- The direction is stable across at least two distinct semantic template families.

*If no candidate subspace satisfies these criteria across 40 screening trials, terminate S16 immediately and declare no separable monitor exists in frozen weights.*

---

## 4. Phase B: Causal $2 \times 2$ Factorial Intervention Protocol

**Objective:** Test whether candidate subspace $M$ and first-order decision state $C$ can be causally double-dissociated.

```
                              THE 2 x 2 FACTORIAL INTERVENTION MATRIX
                              
                        ┌──────────────────────────────┬──────────────────────────────┐
                        │      Monitor Intact (M_0)    │      Monitor Altered (M_1)   │
  ┌─────────────────────┼──────────────────────────────┼──────────────────────────────┤
  │ Content Intact (C_0)│ Baseline task answer         │ Same task answer;            │
  │                     │ Baseline confidence / report │ Changed confidence / report  │
  ├─────────────────────┼──────────────────────────────┼──────────────────────────────┤
  │ Content Altered(C_1)│ Changed task answer;         │ Changed task answer;         │
  │                     │ Monitor tracks the change    │ Changed confidence / report  │
  └─────────────────────┴──────────────────────────────┴──────────────────────────────┘
```

### Pre-Registered Causal Thresholds:
1. **First-Order Preservation Tolerance ($\epsilon$):** In the $(C_0, M_1)$ cell, first-order decision margin must remain stable within $|\Delta D| \le 0.15$ logits.
2. **Monitor Steering Effect:** In the $(C_0, M_1)$ cell, the verbalized confidence report or downstream routing choice must shift by at least $|\Delta \text{Report}| \ge 0.40$ logits ($p < 0.05$).
3. **Controls Required:**
   - *Norm-matched random direction control:* Verifies effects are not generic activation shocks.
   - *Language fluency check:* Output perplexity on standard filler text must increase by less than $10\%$.
   - *Matched observer control:* Target report shift must exceed third-party text inference.

---

## 5. Hard Budget & Pre-Registered Stop Rules

1. **Execution Budget:** Maximum 1 task domain, 3 candidate projection vectors, 40 total evaluation trials.
2. **Stop Rule:** If every intervention on $M$ symmetrically alters $C$ (or vice-versa), or if perturbing $M$ causes generic language degradation, **terminate the scout and declare that no separable monitor state exists in the frozen pretrained architecture.**
3. **Transition Trigger:** A clean null on this $2 \times 2$ scout establishes the definitive boundary of Horizon 2 and triggers the formal transition to **Horizon 3 (The Continuity Garden / Developmental Organism Bring-Up)**.
