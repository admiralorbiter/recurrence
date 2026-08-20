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

## 2. The $2 \times 2$ Factorial Double Dissociation Design

We define two measurable computational quantities in a decision task:
1. **First-Order Content ($C$):** The specific task decision or token disposition ($D(x, y) = z(x) - z(y)$).
2. **Monitor State ($M$):** The internal representation of uncertainty, conflict, or readiness (e.g., confidence margin, entropy, or metacognitive judgment).

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

### Diagnostic Success Criteria:
- **Cell $(C_0, M_1)$:** First-order answer remains stable ($|D_{C_0, M_1} - D_{C_0, M_0}| < \epsilon$), but the verbal confidence report or routing choice shifts significantly.
- **Cell $(C_1, M_0)$:** First-order answer changes direction ($D_{C_1, M_0} \cdot D_{C_0, M_0} < 0$), while the monitor correctly reflects the new state without being identical to the content representation.

---

## 3. Operational Global-Workspace Hallmarks

Following recent operational workspace frameworks (e.g., Anthropic, 2026), a candidate monitor/workspace representation must satisfy four functional properties:

1. **Reportability:** The monitor state can be queried and verbalized across different prompt formats.
2. **Controllability:** Intervening on the candidate subspace predictably modulates confidence/routing.
3. **Flexible Task Reuse:** The same candidate subspace is utilized across at least two distinct task domains (e.g. factual retrieval vs. lexical choice).
4. **Selective Non-Destructive Mediation:** Lesioning or clamping the candidate monitor subspace alters self-assessment without destroying general language fluency or first-order accuracy.

---

## 4. Candidate State Subspaces

1. **Subspace 1 (Store Isolation):** RG-LRU recurrent memory (slow carrier) vs. Top-layer residual / Conv1D buffer (fast monitor).
2. **Subspace 2 (Attention Entropy Direction):** The projection along the singular vector that tracks attention dispersion or token competition during ambiguous prompts.
3. **Subspace 3 (Readiness Vector):** The differential activation vector between high-conflict and low-conflict decision states.

---

## 5. Hard Budget & Pre-Registered Stop Rules

To prevent endless exploratory tuning of frozen pretrained weights, S16 operates under strict pre-registered stop rules:

1. **Execution Budget:** Maximum 1 task domain, 3 candidate projection vectors, 40 total evaluation trials.
2. **Stop Rule (No Dissociation):** If every intervention on $M$ symmetrically alters $C$ (or vice-versa), or if perturbing $M$ causes generic language degradation, **terminate the scout and declare that no separable monitor state exists in the frozen pretrained architecture.**
3. **Transition Trigger:** A clean null on this $2 \times 2$ scout establishes the definitive boundary of Horizon 2 and triggers the formal transition to **Horizon 3 (The Continuity Garden / Developmental Organism Bring-Up)**.
