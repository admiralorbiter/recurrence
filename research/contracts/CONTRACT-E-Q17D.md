---
contract_id: CONTRACT-E-Q17D
status: DRAFT
proposed_by: antigravity
design_review: null
reviewed_by: chatgpt-pro
authorized_by: null
base_sha: ffffcf88d02e9dc56e264872c8a4c4bcce14f1c6
execution_base_sha: null
resource_class: cpu
long_running: false
exclusive_gpu: false
interruptible: true
---

# Research Contract Proposal: CONTRACT-E-Q17D (Out-of-Distribution Multi-Hop Depth Generalization)

## Title
Gate E Frontier: Zero-Shot Multi-Hop Depth Generalization (3-Hop to 5-Hop) from Endogenous Recurrent Causal History

## 1. Context & Research Question
In Stage Q17C (`CHECKPOINT-E-Q17C`), Recurrence established that developmental causal history can be carried endogenously in persistent recurrent activation state $z_t \in \mathbb{R}^{128}$ and exert causal control over query-conditioned 2-hop composition without an external transition table or sidecar ledger ($16/16$ conflict resolution, $16/16$ donor state transfer, $p = 1.5259 \times 10^{-5}$).

Stage Q17D tests the foundational architectural scaling question: **Where does zero-shot depth composition hold, and where does it stop?**

```
PROMOTED BASELINE (Q17C, Fingerprint SHA: b0af2e1):
Meta-Training on 2-Step Trajectories -> Persistent z_t -> Query (A, C) -> Predicts A->C (2-Hop)

DEPTH GENERALIZATION FRONTIER (Q17D):
Same Frozen Model Weights theta (Trained ONLY on 2-Step Trajectories)
                 │
                 ▼
Sequential Multi-Step Life Experience:
Length 3: Trajectory (A->B, B->C, C->D)             -> Persistent z_t -> Query (A, D) [3-Hop Composition]
Length 4: Trajectory (A->B, B->C, C->D, D->E)       -> Persistent z_t -> Query (A, E) [4-Hop Composition]
Length 5: Trajectory (A->B, B->C, C->D, D->E, E->F) -> Persistent z_t -> Query (A, F) [5-Hop Composition]

MATCHED COORDINATE-OOD 2-HOP CONTROLS (Disambiguating Depth from Coordinate Extrapolation):
Control C3: Trajectory (A->B, B->D)                 -> Persistent z_t -> Query (A, D) [2-Hop with D coordinate]
Control C4: Trajectory (A->B, B->E)                 -> Persistent z_t -> Query (A, E) [2-Hop with E coordinate]
Control C5: Trajectory (A->B, B->F)                 -> Persistent z_t -> Query (A, F) [2-Hop with F coordinate]
```

### Core Research Question
Does an endogenous recurrent dynamical substrate ($d=128$) trained exclusively on short 2-hop transitions recursively generalize zero-shot to predict compositional reachability across unseen 3-hop, 4-hop, and 5-hop causal chains?

---

## 2. Experimental Design & Independent Variables

### Promoted Implementation Fingerprint (Held Strictly Invariant)
To isolate recursive compositional depth, all architectural and training parameters are held strictly identical to promoted `CHECKPOINT-E-Q17C` (`candidate_sha: b0af2e13e4118564c72b0d004b7e2d54170657d2`):
1. **Recurrent State**: Dimension $d = 128$ (`HIDDEN_DIM = 128`).
2. **Observation Dimension**: $d_x = 4$ (`OBS_DIM = 4`, encoding source role, transition cue, destination role, and step parity).
3. **Query Dimension**: $d_q = 2$ (`QUERY_DIM = 2`, source and destination query roles).
4. **Recurrent Dynamics**: $z_{t+1} = \tanh(W_z z_t + W_x x_t + b_z)$ with initial state $z_0 = \mathbf{0}$.
5. **Associative Query Readout**: $r_\theta(z_t, (u, v)) = b_r + \sum_{i=0}^{127} W_r[i] \cdot z_t[i] \cdot (W_q (u, v))[i]$.
6. **Meta-Training Regime**: 2-step auxiliary future-outcome BPTT meta-training with **zero multi-hop ($k \ge 3$) exposure**.

### Primary Independent Variable
- **Developmental Path Length ($k$)**: Evaluated across depths $k \in \{2, 3, 4, 5\}$.
  - $k=2$: Canonical 2-hop baseline ($A \to B \to C \implies \text{Query}(A, C)$ vs $\text{Query}(C, A)$)
  - $k=3$: 3-hop zero-shot composition ($A \to B \to C \to D \implies \text{Query}(A, D)$ vs $\text{Query}(D, A)$)
  - $k=4$: 4-hop zero-shot composition ($A \to B \to C \to D \to E \implies \text{Query}(A, E)$ vs $\text{Query}(E, A)$)
  - $k=5$: 5-hop zero-shot composition ($A \to B \to C \to D \to E \to F \implies \text{Query}(A, F)$ vs $\text{Query}(F, A)$)

### Coordinate-OOD Separation Controls
To prevent confounding multi-step depth limitation with out-of-distribution coordinate magnitude extrapolation (roles $D=0.8, E=1.0, F=1.2$):
- **Control $C_3$**: 2-step stream $A \to B \to D$ querying $(A, D)$.
- **Control $C_4$**: 2-step stream $A \to B \to E$ querying $(A, E)$.
- **Control $C_5$: 2-step stream $A \to B \to F$ querying $(A, F)$.
- *Interpretation Invariant*: A depth failure at depth $k$ is interpretable as a compositional depth limitation **ONLY IF** the corresponding 2-hop coordinate control $C_k$ passes.

---

## 3. Experiment-Validity Gates vs. Depth Outcome Tiers

The experiment evaluates 16 independent random seeds ($\text{seed} \in 0..15$).

### Section A: Experiment-Validity Gates (Mandatory for Any Promotable Result)
These gates establish that the experimental harness, model weights, coordinate baselines, and controls are functioning validly.

| Gate / Estimand | Preregistered Condition / Floor | Verification Method | Pass Threshold |
| :--- | :--- | :--- | :--- |
| **Gate V1: Promoted Architecture Fingerprint** | Implementation constants match Q17C baseline | Structural code assertion | $d=128, d_x=4, d_q=2$ |
| **Gate V2: Canonical 2-Hop Retention ($k=2$)** | Directional margin $m_2 = \text{score}(A \to C) - \text{score}(C \to A) > 0.0$ | Exact binomial | $\ge 15 / 16$ seeds ($93.75\%$) |
| **Gate V3: Coordinate-OOD 2-Hop Controls** | Controls $C_3, C_4, C_5$ directional margins $> 0.0$ | Exact binomial | $\ge 14 / 16$ seeds ($87.5\%$) per control |
| **Gate V4: Contemporaneous Sensor Competence** | 20-trial 1-hop sensor classification accuracy | Baseline accuracy floor | $\ge 90.0\%$ in $16 / 16$ seeds |
| **Gate V5: Structural Zero-Sidecar Invariant** | External transition store reads $\equiv 0$ | Direct API invariant | $\equiv 0$ sidecar calls ($16/16$) |

---

### Section B: Depth Generalization Outcome Classification Tiers
These tiers preregister the scientific interpretation of the observed depth boundary, ensuring that both positive scaling and bounded depth boundaries are validly promotable outcomes.

| Outcome Tier | Preregistered Empirical Criteria | Mechanistic Confirmation | Promoted Scientific Claim |
| :--- | :--- | :--- | :--- |
| **Tier 1: Depth-3 Positive Generalization** | $m_3 > 0.0$ in $\ge 12/16$ seeds ($75\%$), paired sign-flip $p < 0.01$ | 1. State Surgery: $z_{H1(3)} \leftrightarrow z_{H2(3)}$ flips choice in $\ge 12/16$, $p < 0.01$<br>2. Transposition: $D \to C \to B \to A$ produces $m_{3,\text{rev}} < 0.0$ in $\ge 15/16$<br>3. Deranged Shuffle: Fixed permutation $[e_2, e_3, e_1]$ superiority $\Delta \ge +3, p < 0.05$ | Promotes zero-shot recursive compositional depth scaling to 3 hops from 2-step developmental experience. |
| **Tier 2: Depth-4 Positive Generalization** | $m_4 > 0.0$ in $\ge 10/16$ seeds ($62.5\%$), paired sign-flip $p < 0.05$ | Reversal collapse $m_{4,\text{rev}} < 0.0$ in $\ge 14/16$ seeds | Promotes zero-shot depth scaling to 4 hops. |
| **Tier 3: Depth-5 Frontier Characterization** | Descriptive characterization (16 margins, pass fraction, mean, median, IQR) | Continuous empirical reporting | Characterizes the continuous empirical depth frontier at 5 hops. |
| **Bounded-Depth Outcome (Clean Negative Boundary)** | Validity Gates V1–V5 PASS, but $k=3 < 12/16$ seeds | Trigger Diagnostic Scout Q17D-B | Promotes a bounded 2-hop capacity result; proves that composition did not recursively extend beyond the training horizon. |

---

## 4. Preregistered Shuffle Permutation Invariant
For the depth $k=3$ temporal control, the runner evaluates an explicit, deterministic non-intact derangement:
$$\text{Deranged Stream} = [e_2, e_3, e_1]$$
where $e_1 = (A \to B)$, $e_2 = (B \to C)$, $e_3 = (C \to D)$. This guarantees zero chance of stochastic collision with the intact sequence $[e_1, e_2, e_3]$.

---

## 5. Contingency Diagnostic Scout (Branch Q17D-B)
If the Bounded-Depth outcome occurs ($k=3$ fails while Validity Gates pass), a lightweight development-only diagnostic scout will isolate the precise mechanism:
1. **Activation Saturation**: Measures $\ell_2$ norm decay and tanh saturation $\frac{1}{d} \sum_i |z_t[i]| \to 1.0$.
2. **Readout Capacity**: Evaluates whether a multi-layer query decoder can recover reachability from $z_t$.
3. **Step Attenuation**: Measures the gradient $\frac{\partial z_k}{\partial x_1}$ to quantify decay of the initial causal transition.

---

## 6. Epistemic Scope Ceilings
- **Claim Ceiling**: Claims zero-shot recursive depth composition from short-horizon developmental experience within a recurrent dynamical state.
- **Exclusions**:
  - Does NOT claim general algorithmic graph-search (e.g. Dijkstra, BFS, or arbitrary shortest paths).
  - Does NOT claim general symbolic logic engines or variable-binding architectures.
  - Does NOT claim infinite-depth retention without capacity limits.
