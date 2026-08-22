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
PROMOTED BASELINE (Q17C, Baseline Fingerprint SHA: b0af2e1):
Meta-Training on 2-Step Trajectories -> Persistent z_t -> Query (A, C) -> Predicts A->C (2-Hop)

DEPTH GENERALIZATION FRONTIER (Q17D):
Same Frozen Model Weights theta_i (Trained ONLY on 2-Step Trajectories)
                 │
                 ▼
Sequential Multi-Step Life Experience:
Length 3: Trajectory (A->B, B->C, C->D)             -> Persistent z_t -> Query (A, D) [3-Hop Composition]
Length 4: Trajectory (A->B, B->C, C->D, D->E)       -> Persistent z_t -> Query (A, E) [4-Hop Composition]
Length 5: Trajectory (A->B, B->C, C->D, D->E, E->F) -> Persistent z_t -> Query (A, F) [5-Hop Composition]

DEPTH-SPECIFIC COORDINATE-OOD CONTROLS (Disambiguating Depth from Coordinate Extrapolation):
Control C3: Trajectory (A->B, B->D)                 -> Persistent z_t -> Query (A, D) [2-Hop with D coordinate]
Control C4: Trajectory (A->B, B->E)                 -> Persistent z_t -> Query (A, E) [2-Hop with E coordinate]
Control C5: Trajectory (A->B, B->F)                 -> Persistent z_t -> Query (A, F) [2-Hop with F coordinate]
```

### Core Research Question
Does an endogenous recurrent dynamical substrate ($d=128$) trained exclusively on short 2-hop transitions recursively generalize zero-shot to predict compositional reachability across unseen 3-hop, 4-hop, and 5-hop causal chains?

---

## 2. Experimental Design & Invariants

### Exact Promoted Architecture Fingerprint
To isolate recursive compositional depth, all architectural parameters are held strictly identical to promoted `CHECKPOINT-E-Q17C`:
1. **Recurrent State**: Dimension $d = 128$ (`HIDDEN_DIM = 128`).
2. **Observation Dimension**: $d_x = 4$ (`OBS_DIM = 4`), encoding $[\text{src role}, \text{action}, \text{dst role}, \text{constant bias}]$ with $v[3] = 1.0$.
3. **Query Dimension**: $d_q = 2$ (`QUERY_DIM = 2`, source and destination query roles).
4. **Recurrent Dynamics**: $z_{t+1} = \tanh(W_z z_t + W_x x_t + b_z)$ with initial state $z_0 = \mathbf{0}$.
5. **Associative Query Readout**: $r_\theta(z_t, (u, v)) = b_r + \sum_{i=0}^{127} W_r[i] \cdot z_t[i] \cdot (W_q (u, v))[i]$.

### Operational Definition of "Same Frozen Weights"
For each random seed $i \in 0..15$:
1. Model parameters $\theta_i = (W_z, W_x, b_z, W_q, W_r, b_r)$ are reproduced and meta-trained **ONCE** using the exact promoted Q17C deterministic training procedure, loss function, and auxiliary synthetic seed schedule (`candidate_sha: b0af2e13e4118564c72b0d004b7e2d54170657d2`).
2. A cryptographic SHA-256 hash $\text{theta\_hash}_i$ is computed from the serialized parameter byte array.
3. That exact parameter set $\theta_i$ is **frozen (ZERO subsequent weight updates or fine-tuning)** and cloned across all experimental conditions for seed $i$:
   - Canonical 2-hop baseline ($k=2$)
   - Coordinate control $C_3$
   - 3-hop composition ($k=3$)
   - Coordinate control $C_4$
   - 4-hop composition ($k=4$)
   - Coordinate control $C_5$
   - 5-hop frontier characterization ($k=5$)
   - Multi-hop transposition collapse
   - Deterministic temporal shuffle control
   - Multi-hop causal state surgery
4. The acceptance verifier asserts byte-level parameter equality ($\text{theta\_hash}_i$) across all conditions within seed $i$.

---

## 3. Independent Variables & Controls

### Primary Independent Variable: Developmental Path Length ($k$)
- $k=2$: Canonical 2-hop baseline ($A \to B \to C \implies \text{Query}(A, C)$ vs $\text{Query}(C, A)$)
- $k=3$: 3-hop zero-shot composition ($A \to B \to C \to D \implies \text{Query}(A, D)$ vs $\text{Query}(D, A)$)
- $k=4$: 4-hop zero-shot composition ($A \to B \to C \to D \to E \implies \text{Query}(A, E)$ vs $\text{Query}(E, A)$)
- $k=5$: 5-hop zero-shot composition ($A \to B \to C \to D \to E \to F \implies \text{Query}(A, F)$ vs $\text{Query}(F, A)$)

### Depth-Specific Coordinate-OOD Separation Controls
To prevent confounding recursive multi-step depth limitation with out-of-distribution coordinate magnitude extrapolation (roles $D=0.8, E=1.0, F=1.2$):
- **Control $C_3$**: 2-step stream $A \to B \to D$ querying $(A, D)$.
- **Control $C_4$**: 2-step stream $A \to B \to E$ querying $(A, E)$.
- **Control $C_5$**: 2-step stream $A \to B \to F$ querying $(A, F)$.
- *Interpretation Invariant*:
  - Interpretation of depth $k=3$ is valid **if and only if** Control $C_3 \ge 14/16$.
  - Interpretation of depth $k=4$ is valid **if and only if** Control $C_4 \ge 14/16$.
  - Interpretation of depth $k=5$ is valid **if and only if** Control $C_5 \ge 14/16$.
  - If $C_k < 14/16$, failure at depth $k$ is classified as a coordinate extrapolation limitation rather than a recursive depth failure, and lower depths where coordinate controls passed remain fully interpretable.

### Preregistered Deterministic Shuffle Control Invariant
For the depth $k=3$ temporal control, the runner evaluates an explicit, deterministic non-intact derangement:
$$\text{Deranged Stream} = [e_2, e_3, e_1]$$
where $e_1 = (A \to B)$, $e_2 = (B \to C)$, $e_3 = (C \to D)$.

---

## 4. Experiment-Validity Gates vs. Nested Outcome Tiers

The experiment evaluates 16 independent random seeds ($\text{seed} \in 0..15$).

### Section A: Global Experiment-Validity Gates (Mandatory for Any Promotable Execution)
These gates establish that the experimental harness, model weights, and baseline capacities are functioning validly.

| Gate / Estimand | Preregistered Condition / Floor | Verification Method | Pass Threshold |
| :--- | :--- | :--- | :--- |
| **Gate V1: Promoted Architecture & Weight Fingerprint** | Parameter shapes match Q17C; $\text{theta\_hash}_i$ identical across conditions | Structural code assertion & SHA-256 byte check | $d=128, d_x=4, d_q=2$; $16/16$ hashes verified |
| **Gate V2: Canonical 2-Hop Retention ($k=2$)** | Directional margin $m_2 = \text{score}(A \to C) - \text{score}(C \to A) > 0.0$ | Exact binomial | $\ge 15 / 16$ seeds ($93.75\%$) |
| **Gate V3: Contemporaneous Sensor Competence** | 20-trial 1-hop sensor classification accuracy | Baseline accuracy floor | $\ge 90.0\%$ in $16 / 16$ seeds |
| **Gate V4: Structural Zero-Sidecar Invariant** | External transition store reads $\equiv 0$ | Direct API invariant | $\equiv 0$ sidecar calls ($16/16$) |

---

### Section B: Nested Depth Outcome Classification Tiers
These tiers preregister the scientific interpretation of the observed depth boundary, ensuring that both positive scaling and bounded depth boundaries are validly promotable outcomes.

| Outcome Tier | Preregistered Empirical Criteria | Mechanistic Confirmation | Promoted Scientific Claim |
| :--- | :--- | :--- | :--- |
| **Tier 1: Depth-3 Positive Generalization** | Global Validity PASS + $C_3 \ge 14/16$ + $m_3 > 0.0$ in $\ge 12/16$ seeds ($75\%$), paired sign-flip $p < 0.01$ | 1. State Surgery: $z_{H1(3)} \leftrightarrow z_{H2(3)}$ flips choice in $\ge 12/16, p < 0.01$<br>2. Transposition: $D \to C \to B \to A$ produces $m_{3,\text{rev}} < 0.0$ in $\ge 15/16$<br>3. Deranged Shuffle: Fixed permutation $[e_2, e_3, e_1]$ superiority $\Delta \ge +3, p < 0.05$ | Promotes zero-shot recursive compositional depth scaling to 3 hops from 2-step developmental experience. |
| **Tier 2: Depth-4 Positive Generalization** | Tier 1 Satisfied + $C_4 \ge 14/16$ + $m_4 > 0.0$ in $\ge 10/16$ seeds ($62.5\%$), paired sign-flip $p < 0.05$ | 4-hop reversal collapse $m_{4,\text{rev}} < 0.0$ in $\ge 14/16$ seeds | Promotes zero-shot recursive compositional depth scaling to 4 hops. |
| **Tier 3: Depth-5 Frontier Characterization** | Tier 2 Satisfied + $C_5 \ge 14/16$ + continuous reporting (16 margins, pass fraction, mean, median, IQR) | Continuous empirical distribution characterization | Characterizes the continuous empirical depth frontier at 5 hops. |
| **Bounded-Depth Outcome (Clean Negative 2-Hop Boundary)** | Global Validity PASS + $C_3 \ge 14/16$ valid, but $k=3 < 12/16$ seeds | Trigger Diagnostic Scout Q17D-B | Promotes a clean bounded 2-hop result: composition did not reliably extend to 3-hop depth under the frozen Q17C architecture and assay. |
| **Non-Monotonic / Anomalous Depth Profile** | Higher depth passes while lower depth fails (e.g. $k=3$ fails, $k=4$ passes) | Empirical anomaly audit | Reports non-monotonic depth profile as an empirical anomaly; bars simple depth-scaling promotion and triggers diagnostic investigation. |

---

## 5. Contingency Diagnostic Scout (Branch Q17D-B)
If the Bounded-Depth outcome occurs ($k=3$ fails while Global Validity and $C_3$ pass), a lightweight development-only diagnostic scout will isolate the precise mechanism:
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
