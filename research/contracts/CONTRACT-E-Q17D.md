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
In Stage Q17C (`CHECKPOINT-E-Q17C`), Recurrence established that developmental causal history can be endogenously instantiated in persistent recurrent activation state $z_t \in \mathbb{R}^{128}$ and exert causal control over query-conditioned 2-hop composition without an external transition table or sidecar ledger ($16/16$ conflict resolution, $16/16$ donor state transfer, $p = 1.5259 \times 10^{-5}$).

Stage Q17D tests the foundational architectural hypothesis: **Out-of-Distribution Compositional Depth Scaling**.

```
PROMOTED BASELINE (Q17C, Validated):
Meta-Training on 2-Step Trajectories -> Persistent z_t -> Query (A, C) -> Predicts A->C (2-Hop)

DEPTH GENERALIZATION FRONTIER (Q17D):
Same Frozen Model Weights theta (Trained ONLY on 2-Step Trajectories)
                 │
                 ▼
Sequential Multi-Step Life Experience:
Length 3: Trajectory (A->B, B->C, C->D)           -> Persistent z_t -> Query (A, D) [3-Hop Composition]
Length 4: Trajectory (A->B, B->C, C->D, D->E)     -> Persistent z_t -> Query (A, E) [4-Hop Composition]
Length 5: Trajectory (A->B, B->C, C->D, D->E, E->F) -> Persistent z_t -> Query (A, F) [5-Hop Composition]
```

### Core Research Question
Does an endogenous recurrent dynamical substrate ($d=128$) trained exclusively on short 2-hop transitions recursively generalize zero-shot to predict compositional reachability across unseen 3-hop, 4-hop, and 5-hop causal chains?

---

## 2. Experimental Design & Independent Variables

### Inherited Promoted Invariants (Held Strictly Fixed)
To ensure isolation of the causal mechanism, all non-depth variables are held identical to promoted `CHECKPOINT-E-Q17C`:
1. **Model Architecture**: Recurrent state dimension $d = 128$, input dimension $d_x = 24$, query dimension $d_q = 2$.
2. **Dynamic Update**: $z_{t+1} = \tanh(W_z z_t + W_x x_t + b_z)$ with initial state $z_0 = \mathbf{0}$.
3. **Query Readout**: Associative dot-product readout $r_\theta(z_t, (u, v)) = b_r + \sum_{i=0}^{127} W_r[i] \cdot z_t[i] \cdot (W_q (u, v))[i]$.
4. **Meta-Training Regime**: BPTT meta-training exclusively on 2-step auxiliary synthetic transitions ($y=1.0$ for observed path, $y=0.0$ for reverse/distractor). **Zero multi-hop ($k \ge 3$) exposure during training.**

### Primary Independent Variable
- **Developmental Path Length ($k$)**: Evaluated across depths $k \in \{2, 3, 4, 5\}$.
  - $k=2$: Canonical 2-hop baseline ($A \to B \to C \implies \text{Query}(A, C)$ vs $\text{Query}(C, A)$)
  - $k=3$: 3-hop zero-shot composition ($A \to B \to C \to D \implies \text{Query}(A, D)$ vs $\text{Query}(D, A)$)
  - $k=4$: 4-hop zero-shot composition ($A \to B \to C \to D \to E \implies \text{Query}(A, E)$ vs $\text{Query}(E, A)$)
  - $k=5$: 5-hop zero-shot composition ($A \to B \to C \to D \to E \to F \implies \text{Query}(A, F)$ vs $\text{Query}(F, A)$)

---

## 3. Preregistered Acceptance Gates & Statistical Thresholds

The experiment evaluates 16 independent random seeds ($\text{seed} \in 0..15$).

| Gate / Estimand | Preregistered Condition / Floor | Statistical Test | Pass Threshold |
| :--- | :--- | :--- | :--- |
| **Gate 1: 2-Hop Retention Floor ($k=2$)** | Directional margin $m_2 = \text{score}(A \to C) - \text{score}(C \to A) > 0.0$ | Exact binomial | $\ge 15 / 16$ seeds ($93.75\%$) |
| **Gate 2: 3-Hop Zero-Shot Generalization ($k=3$)** | Directional margin $m_3 = \text{score}(A \to D) - \text{score}(D \to A) > 0.0$ | Paired sign-flip permutation | $\ge 12 / 16$ seeds ($75.0\%$), $p < 0.01$ |
| **Gate 3: 4-Hop Zero-Shot Generalization ($k=4$)** | Directional margin $m_4 = \text{score}(A \to E) - \text{score}(E \to A) > 0.0$ | Paired sign-flip permutation | $\ge 10 / 16$ seeds ($62.5\%$), $p < 0.05$ |
| **Gate 4: 5-Hop Frontier Characterization ($k=5$)** | Directional margin $m_5 = \text{score}(A \to F) - \text{score}(F \to A)$ | Continuous margin reporting | Continuous empirical floor $\bar{m}_5 > 0.0$ |
| **Gate 5: Multi-Hop Transposition Collapse** | Transposed reversed streams ($D \to C \to B \to A$) produce negative margins $m_{3,\text{rev}} < 0.0$ | Directional specificity check | $\ge 15 / 16$ seeds ($93.75\%$) |
| **Gate 6: Multi-Hop Causal State Surgery ($k=3$)** | Cloned twin swap ($z_{H1(3)} \leftrightarrow z_{H2(3)}$) causally flips 3-hop directional preference | Paired sign-flip permutation | $\ge 12 / 16$ seeds ($75.0\%$), $p < 0.01$ |
| **Gate 7: First-Order Competence Retention** | Contemporaneous 20-trial 1-hop sensor classification accuracy | Baseline accuracy floor | $\ge 90.0\%$ in $\ge 15 / 16$ seeds |
| **Gate 8: Multi-Hop Temporal Shuffle Superiority ($k=3$)** | Intact vs shuffled 3-hop sequence comparison | Exact McNemar paired | $n_{10} - n_{01} \ge 3, p < 0.05$ |
| **Gate 9: Structural Zero-Sidecar Invariant** | External transition store reads $\equiv 0$ | Direct API invariant | $\equiv 0$ sidecar calls ($16/16$) |

---

## 4. Contingency Scouting Protocol (Branch Q17D-B)
- **Contingency Trigger**: If depth generalization collapses at $k=3$ ($< 10/16$ seeds pass Gate 2), execution halts before claiming positive depth scaling.
- **Diagnostic Scout**: A lightweight development-only diagnostic scout will isolate whether the bottleneck stems from:
  1. *State Saturation*: $\tanh$ activations compressing multi-step information.
  2. *Readout Capacity*: Associative dot-product rank limitation.
  3. *Step-Invariant Decay*: Attenuation of initial step $A \to B$ over recursive applications of $W_z$.

---

## 5. Epistemic Scope Ceilings
- **Claim Ceiling**: Claims zero-shot recursive depth composition from short-horizon developmental experience within a recurrent dynamical state.
- **Exclusions**:
  - Does NOT claim general algorithmic graph-search (e.g. Dijkstra, BFS, or arbitrary shortest paths).
  - Does NOT claim general symbolic logic engines or variable-binding architectures.
  - Does NOT claim infinite-depth retention without capacity limits.
