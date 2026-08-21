---
contract_id: CONTRACT-E-Q17C
status: DRAFT
proposed_by: antigravity
design_review: APPROVED
reviewed_by: chatgpt-pro
authorized_by: null
base_sha: 1cb1ccce3df3f2ebfef7d53399d1f5dc242b7f0d
execution_base_sha: null
resource_class: cpu
long_running: false
exclusive_gpu: false
interruptible: true
---

# Research Contract Proposal: CONTRACT-E-Q17C (Hardened)

## Title
Gate E Frontier: Endogenous Recurrent Causal History & Latent State Surgery for Zero-Shot Composition

## 1. Context & Research Question
In exploration stage Q17A, neural composition was demonstrated under explicit two-hop supervision using an external transition table. In stage Q17B, composition was shown to emerge from self-supervised trajectory experience paired with empirical transition statistics, while still relying on an externally persisted transition matrix $E$ to supply input pairs $(e_{AB}, e_{BC})$ to the learned kernel $f_\theta$.

Stage Q17C isolates the next critical scaffolding migration: **removing the external causal-history sidecar entirely** and encoding development-specific transition history within the organism's persistent recurrent activation state $z_t$.

```
Q17B (External Memory Sidecar):
developmental experience -> external causal matrix E -> learned kernel f_theta(e_AB, e_BC) -> behavior

Q17C (Endogenous Recurrent Memory):
developmental experience -> persistent recurrent state z_t -> learned readout -> behavior
(No external E matrix, transition table, or episode buffer persisted at query time)
```

### Core Research Question
Can the causal history required for previously validated two-hop composition be stored in persistent recurrent activation state $z_t$, such that the organism retains composition-capable behavior after all external causal-history sidecars are removed?

---

## 2. Recurrent Mechanism & Auxiliary Meta-Training

### Architectural Formulation
1. **Recurrent State Update**:
   $$z(t+1) = g_\theta(z(t), x(t))$$
   where $x(t)$ contains **ONLY** the current local transition observation $(s_t, a_t, s_{t+1})$. Past history $x(1:t-1)$ is strictly unavailable except insofar as represented dynamically within $z(t)$.
2. **Composition Readout**:
   $$m = r_\theta(z(t), q_{\text{current}})$$
   where $q_{\text{current}}$ is the active test query and $m$ is the continuous directional composition margin:
   $$m = \text{score}(A \to C) - \text{score}(C \to A)$$

### Scaffolding Invariant: Weights vs. Activations
- **Synaptic Weights ($\theta$)**: Pre-trained **exclusively on auxiliary synthetic worlds** using the Q17B self-supervised trajectory future-outcome objective.
- **Sealed Test Isolation**: Test-world $A/B/C$ developmental histories **NEVER update $\theta$**.
- **Epistemic Principle**: *"Weights learn how to remember; activations remember this particular life."*

---

## 3. Structural Zero-Sidecar Enforcement

Under the canonical Recurrence state decomposition ($z_t, M_t, \theta_t, C_t, E_t$):
- **$z_t$ (Latent Recurrent Activation State)**: Allowed to accumulate and carry development-specific trajectory history.
- **$M_t$ (Explicit Memory Buffer)**: $\equiv \emptyset$ (strictly disabled / empty).
- **$\theta_t$ (Synaptic Weights)**: Fixed during test-world evaluation.
- **$C_t$ (Transient Computational State)**: Flushed between active decision steps.
- **$E_{\text{sidecar}}$ (External Causal Table)**: **STRICTLY PROHIBITED**.

### Structural API Invariant
The test-time query interface must strictly adhere to:
```rust
fn query(z: &RecurrentState, current_obs: &Observation, weights: &ModelWeights) -> ReadoutMargin;
```
No developmental history object, episode buffer, adjacency matrix, or transition ledger is in scope or accessible. Any query-time attempt to access external historical logs throws and immediately fails closed.

---

## 4. Matched-Twin Experimental Assays & State Surgery

Holding the validated $A \to B \to C$ two-hop task and challenge episodes fixed, cloned twin organisms (identical $\theta$, initial $z_0$, action menus, marginal experience distributions, and episode lengths) are evaluated across four decisive conditions:

### 1. Persistent State ($z_{\text{persistent}}$)
- Organism undergoes sequential developmental trajectory experience in the test environment, accumulating history into $z_t$.
- Evaluates zero-shot multi-hop conflict resolution and laundering discrimination directly from $z_t$.

### 2. Latent Reset ($z_t \to z_0$)
- Hidden state is mechanically reset to $z_0$ prior to the test query.
- Proves performance causally depends on recurrent activation history rather than static weights $\theta$ or visible affordances. Performance must drop near chance.

### 3. Matched-History State Swap ($z_{H1} \leftrightarrow z_{H2}$)
- Evaluates two matched twin organisms developed under divergent causal histories:
  - **History $H_1$**: $A \to B \to C$ (directional reachability $A \leadsto C \implies z_{H1}$, orientation $s(H_1) = +1$).
  - **History $H_2$**: $C \to B \to A$ (directional reachability $C \leadsto A \implies z_{H2}$, orientation $s(H_2) = -1$).
- Both organisms receive **identical convergent current cues** at test time.
- State surgery performs an exact latent swap: $z_{H1} \leftrightarrow z_{H2}$.
- Asserts that swapping $z_t$ causally transfers the history-dependent directional choice:
  - Organism 1 ($H_1$) with donor $z_{H2} \implies$ exhibits reverse preference ($C \leadsto A$).
  - Organism 2 ($H_2$) with donor $z_{H1} \implies$ exhibits forward preference ($A \leadsto C$).

### 4. Same-History State Swap & Specificity Controls
- **Same-History Swap Control**: Swapping hidden states between identical-history clones ($z_{H1,\text{clone } A} \leftrightarrow z_{H1,\text{clone } B}$) maintains behavioral stability ($\ge 15/16$ seeds unchanged).
- **Competence Preservation**: State reset or swap must not globally impair unrelated contemporaneous first-order task performance ($\ge 15/16$ seeds maintain $\ge 90\%$ baseline accuracy, proving selective memory surgery rather than generalized network damage).
- **Matched Shuffled-History Control ($z_{\text{shuffled}}$)**: Exposure to temporally shuffled transitions with identical marginal statistics confirms sequential causal organization.

---

## 5. Frozen Acceptance Criteria & Exact Statistical Gates

| Gate / Estimand | Formal Definition & Evaluated Condition | Pre-registered Acceptance Threshold |
| :--- | :--- | :--- |
| **Gate 1: Endogenous Two-Hop Conflict** | Conflict resolution accuracy with persistent $z_t$ and zero external sidecars | $\ge 10/16$ seeds ($62.5\%$) |
| **Gate 2: Endogenous Laundering Discrimination** | Laundering discrimination accuracy with persistent $z_t$ | $\ge 10/16$ seeds ($62.5\%$) |
| **Gate 3: Continuous Latent Reset Lesion Effect** | Continuous margin drop $\Delta_{\text{reset}} = m_{\text{persistent}} - m_{\text{reset}}$ with reset approaching chance ($7\text{--}9/16$) | Exact 16-seed paired sign-flip $p < 0.01$ |
| **Gate 4: Continuous Donor-State Swap Transfer** | Behavioral transfer ($\ge 12/16$ seeds) and donor-aligned continuous effect $\Delta_{\text{swap},i} = \frac{1}{2} [s(H_2)(m_{H1 \leftarrow H2} - m_{H1\text{-own}}) + s(H_1)(m_{H2 \leftarrow H1} - m_{H2\text{-own}})]$ | Exact 16-seed paired sign-flip $p < 0.01$ |
| **Gate 5: Same-History Swap Stability** | Behavioral consistency under identical-history twin swap ($z_{H1,A} \leftrightarrow z_{H1,B}$) | $\ge 15/16$ seeds ($\le 1/16$ deviations) |
| **Gate 6: First-Order Competence Preservation** | Unrelated contemporaneous 1-hop sensor task accuracy under state surgery | $\ge 15/16$ seeds maintain $\ge 90.0\%$ |
| **Gate 7: Temporal Shuffle Superiority** | McNemar paired superiority over matched $z_{\text{shuffled}}$ control | $n_{10} - n_{01} \ge 3, p < 0.05$ |
| **Gate 8: Structural Zero-Sidecar Invariant** | Verification that query interface received only $(z_t, q_{\text{current}}, \theta)$ with zero sidecar accesses | $\equiv 0$ sidecar accesses ($100\%$ verified) |

---

## 6. Pre-Sealing Protocol & Epistemic Scope Ceilings
- **Pre-Sealing**: Architecture, hidden-state dimension ($d=128$), optimizer, stopping rules, auxiliary meta-training worlds, and state surgery protocols must be frozen before opening test-world histories.
- **Claim Ceiling**: Development-specific causal history can be encoded in persistent recurrent activation state $z_t$ and exert causal control over previously validated composition-dependent behavior, without an external causal-history store.
- **Exclusions**:
  - Does NOT claim an abstract causal self-model or symbolic reasoning engine.
  - Does NOT claim arbitrary $N$-hop graph reasoning ($N \ge 3$ reserved for Q17D).
  - Does NOT claim long-horizon autobiographical memory consolidation across competing multi-day task interference (reserved for Q17E).
