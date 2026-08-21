---
contract_id: CONTRACT-E-Q17C
status: DRAFT
proposed_by: antigravity
design_review: null
reviewed_by: chatgpt-pro
authorized_by: null
base_sha: 1cb1ccce3df3f2ebfef7d53399d1f5dc242b7f0d
execution_base_sha: null
resource_class: cpu
long_running: false
exclusive_gpu: false
interruptible: true
---

# Research Contract Proposal: CONTRACT-E-Q17C

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

## 2. Canonical State Decomposition & Scaffolding Invariant

Under the canonical Recurrence state decomposition ($z_t, M_t, \theta_t, C_t, E_t$):
- **$z_t$ (Latent Recurrent Activation State)**: Allowed to accumulate and carry development-specific trajectory history.
- **$M_t$ (Explicit Memory Buffer)**: $\equiv \emptyset$ (strictly disabled / empty).
- **$\theta_t$ (Synaptic Weights)**: Fixed during test-world evaluation. Pre-trained across auxiliary meta-training worlds so the network learns the recurrent memory algorithm ("weights learn how to remember; activations remember this life").
- **$C_t$ (Transient Computational State)**: Flushed between active decision steps.
- **$E_{\text{sidecar}}$ (External Causal Table)**: **STRICTLY PROHIBITED**.

### Nasty Fail-Fast Rule (Scaffolding Elimination Invariant)
No accumulated pairwise causal table, adjacency matrix, provenance sidecar, episode summary, explicit memory buffer, replay buffer, or query-time reconstruction of developmental history may persist outside $z_t$. The runner and verifier must fail closed if any external history data structure is accessed during evaluation query steps.

---

## 3. Four Core Experimental Conditions & State Surgery Assays

Holding the validated $A \to B \to C$ two-hop task and evaluation challenge episodes fixed, the organism is evaluated across four distinct activation conditions:

### 1. Persistent Recurrent State ($z_{\text{persistent}}$)
- The organism undergoes sequential developmental trajectory experience in the test environment, accumulating history into $z_t$.
- At query time, given visible local cues, the organism must execute two-hop conflict resolution and laundering discrimination directly from $z_t$.

### 2. Latent Reset Control ($z_t \to z_0$)
- Following identical developmental exposure, the hidden state is mechanically reset to baseline $z_0$ prior to the test query.
- Proves that composition performance depends causally on recurrent activation history rather than static weights $\theta$ or visible environment affordances. Performance must collapse to chance.

### 3. Matched-History State Swap Assay ($z_{H1} \leftrightarrow z_{H2}$)
- Two matched organisms develop under divergent causal histories:
  - Organism 1 undergoes History $H_1$: $A \to B \to C$ (directional reachability $A \leadsto C$).
  - Organism 2 undergoes History $H_2$: $C \to B \to A$ (directional reachability $C \leadsto A$).
- At test time, both organisms receive **identical convergent current cues**.
- State surgery performs an exact hidden state swap: $z_{H1} \leftrightarrow z_{H2}$.
- Asserts that swapping $z_t$ causally transfers the history-dependent directional choice ($P(\text{Action}_1 \mid z_{H1}) \neq P(\text{Action}_1 \mid z_{H2})$ and moves with the transplanted state).

### 4. Matched Shuffled-History Control ($z_{\text{shuffled}}$)
- An organism is exposed to temporally shuffled transitions with identical marginal statistics.
- Proves that internal recurrent organization in $z_t$ encodes true sequential causal ordering rather than non-specific activation energy or scalar familiarity.

---

## 4. Frozen Acceptance Criteria & Statistical Gates

| Gate / Estimand | Formal Definition & Evaluated Condition | Pre-registered Acceptance Threshold |
| :--- | :--- | :--- |
| **Gate 1: Endogenous Two-Hop Conflict** | Conflict resolution accuracy with persistent $z_t$ and zero external sidecars | $\ge 10/16$ seeds ($62.5\%$) |
| **Gate 2: Endogenous Laundering Discrimination** | Laundering discrimination accuracy with persistent $z_t$ | $\ge 10/16$ seeds ($62.5\%$) |
| **Gate 3: Latent Reset Lesion Effect** | Performance drop under $z_t \to z_0$: $a_{\text{persistent}} - a_{\text{reset}}$ | $\Delta a \ge 0.40$ in $\ge 12/16$ seeds ($p < 0.01$) |
| **Gate 4: Causal State Swap Transfer** | Behavioral transfer under $z_{H1} \leftrightarrow z_{H2}$ in convergent-cue assay | $\ge 12/16$ seeds predict transfer ($p < 0.01$) |
| **Gate 5: Temporal Shuffle Control Superiority** | McNemar paired superiority over matched $z_{\text{shuffled}}$ control | $n_{10} - n_{01} \ge 3, p < 0.05$ |
| **Gate 6: Zero-Sidecar Invariant** | Verification that no external matrix $E$ or replay buffer was allocated | $\equiv 0$ sidecar accesses ($100\%$ verified) |

---

## 5. Epistemic Boundaries & Strict Claim Ceiling
- **Claim**: Development-specific causal history can be encoded in persistent recurrent activation state $z_t$ and exert causal control over previously validated composition-dependent behavior, without an external causal-history store.
- **Exclusions**:
  - Does NOT claim arbitrary $N$-hop graph reasoning ($N \ge 3$ is reserved for Q17D).
  - Does NOT claim long-horizon autobiographical memory consolidation across competing multi-day task interference (reserved for Q17E).
  - Does NOT claim autonomous discovery of composition architecture.
