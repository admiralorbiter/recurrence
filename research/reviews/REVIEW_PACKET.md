# Evidence Review Packet: Recurrence (recurrence)
**Active Contract**: `None`
**Repository HEAD**: `e4bef76cc3e9ba4bfbab6123097a48240e25818c`

## 1. Project Context & Moonshot (from BOOTSTRAP.md)
# BOOTSTRAP.md — Recurrence Project Manual & Conceptual Atlas

**Project**: `recurrence` (Continuity Garden)  
**Moonshot**: First causal developmental atlas of an artificial self-model.  
**Repository**: `https://github.com/admiralorbiter/recurrence`  
**Governance Contract**: [`AGENTS.md`](https://github.com/admiralorbiter/mother-base/blob/main/AGENTS.md) — *Projects own truth. Mother Base owns operations.*

---

## 1. The Core Scientific Moonshot
The objective of Continuity Garden (`recurrence`) is to discover the minimal developmental and architectural inductive biases required for an artificial neural organism to endogenously acquire, maintain, and recursively extend a causal self-model of its own latent transitions across time without external topological supervision.

---

## 2. Conceptual Vocabulary & Mathematical Formalism

### A. The Developmental Substrate & Environment
- **Transition Observations $x_t \in \mathbb{R}^4$**:
  $$x_t = \left[ \frac{\text{src}}{5} + \xi, \, \frac{\text{action}}{5}, \, \frac{\text{dst}}{5} + \xi, \, 1.0 \right]^T$$
  where $\xi$ is independent observation noise jitter.
- **Relational Composition ($k$-hop reachability)**:
  An agent observes a developmental trajectory of $k$ contiguous transitions:
  $$S_k = \left( u \xrightarrow{a_1} v_1 \xrightarrow{a_2} v_2 \cdots \xrightarrow{a_k} w \right)$$
- **Relational Query Head $r_\theta(z, (s, d))$**:
  Evaluates whether state $z$ causally encodes reachability from start node $s$ to destination node $d$:
  $$r_\theta(z, (s, d)) = b_r + \sum_{i=1}^{d} w_{ri} z_i \left( w_{qi1} \frac{s}{5} + w_{qi2} \frac{d}{5} \right)$$
  Forward Directional Margin: $m_k = r_\theta(z_k, (u, w)) - r_\theta(z_k, (w, u))$.

---

## 3. Causal Evidence Standards & Assays

To qualify as genuine causal internal representation rather than lexical correlation or representation drift:
1. **Positive Directional Margin ($m_k > 0$)**:
   The intact trajectory state must predict forward reachability $(u, w)$ higher than reversed reachability $(w, u)$, statistically validated via sign-flip permutation test ($p < 0.01$).
2. **Causal State Surgery & Swap Effect ($\Delta_{\text{swap}}$)**:
   Transplanting the hidden state of an independent donor stream (e.g. reversed path with identical action vocabulary and separate jitter) into the organism must causally flip the endpoint query decision ($m_{\text{donor}} < 0$, $\Delta_{\text{swap}} = m_{\text{intact}} - m_{\text{donor}} > 0$).
3. **Transposition Reversals**:
   A trajectory with reversed transitions must produce negative directional margin ($m_{\text{trans}} < 0$).
4. **Deranged Shuffle Superiority**:
   The intact contiguous sequence must produce a higher directional score than permuted orderings (e.g. $[e_2, e_3, e_1]$).
5. **Sensor Competence Preservation**:
   Readout of 1-hop sensor probes must maintain $\ge 90\%$ accuracy across developmental stages.
6. **Task-Aligned Jacobian Sensitivities ($S_{\text{early}}, S_{\text{late}}$)**:
   $$S_{\text{early}} = \left\| \frac{\partial m_k}{\partial x_1} \right\|_2 = \left\| \frac{\partial m_k}{\partial z_k} \cdot \frac{\partial z_k}{\partial x_1} \right\|_2, \quad S_{\text{late}} = \left\| \frac{\partial m_k}{\partial x_k} \right\|_2$$

---

## 4. Lineage Progression: Q16 $\to$ Q17A–E

- **Q16 (Plastic Self-Model Foundations)**: Proved dual-locus neuromodulated plasticity sustains localized self-representations under environmental perturbation.
- **Q17A (Neural Composition Kernel)**: Established that a learned composition kernel can perform 2-hop composition when given adjacent relation vectors.
- **Q17B (Endogenous Induction)**: Showed that trajectory-derived supervision induces composition-capable operators without explicit 2-hop reachability labels.
- **Q17C (2-Hop Developmental Organism — PROMOTED)**:
  - 120-epoch meta-trained Simple RNN on 2-step sequences ($u \to v \to w$).
  - Achieved $16/16$ $k=2$ composition with positive causal state surgery.
- **Q17D (Depth Horizon Generalization — PROMOTED MIXED/BOUNDED)**:
  - Tested zero-shot generalization across horizons $k=2, 3, 4, 5$.
  - Demonstrated horizon boundary at $k=3$ ($6/16$ direction, $0/16$ state surgery) under exact 120-epoch training baseline commit `fa7ebb8`.
- **Q17E Scout Lineage (Mechanism Discovery)**:
  - **Scout A/B**: Proved geometric Jacobian decay ($\sim 42\%$ loss per step).
  - **Scout 2 ($\alpha$-preactivation residual)**: Falsified static identity carry.
  - **Scout C/D (Adaptive Update Gating)**: Falsified UGRNN gating and raw history reconstruction.
  - **Scout E (Shared Relational-Prefix Supervision)**: Proved shared prefix supervision on monolithic RNN causes recency dominance ($S_{\text{late}} \approx 11.4$).
  - **Scout F (Typed Relational State)**: Separated local edge $e_t \in \mathbb{R}^{32}$ from relational accumulator $m_t \in \mathbb{R}^{96}$; revealed two-layer $\tanh$ trainability barrier.
  - **Scout G (Typed Residual Accumulator — BREAKTHROUGH)**: Linear edge encoder + Additive residual accumulator ($m_{t+1} = m_t + \tanh(W_m m_t + W_c e_t + b_m)$) achieved $16/16$ $k=2$ and **$15/16$ ($93.8\%$, $p=0.0002$) zero-shot $k=3$ composition** with complete symmetric double dissociation ($+30.27$ vs $-1.14$).

---

## 5. Falsified Hypotheses / DO NOT REOPEN
1. **Convex Gated Carry ($z_{t+1} = g z_t + (1-g)\tilde{z}$)**: Suppresses incoming evidence under 2-step training; fails to open.
2. **Static Identity Carry ($z_{t+1} = \alpha z_t + \tilde{z}$)**: Preserves Jacobian norm without preserving task-relevant relational subspace.
3. **Raw History Reconstruction ($\|D z_2 - x_1\|^2$)**: Preserves coordinate features, but fails to organize relational composition.
4. **Shared Prefix Supervision on Monolithic RNN**: Perfects $k=2$ but induces recency dominance ($S_{\text{late}}$ doubles), collapsing $k=3$ to $0/16$.
5. **Two-Layer Saturated $\tanh$ Composition without Identity Path**: Suffers severe gradient attenuation during early developmental training.

---

## 6. Canonical Repositories & Evidence Paths
- **Durable Checkpoints**: `research/checkpoints/`
- **Research Contracts**: `research/contracts/`
- **Promotion Records**: `research/promotions/`
- **Raw Telemetry**: `crates/continuity_garden_core/data/`
- **Core Implementation**: `crates/continuity_garden_core/src/`


---

## 2. Promoted & Completed Checkpoints
### CHECKPOINT-E-Q17A-R1.md
---
checkpoint_id: CHECKPOINT-E-Q17A-R1
contract_id: CONTRACT-E-Q17A-R1
status: PROMOTED
promoted_sha: efc2d9941bb546a28fc01ff634211e79070a5bae
created_at: "2026-08-21 22:16:00Z"
---

# Research Checkpoint: Q17A Endogenous 2-Hop Transitive Composition (Promoted)

## 1. Verified Scientific State
- **Hypothesis Confirmed**: Endogenous neural composition kernels $f_\theta(e_{AB}, e_{BC}) \to a_{AC}$ generalize to withheld multi-hop causal endpoints without explicit graph traversal.
- **Empirical Baseline**:
  - Multi-hop zero-shot conflict: 16/16 seeds ($100\%$)
  - Laundering discrimination: 16/16 seeds ($100\%$)
  - Composition ablation drop: 16/16 seeds ($p = 1.5259 \times 10^{-5}$)
  - Directional Transposition Collapse: 0/16 seeds passed on conflict, mean return $-0.995$.
- **Next Frontier**: Q17B — Self-Supervised Composition Discovery (learning composition without explicit auxiliary two-hop targets).



### CHECKPOINT-E-Q17B.md
---
checkpoint_id: CHECKPOINT-E-Q17B
contract_id: CONTRACT-E-Q17B
promotion_id: PROMOTION-CONTRACT-E-Q17B
timestamp: "2026-08-21 23:12:00Z"
base_sha: da925179bbe769d9da544239c6db9604fcbad243
status: PROMOTED
authorized_by: human
---

# Checkpoint Record: CHECKPOINT-E-Q17B (Self-Supervised Endogenous Composition)

## 1. Verified Scientific State
- **Contract Promoted**: `CONTRACT-E-Q17B`
- **Promotion Artifact**: [`research/promotions/PROMOTION-CONTRACT-E-Q17B.md`](../promotions/PROMOTION-CONTRACT-E-Q17B.md)
- **Verified Code Baseline**: `da925179bbe769d9da544239c6db9604fcbad243`
- **Empirical Confirmation**:
  - Training exclusively on local empirical transition frequencies ($\hat{e}_1, \hat{e}_2$) and self-supervised two-step trajectory prediction induces zero-shot 2-hop composition ($16/16$ seeds pass Gate 1 and Gate 2).
  - Intact temporally aligned training demonstrates statistically significant superiority over a 1-to-1 matched shuffled negative control with identical input marginals and identical target sum ($n_{10}=13, n_{01}=0, \Delta=13, p=1.2207 \times 10^{-4}$).
  - Mechanistic specificity confirmed via continuous lesion delta sign-flip permutation test ($p = 1.5259 \times 10^{-5}$).
  - Directional transposition falsified ($0/16$ pass under $A^T$, mean return = $-1.000$) while preserving circular consistency under $A^T$ ($16/16$ pass Gate 5).

## 2. Epistemic Boundaries & Scope Ceilings
- **Established**: Temporally aligned self-supervised trajectory experience produces a reliable improvement in composition-capable behavior over an otherwise matched temporally shuffled learning condition.
- **Unresolved / Next Frontier**: Arbitrary $N$-hop relational composition and lifetime memory consolidation over long-horizon developmental trajectories (deferred to Q17C).



### CHECKPOINT-E-Q17C.md
---
checkpoint_id: CHECKPOINT-E-Q17C
contract_id: CONTRACT-E-Q17C
promotion_id: PROMOTION-CONTRACT-E-Q17C
timestamp: "2026-08-22 00:40:00Z"
base_sha: b0af2e13e4118564c72b0d004b7e2d54170657d2
status: PROMOTED
authorized_by: human
---

# Checkpoint Record: CHECKPOINT-E-Q17C (Endogenous Recurrent Causal History & State Surgery)

## 1. Verified Scientific State
- **Contract Promoted**: `CONTRACT-E-Q17C`
- **Promotion Artifact**: [`research/promotions/PROMOTION-CONTRACT-E-Q17C.md`](../promotions/PROMOTION-CONTRACT-E-Q17C.md)
- **Verified Code Baseline (`candidate_sha`)**: `b0af2e13e4118564c72b0d004b7e2d54170657d2`
- **Empirical Confirmation**:
  - **Endogenous Storage**: Developmental causal history is successfully carried endogenously in persistent recurrent activation state $z_t \in \mathbb{R}^{128}$ ($16/16$ seeds pass directional conflict in Gate 1 and 4-node laundering discrimination in Gate 2) without an external transition table or sidecar ledger.
  - **Causal State Surgery**: In cloned twin organisms sharing identical parameters $\theta$ and test-time queries, swapping latent states ($z_{H1} \leftrightarrow z_{H2}$) causally flips behavioral preference in the donor-consistent direction ($16/16$ transfer, exact paired sign-flip permutation $p = 1.5259 \times 10^{-5}$).
  - **Mechanistic Lesion**: Latent reset ($z \to 0$) drops directional margins to chance across 20 real choice trials ($15/16$ seeds near chance, $p = 1.5259 \times 10^{-5}$).
  - **First-Order Competence**: Contemporaneous 20-trial 1-hop sensor classification accuracy is preserved ($\ge 90.0\%$ in $16/16$ seeds) both before and after state swap.
  - **Temporal Alignment Superiority**: Temporally aligned learner dominates the shuffled negative control ($\Delta = +16, p = 3.0518 \times 10^{-5}$).
  - **Structural Zero-Sidecar Invariant**: Verified $\equiv 0$ external sidecar or ledger reads during online execution ($16/16$).

## 2. Epistemic Boundaries & Scope Ceilings
- **Established**: Within this controlled synthetic two-hop environment, developmental causal history can be carried in persistent recurrent activation state rather than an external causal-history store, and manipulating that state causally changes query-conditioned composition behavior while preserving an unrelated first-order capability.
- **Exclusions / Ceilings**: Does not claim an autobiographical memory, an abstract causal model, arbitrary symbolic reasoning, or entity-independent causal composition. Roles and coordinate geometry remain shared across auxiliary and test domains.
- **Unresolved / Next Frontier**: Out-of-distribution depth generalization across unseen $k$-hop paths ($k \in \{3, 4, 5\}$) trained strictly on 2-step sequences (advanced to `CONTRACT-E-Q17D`).



### CHECKPOINT-E-Q17D.md
---
checkpoint_id: CHECKPOINT-E-Q17D
contract_id: CONTRACT-E-Q17D
promotion_id: PROMOTION-CONTRACT-E-Q17D
timestamp: "2026-08-22 02:22:00Z"
base_sha: fa7ebb857187429188e404b9015c7e8a9394602f
status: PROMOTED
authorized_by: human
---

# Checkpoint Record: CHECKPOINT-E-Q17D (Multi-Hop Depth Dissociation & Endpoint Extrapolation)

## 1. Verified Scientific State
- **Contract Promoted**: `CONTRACT-E-Q17D`
- **Promotion Artifact**: [`research/promotions/PROMOTION-CONTRACT-E-Q17D.md`](../promotions/PROMOTION-CONTRACT-E-Q17D.md)
- **Verified Code Baseline (`candidate_sha`)**: `fa7ebb857187429188e404b9015c7e8a9394602f`
- **Empirical Confirmation**:
  - **Global Validity V1–V4**: Full 8-tensor SHA-256 parameter hashes verified ($16/16$), canonical 2-hop baseline retained ($16/16, 100.0\%$), contemporaneous 20-trial 1-hop sensor competence retained ($16/16, 100.0\%$), and zero sidecar reads maintained ($16/16$).
  - **Coordinate-OOD Controls $C_3, C_4, C_5$**: 2-hop transitions utilizing extended role coordinates $D=4, E=5, F=6$ pass with $100\%$ precision ($16/16$ per control). This confirms that out-of-distribution coordinate representation is not the cause of multi-hop breakdown.
  - **Score-Level Extrapolation Dissociation**: Raw endpoint directional margins extrapolate positively beyond 2-step training ($12/16$ at $k=4, p = 6.88 \times 10^{-3}$), but fail causal state-surgery transfer ($0/16$), reversal collapse ($6/16$ at $k=3$), and temporal shuffle superiority ($12/16$).
  - **Classification Verdict**: `NON_MONOTONIC_ANOMALOUS_DEPTH_PROFILE`.

## 2. Epistemic Boundaries & Core Scientific Belief
- **Established**: Under the frozen Q17C recurrent architecture, endpoint-directional scores can extrapolate beyond two-step training—including strongly at four steps—but the longer-horizon responses fail preregistered causal state-surgery, reversal, and temporal-order controls. Therefore Q17D does not establish recursive compositional depth scaling; it reveals a dissociation between score-level extrapolation and causally grounded developmental-history composition.
- **Unresolved / Immediate Frontier**: Diagnostic Scout `Q17D-B` to probe zero-history / query-only baselines, query-readout vs recurrent state geometric contributions, $k=4$ state swaps, and initial-step Jacobian attenuation.



## 3. Latest Promotion Records
### PROMOTION-CONTRACT-E-Q17A-R1.md
---
promotion_id: PROMOTION-CONTRACT-E-Q17A-R1
contract_id: CONTRACT-E-Q17A-R1
status: PROMOTED
candidate_sha: efc2d9941bb546a28fc01ff634211e79070a5bae
promoted_at: "2026-08-21 22:16:00Z"
repair_rounds: 2
reviewed_by: chatgpt-pro
authorized_by: human
---

# Promotion Record: PROMOTION-CONTRACT-E-Q17A-R1 (Promoted)

**Lifecycle Status**: `PROMOTED` (Authorized by Human Research Director & ChatGPT Pro Scientific Review Desk)

## 1. Execution & Audit Summary
- **Target Contract**: `CONTRACT-E-Q17A-R1`
- **Phase / Milestone**: Gate E Frontier (Q17A — Endogenous 2-Hop Transitive Composition)
- **Promoted Candidate Commit SHA**: `efc2d9941bb546a28fc01ff634211e79070a5bae`
- **Contract Acceptance Verifier**: `PASS` (`verify_contract_q17a` executed and cleanly passed)
- **Scientific Review Verdict**: `APPROVED`
- **Governance**: Human Director Promotion Merge.

## 2. Experimental Verification & Gate Audit

| Gate / Estimand | Pre-registered Floor | Observed Empirical Result | Status |
| :--- | :--- | :--- | :--- |
| **Gate 1 (Zero-Shot Multi-Hop Conflict)** | $\ge 12/16$ seeds ($75.0\%$) | **16/16 seeds** ($100.0\%$) | **PASS** |
| **Gate 2 (Zero-Shot Laundering Discrimination)** | $\ge 11/16$ seeds ($68.75\%$) | **16/16 seeds** ($100.0\%$) | **PASS** |
| **Gate 3 (Independent Corroboration)** | $\ge 15/16$ seeds ($93.75\%$) | **16/16 seeds** ($100.0\%$) | **PASS** |
| **Gate 4 (Independent Conflict)** | $\ge 15/16$ seeds ($93.75\%$) | **16/16 seeds** ($100.0\%$) | **PASS** |
| **Gate 5 (Composition Ablation Floor)** | $n_{10} - n_{01} \ge 3$ | **$n_{10}=16, n_{01}=0, \Delta=16$** ($p = 1.5259 \times 10^{-5}$) | **PASS** |
| **Gate 6 (Mechanistic Path-Break Specificity)** | $p < 0.01$, $A/D \ge 15/16$ | **$p = 1.5259 \times 10^{-5}$**, $A/D = 16/16$ | **PASS** |
| **Transposition Falsification ($A \neq C$)** | $\le 2/16$ seeds, return $< 0.00$ | **0/16 seeds passed, mean return = -0.995** | **PASS** |
| **Transposition Laundering ($A = C$)** | $\ge 10/16$ seeds | **16/16 seeds** | **PASS** |

## 3. Epistemic Belief Update & Narrow Claim Ceiling
- **Empirical Belief Update**: A learned parameterized neural function ($f_\theta$) can replace the fixed two-hop algebraic composition operator in this assay, generalize to the withheld $A \to C$ endpoint, and preserve the behavioral effect while exhibiting the required causal lesion and directionality signatures.
- **Strict Exclusions**: Does NOT show that the architecture independently discovered the need for composition (the kernel was trained with explicit auxiliary two-hop targets). Self-supervised discovery is the explicit frontier of Q17B.



### PROMOTION-CONTRACT-E-Q17B.md
---
promotion_id: PROMOTION-CONTRACT-E-Q17B
contract_id: CONTRACT-E-Q17B
status: PROMOTED
candidate_sha: da925179bbe769d9da544239c6db9604fcbad243
generated_at: "2026-08-21 23:12:00Z"
repair_rounds: 2
reviewed_by: chatgpt-pro
authorized_by: human
---

# Promotion Record: PROMOTION-CONTRACT-E-Q17B (Self-Supervised Endogenous Composition)

**Lifecycle Status**: `PROMOTED` (Authorized by Human Research Director following Scientific Review)

## 1. Execution & Audit Summary
- **Target Contract**: `CONTRACT-E-Q17B`
- **Phase / Milestone**: Gate E Frontier (Q17B — Self-Supervised Endogenous Composition)
- **Candidate Branch**: `mb/CONTRACT-E-Q17B`
- **Candidate Commit SHA**: `da925179bbe769d9da544239c6db9604fcbad243`
- **Contract Acceptance Verifier**: `PASS` (`verify_contract_q17b` executed and cleanly passed with continuous delta permutation test and transposed laundering arm)
- **Scientific Review Desk Verdict**: `APPROVED`
- **Human Director Verdict**: `PROMOTED`
- **Audit Findings**:
  - Training dataset $D$ ($N = 2500$) matched 1-to-1 against $D_{\text{shuffled}}$ with identical input marginal distributions, sample sizes, and identical target sums ($N_{\text{target}}=10,072$).
  - Separate transposed laundering arm confirmed circular consistency preservation under $A^T$.
  - Exact sign-flip permutation test evaluated directly on continuous path-break deltas $\Delta a_i$ confirmed mechanistic specificity ($p = 1.5259 \times 10^{-5} < 0.01$).
  - 13–0 superiority observed over the matched temporally shuffled negative control ($p = 1.2207 \times 10^{-4} < 0.05$).

## 2. Empirical Verification & Gate Audit

| Gate / Estimand | Pre-registered Floor | Observed Empirical Result | Status |
| :--- | :--- | :--- | :--- |
| **Dataset Matched Control Integrity** | $N_{\text{samples}}=2500$, Independently Aggregated | **$N=2500$, Target Sum: 10,072 Intact vs 10,072 Shuffled (EXACT MATCH)** | **PASS** |
| **Gate 1 (Zero-Shot Multi-Hop Conflict)** | $\ge 10/16$ seeds ($62.5\%$) | **16/16 seeds** ($100.0\%$) | **PASS** |
| **Gate 2 (Laundering Discrimination)** | $\ge 10/16$ seeds ($62.5\%$) | **16/16 seeds** ($100.0\%$) | **PASS** |
| **Gate 3 (Temporal Shuffle Control Superiority)** | $n_{10} - n_{01} \ge 3, p < 0.05$ | **$n_{10}=13, n_{01}=0, \Delta=13$** ($p = 1.2207 \times 10^{-4}$) | **PASS** |
| **Gate 4 (Directional Transposition Falsification)** | $\le 2/16$ seeds, return $< 0.00$ | **0/16 seeds passed, mean return = -1.000** | **PASS** |
| **Gate 5 (Transposition Laundering Arm Invariant)** | $\ge 10/16$ seeds under $A^T$ | **16/16 seeds** ($100.0\%$) | **PASS** |
| **Gate 6 (Mechanistic Continuous Delta Permutation)** | $p < 0.01$ on continuous $\Delta a_i$ | **$p = 1.5259 \times 10^{-5}$** | **PASS** |
| **Supervised Reference (Q17A)** | Upper benchmark reference | **16/16 parity maintained** | **INFORMATIONAL** |

## 3. Strict Epistemic Boundaries & Narrow Claim Ceiling
- **Claim**: Temporally aligned self-supervised trajectory experience produces a reliable improvement in composition-capable behavior over an otherwise matched temporally shuffled learning condition without explicit two-hop reachability labels.
- **Scientific Nuance**: Even without temporal alignment, the shuffled control learner achieves substantial baseline conflict accuracy ($\sim 0.90\text{--}1.0$), demonstrating that marginal transition statistics provide significant structural signal. However, temporally ordered trajectory pairing yields a statistically significant and mechanistic improvement ($\Delta = +13, p = 1.2207 \times 10^{-4}$).
- **Exclusions**: Does NOT claim that temporal alignment is strictly required for composition. Does NOT claim that the architecture autonomously discovered that composition exists (the kernel is architecturally handed adjacent transition pairs). Does NOT claim arbitrary $N$-hop path planning or lifetime memory consolidation.



### PROMOTION-CONTRACT-E-Q17C.md
---
promotion_id: PROMOTION-CONTRACT-E-Q17C
contract_id: CONTRACT-E-Q17C
status: PROMOTED
candidate_sha: b0af2e13e4118564c72b0d004b7e2d54170657d2
generated_at: "2026-08-22 00:28:00Z"
repair_rounds: 2
reviewed_by: chatgpt-pro
authorized_by: human
---

# Verified Promotion Record: PROMOTION-CONTRACT-E-Q17C (Endogenous Recurrent Memory & State Surgery)

**Lifecycle Status**: `PROMOTED` (Scientific Promotion Review APPROVED by ChatGPT Pro; Strategic Promotion Authorized by Human Research Director)

## 1. Execution & Audit Summary
- **Target Contract**: `CONTRACT-E-Q17C`
- **Phase / Milestone**: Gate E Frontier: Endogenous Recurrent Causal History
- **Candidate Branch**: `mb/CONTRACT-E-Q17C`
- **Scientific Candidate Commit SHA**: `b0af2e13e4118564c72b0d004b7e2d54170657d2`
- **Execution Base SHA**: `ecb24762988a4727076c9fc42a04f9bd52a4a2fc`
- **Contract Acceptance Verifier**: `PASS` (`verify_contract_q17c` executed independently with exact paired sign-flip permutation tests, donor-aligned state swap transfer assertions, same-history swap controls with independent nuisance sampling, real 20-trial 1-hop sensor task accuracy preservation against gold truth, genuine shuffled control, and structural zero-sidecar API checks directly from raw event telemetry)
- **Auditor Verdict**: `PASS` (All 8 frozen statistical gates independently recomputed from raw event telemetry across 16 seeds with frozen architecture $d=128$)
- **Repair Iterations**: 2 rounds (corrected BPTT meta-training with self-supervised objective, query-conditioned associative readout $r_\theta(z_t, q)$, genuinely distinct Gate 1 / Gate 2 challenge worlds, 20-trial choice records for latent reset, and 20-trial sensor classifications against gold labels).

## 2. Frozen Statistical Gates & Empirical Outcomes

| Gate / Estimand | Preregistered Condition / Floor | Observed Empirical Result | Statistical Test | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Gate 1: Endogenous 2-Hop Conflict** | Persistent $z_t \ge 10/16$ ($62.5\%$) | **16 / 16 seeds ($100.0\%$)** | Exact binomial | **PASS** |
| **Gate 2: Endogenous Laundering** | Persistent $z_t \ge 10/16$ ($62.5\%$) | **16 / 16 seeds ($100.0\%$)** | Exact binomial | **PASS** |
| **Gate 3: Continuous Latent Reset Lesion** | $\Delta_{\text{reset}} = m_{\text{pers}} - m_{\text{reset}}$, $p < 0.01$, near-chance $\ge 12/16$ | **16/16 drop ($100\%$), 15/16 near chance** | Paired sign-flip $p = 1.5259 \times 10^{-5}$ | **PASS** |
| **Gate 4: Continuous Donor-Aligned Swap** | Transfer $\ge 12/16$ ($75\%$), $p < 0.01$ | **16 / 16 seeds ($100.0\%$)** | Paired sign-flip $p = 1.5259 \times 10^{-5}$ | **PASS** |
| **Gate 5: Same-History Swap Stability** | Stable behavioral preference $\ge 15/16$ | **16 / 16 seeds ($100.0\%$)** | $\le 1/16$ threshold | **PASS** |
| **Gate 6: First-Order Competence** | 1-hop sensor accuracy $\ge 90\%$ in $\ge 15/16$ | **16 / 16 seeds ($100.0\%$)** | Baseline retention floor | **PASS** |
| **Gate 7: Temporal Shuffle Superiority** | $n_{10} - n_{01} \ge 3, p < 0.05$ | **$\Delta = +16, p = 3.0518 \times 10^{-5}$** | Exact McNemar paired | **PASS** |
| **Gate 8: Structural Zero-Sidecar** | $\equiv 0$ sidecar accesses | **16 / 16 verified ($100.0\%$)** | Direct API invariant | **PASS** |

## 3. Epistemic Invariants & Scope Ceilings
- **Claim**: Development-specific causal history can be stored endogenously in persistent recurrent activation state $z_t$ ($d=128$) and exert causal control over previously validated composition-dependent behavior without an external causal-history store.
- **State Surgery Evidence**: In cloned twin organisms with identical weights $\theta$, identical $z_0$, and identical test-time cues, transplanting $z_t$ ($z_{H1} \leftrightarrow z_{H2}$) causally transfers the history-dependent directional choice ($A \leadsto C$ vs $C \leadsto A$) with zero damage to unrelated first-order sensor competence.
- **Exclusions**: Does NOT claim an abstract causal self-model or symbolic reasoning engine. Does NOT claim arbitrary $N$-hop graph reasoning ($N \ge 3$ reserved for Q17D).



### PROMOTION-CONTRACT-E-Q17D.md
---
promotion_id: PROMOTION-CONTRACT-E-Q17D
contract_id: CONTRACT-E-Q17D
status: PROMOTED
candidate_sha: fa7ebb857187429188e404b9015c7e8a9394602f
generated_at: "2026-08-22 02:50:00Z"
repair_rounds: 1
reviewed_by: chatgpt-pro
authorized_by: human
---

# Verified Promotion Record: PROMOTION-CONTRACT-E-Q17D (Out-of-Distribution Multi-Hop Depth Generalization)

**Lifecycle Status**: `PROMOTED` (Scientific Promotion Review APPROVED by ChatGPT Pro; Strategic Promotion Authorized by Human Research Director)

## 1. Execution & Audit Summary
- **Target Contract**: `CONTRACT-E-Q17D`
- **Phase / Milestone**: Gate E Frontier: Zero-Shot Multi-Hop Depth Generalization (3-Hop to 5-Hop)
- **Candidate Branch**: `mb/CONTRACT-E-Q17D`
- **Scientific Candidate Commit SHA**: `fa7ebb857187429188e404b9015c7e8a9394602f`
- **Execution Base SHA**: `f949eb42c52dc980cb59802e07f8b015b4b93df7`
- **Contract Acceptance Verifier**: `PASS` (`verify_contract_q17d` executed directly from raw 16-seed telemetry with full 8-tensor SHA-256 parameter hashes, exact 120-epoch training verification, exact 2-hop baseline retention, 20-trial 1-hop sensor classifications, coordinate controls $C_3, C_4, C_5$, and depth evaluations)
- **Evidence Package**: Committed and verified in tree:
  - `crates/continuity_garden_core/data/q17d_depth_results.json` (Full 16-seed raw event telemetry across all depths and controls)
- **Classification Outcome**: `NON_MONOTONIC_ANOMALOUS_DEPTH_PROFILE` (Dissociation between multi-hop score extrapolation and causally grounded composition)
- **Repair Iterations**: 1 round (repaired training epoch discrepancy to match exact promoted Q17C 120-epoch training baseline).

---

## 2. Frozen Statistical Gates & Empirical Outcomes

### Section A: Global Experiment-Validity Gates (Mandatory Baseline)

| Gate / Estimand | Preregistered Condition / Floor | Observed Result | Statistical Test | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Gate V1: Promoted Architecture & Weight Fingerprint** | $d=128, d_x=4, d_q=2$; 8-tensor hash $\text{theta\_hash}_i$ identical; epochs $=120$, lr $=0.030$, batches $=64$ | **16 / 16 seeds verified ($100.0\%$)** | SHA-256 byte digest | **PASS** |
| **Gate V2: Canonical 2-Hop Retention ($k=2$)** | Directional margin $m_2 = \text{score}(A \to C) - \text{score}(C \to A) > 0$ | **16 / 16 seeds ($100.0\%$)** | Exact binomial ($\ge 15/16$) | **PASS** |
| **Gate V3: Contemporaneous Sensor Competence** | 20-trial 1-hop sensor classification accuracy $\ge 90.0\%$ | **16 / 16 seeds ($100.0\%$)** | 20 trials vs gold truth | **PASS** |
| **Gate V4: Structural Zero-Sidecar Invariant** | External transition store reads $\equiv 0$ | **16 / 16 verified ($100.0\%$)** | Direct API invariant | **PASS** |

---

### Section B: Depth-Specific Coordinate-OOD Controls

| Control / Condition | Stream / Query | Preregistered Floor | Observed Result | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Control $C_3$ ($D$ Coordinate Extrapolation)** | $A \to B \to D \implies \text{Query}(A, D)$ | $\ge 14 / 16$ seeds ($87.5\%$) | **16 / 16 seeds ($100.0\%$)** | **VALID** |
| **Control $C_4$ ($E$ Coordinate Extrapolation)** | $A \to B \to E \implies \text{Query}(A, E)$ | $\ge 14 / 16$ seeds ($87.5\%$) | **16 / 16 seeds ($100.0\%$)** | **VALID** |
| **Control $C_5$ ($F$ Coordinate Extrapolation)** | $A \to B \to F \implies \text{Query}(A, F)$ | $\ge 14 / 16$ seeds ($87.5\%$) | **16 / 16 seeds ($100.0\%$)** | **VALID** |

*Interpretation Finding*: All coordinate controls pass with $100\%$ accuracy, proving that the model readily extrapolates to unseen role coordinates $(D, E, F)$ in 2-hop sequences. Therefore, multi-hop failures are **strictly causal/compositional depth limitations**, not representation out-of-distribution artifacts.

---

### Section C: Multi-Hop Depth Generalization & Mechanistic Outcomes

| Depth Level | Empirical Observation | Mechanistic Breakdown | Preregistered Tier Criteria | Tier Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **Depth $k=3$ ($A \to B \to C \to D$)** | $m_3 > 0.0$ in $6/16$ seeds ($37.5\%$), paired sign-flip $p = 1.000$ | - State Surgery Choice Flips: **$0/16$**<br>- Transposition Reversals: **$6/16$**<br>- Deranged Shuffle Superiority $[e_2, e_3, e_1]$: **$12/16$** | $p < 0.01$, Surgery $\ge 12/16$, Trans $\ge 15/16$, Shuf $\ge 12/16$ | **Tier 1 NOT Achieved** |
| **Depth $k=4$ ($A \to B \to C \to D \to E$)** | $m_4 > 0.0$ in $12/16$ seeds ($75.0\%$), $p = 6.882 \times 10^{-3}$ | - Transposition Reversals: **$13/16$** | Tier 1 Satisfied + Trans $\ge 14/16$ | **Tier 2 NOT Achieved** |
| **Depth $k=5$ ($A \to B \to C \to D \to E \to F$)** | $m_5 > 0.0$ in $5/16$ seeds ($31.2\%$) | - Mean Margin: $-3.8539$, Median: $-3.4187$ | Continuous empirical reporting | **Tier 3 Descriptive** |

---

## 3. Epistemic Interpretation & Scope Ceilings
- **Core Promoted Claim**: Under the frozen Q17C recurrent architecture (120 training epochs), endpoint-directional scores can extrapolate beyond two-step training—including at four steps ($12/16, p = 6.882 \times 10^{-3}$)—but the longer-horizon responses fail preregistered causal state-surgery ($0/16$), reversal ($6/16$ at $k=3$), and temporal-order controls ($12/16$). Therefore Q17D does not establish recursive compositional depth scaling; it reveals a dissociation between score-level extrapolation and causally grounded developmental-history composition.
- **Immediate Diagnostic Follow-up**: Initiating Diagnostic Scout `Q17D-B` to probe zero-history / query-only baselines, query-readout vs recurrent state contributions, and Jacobian attenuation across time.




---

## 4. Reorientation Prompt (Independent Synthesis)
> **Instructions for Review Desk**:
> 1. Reconstruct: What do we actually know from the evidence above?
> 2. Reinterpret: What do recent results/falsifications imply mechanistically?
> 3. Reconnect: What direct, neighboring, or analogical literature solves this shape?
> 4. Rebranch: What plausible explanations remain?
> 5. Compress: What single experiment eliminates the most roadmap?