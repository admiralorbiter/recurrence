---
contract_id: CONTRACT-E-Q17B
status: FROZEN
title: "Gate E Frontier: Self-Supervised Endogenous Composition Discovery (Q17B)"
phase: E
parent_contract: CONTRACT-E-Q17A-R1
base_sha: efc2d9941bb546a28fc01ff634211e79070a5bae
proposed_by: antigravity
design_review: APPROVED
reviewed_by: chatgpt-pro
authorized_by: human
resource_class: cpu
long_running: false
exclusive_gpu: false
interruptible: true
created_at: "2026-08-21 22:40:00Z"
---

# Research Contract: CONTRACT-E-Q17B (Frozen)

## 1. Epistemic Frontier & Scientific Context
- **Baseline (Q17A Promoted)**: In `CONTRACT-E-Q17A-R1`, we established that a neural composition kernel $f_\theta(e_{AB}, e_{BC}) \to a_{AC}$ generalizes to withheld multi-hop causal endpoints and preserves mechanistic directionality and path-break signatures when trained with explicit two-hop auxiliary reachability targets.
- **Core Question (Q17B)**: Can an identical neural composition kernel learn to compose multi-hop causal representations when **all explicit two-hop reachability supervision is removed**, learning exclusively from naturally observed multi-step trajectory prediction/consistency?

## 2. Experimental Model & Scalar Self-Supervised Formulation
- **Fixed Architecture**: The $f_\theta$ architecture remains strictly identical to Q17A: a 2-layer MLP with scalar logit/score output $f_\theta(e_{AB}, e_{BC}) \in \mathbb{R}$ mapped via sigmoid $\sigma(\cdot)$ to predicted transition strength $\hat{a} \in [0, 1]$.
- **Prohibited Learning Signals**:
  - Strictly NO explicit two-hop reachability labels.
  - Strictly NO transitive edge annotations or transitive closure ground truth.
  - Strictly NO path identifiers, graph search algorithms, or $(A, C)$ annotations during training.
- **Type-Compatible Scalar Self-Supervised Target**:
  - The training target $y_{\text{self-sup}} \in \{0.0, 1.0\}$ is an observable binary scalar indicator derived exclusively from raw episodic trajectories without intermediate reachability labels:
    $$y_{\text{self-sup}} = \mathbb{I}\left( s_{t+2} = s_{\text{target}} \right)$$
    where $s_{t+2}$ is the actual observable environment state realized two steps forward after observing transition $e_t = (s_t, a_t)$ followed by $e_{t+1} = (s_{t+1}, a_{t+1})$.
  - Loss function:
    $$\mathcal{L}_{\text{self-sup}} = \text{BCE}\left(\sigma(f_\theta(e_t, e_{t+1})), y_{\text{self-sup}}\right)$$
- **Critical Negative Control (Temporally Shuffled Pairs)**:
  - An identical kernel $f_\theta^{\text{shuffled}}$ is trained where the temporal pairing between $e_t$ and $e_{t+1}$ is destroyed by randomly shuffling transitions across independent episodes (preserving marginal single-step transition distributions while eliminating multi-step temporal consistency).
  - Q17B must statistically outperform $f_\theta^{\text{shuffled}}$ on zero-shot multi-hop conflict tasks.

## 3. Pre-registered Success Gates & Falsification Criteria

| Gate / Estimand | Target Condition | Pre-registered Floor | Rationale |
| :--- | :--- | :--- | :--- |
| **Gate 1 (Zero-Shot Multi-Hop Conflict)** | Self-Supervised $f_\theta$ | $\ge 10/16$ seeds ($62.5\%$) | Non-trivial multi-hop decision capability |
| **Gate 2 (Laundering Discrimination)** | Self-Supervised $f_\theta$ | $\ge 10/16$ seeds ($62.5\%$) | Discard ungrounded circular transitions |
| **Gate 3 (Temporal Shuffle Control Superiority)** | $f_\theta$ vs $f_\theta^{\text{shuffled}}$ | $n_{10} - n_{01} \ge 3$ ($p < 0.05$) | Proves temporal trajectory signal causes composition |
| **Gate 4 (Directional Transposition Falsification)** | Transposed Matrix $\hat{A}^T$ ($A \neq C$) | $\le 2/16$ seeds, return $< 0.00$ | Falsifies symmetric or non-directional shortcuts |
| **Gate 5 (Transposition Laundering Invariant)** | Transposed Matrix $\hat{A}^T$ ($A = C$) | $\ge 10/16$ seeds | Preserves self-loop discrimination |
| **Gate 6 (Mechanistic Path-Break Specificity)** | Developmental Lesion $A \to B$ | Permutation $p < 0.01$ | Confirms causal dependence on intermediate representation |
| **Reference Baseline (Q17A Supervised)** | Supervised Upper Reference | Informational (not a hard floor) | Measures gap between self-supervised and supervised induction |

## 4. Epistemic Boundaries & Strict Claim Ceilings
- **Authorized Claim**: Self-supervised multi-step trajectory prediction induces a composition-capable neural operator on local transition representations without explicit two-hop reachability labels.
- **Explicit Exclusions**: Does NOT claim that the architecture autonomously discovered that composition exists (the kernel is architecturally handed adjacent transition pairs). Does NOT claim arbitrary $N$-hop path planning or lifetime memory consolidation (deferred to Q17C).
