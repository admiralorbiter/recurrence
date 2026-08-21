---
contract_id: CONTRACT-E-Q17B
status: DRAFT
title: "Gate E Frontier: Self-Supervised Endogenous Composition Discovery (Q17B)"
phase: E
parent_contract: CONTRACT-E-Q17A-R1
base_sha: efc2d9941bb546a28fc01ff634211e79070a5bae
proposed_by: antigravity
design_review: CHANGES_REQUESTED
reviewed_by: chatgpt-pro
authorized_by: null
resource:
  resource_class: CPU
  long_running: false
  exclusive_gpu: false
  interruptible: true
created_at: "2026-08-21 22:17:00Z"
---

# Research Contract Proposal: CONTRACT-E-Q17B (Draft)

## 1. Epistemic Frontier & Research Question
- **Context**: In Q17A, we established that a learned neural composition kernel $f_\theta(e_{AB}, e_{BC})$ can compute multi-hop reachability and preserve causal discrimination. However, $f_\theta$ was trained using explicit auxiliary two-hop scalar supervision targets.
- **Core Question (Q17B)**: Can an endogenous composition mechanism learn to compose multi-hop causal representations $A \to B \to C$ under purely self-supervised multi-step trajectory consistency or contrastive local transition objectives, **without explicit two-hop reachability labels**?

## 2. Hypothesis & Mechanistic Model
- **Hypothesis**: Self-supervised predictive coding across consecutive transitions ($e_{t} \odot e_{t+1} \approx e_{t \to t+2}$) induces a composition operator that achieves non-trivial zero-shot multi-hop choice accuracy on withheld endpoints.
- **Null Hypothesis ($H_0$)**: Self-supervised local objectives fail to structure the latent representation for transitive reachability, resulting in chance performance ($\le 50\%$) on multi-hop conflict tasks.

## 3. Pre-registered Success Criteria & Falsification Floors
- **Gate 1 (Zero-Shot Multi-Hop Conflict Accuracy)**: $\ge 10/16$ seeds ($62.5\%$).
- **Gate 2 (Laundering Discrimination Accuracy)**: $\ge 10/16$ seeds ($62.5\%$).
- **Gate 3 (Directional Transposition Falsification)**: Transposed evaluation accuracy $\le 3/16$ seeds.
- **Gate 4 (Mechanistic Path-Break Lesion)**: Permutation test $p < 0.05$.

## 4. Epistemic Boundaries & Non-Claims
- This contract does NOT claim arbitrary $N$-hop path planning or self-supervised memory consolidation across lifetimes (deferred to Q17C).
