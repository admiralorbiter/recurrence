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
