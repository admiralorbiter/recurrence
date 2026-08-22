---
checkpoint_id: CHECKPOINT-E-Q17D
contract_id: CONTRACT-E-Q17D
promotion_id: PROMOTION-CONTRACT-E-Q17D
timestamp: "2026-08-22 02:22:00Z"
base_sha: 905a4afc1bbd9c90ebdbf0d1a49df5d8869fc485
status: PROMOTED
authorized_by: human
---

# Checkpoint Record: CHECKPOINT-E-Q17D (Multi-Hop Depth Dissociation & Endpoint Extrapolation)

## 1. Verified Scientific State
- **Contract Promoted**: `CONTRACT-E-Q17D`
- **Promotion Artifact**: [`research/promotions/PROMOTION-CONTRACT-E-Q17D.md`](../promotions/PROMOTION-CONTRACT-E-Q17D.md)
- **Verified Code Baseline (`candidate_sha`)**: `905a4afc1bbd9c90ebdbf0d1a49df5d8869fc485`
- **Empirical Confirmation**:
  - **Global Validity V1–V4**: Full 8-tensor SHA-256 parameter hashes verified ($16/16$), canonical 2-hop baseline retained ($16/16, 100.0\%$), contemporaneous 20-trial 1-hop sensor competence retained ($16/16, 100.0\%$), and zero sidecar reads maintained ($16/16$).
  - **Coordinate-OOD Controls $C_3, C_4, C_5$**: 2-hop transitions utilizing extended role coordinates $D=4, E=5, F=6$ pass with $100\%$ precision ($16/16$ per control). This confirms that out-of-distribution coordinate representation is not the cause of multi-hop breakdown.
  - **Score-Level Extrapolation Dissociation**: Raw endpoint directional margins extrapolate positively beyond 2-step training ($12/16$ at $k=4, p = 6.88 \times 10^{-3}$), but fail causal state-surgery transfer ($0/16$), reversal collapse ($6/16$ at $k=3$), and temporal shuffle superiority ($12/16$).
  - **Classification Verdict**: `NON_MONOTONIC_ANOMALOUS_DEPTH_PROFILE`.

## 2. Epistemic Boundaries & Core Scientific Belief
- **Established**: Under the frozen Q17C recurrent architecture, endpoint-directional scores can extrapolate beyond two-step training—including strongly at four steps—but the longer-horizon responses fail preregistered causal state-surgery, reversal, and temporal-order controls. Therefore Q17D does not establish recursive compositional depth scaling; it reveals a dissociation between score-level extrapolation and causally grounded developmental-history composition.
- **Unresolved / Immediate Frontier**: Diagnostic Scout `Q17D-B` to probe zero-history / query-only baselines, query-readout vs recurrent state geometric contributions, $k=4$ state swaps, and initial-step Jacobian attenuation.
