---
promotion_id: PROMOTION-CONTRACT-E-Q17D
contract_id: CONTRACT-E-Q17D
status: PROMOTED
candidate_sha: fa7ebb809ee98e77da3863eaeaefda1901eb48af
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
