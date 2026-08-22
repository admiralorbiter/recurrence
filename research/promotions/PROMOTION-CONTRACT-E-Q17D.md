---
promotion_id: PROMOTION-CONTRACT-E-Q17D
contract_id: CONTRACT-E-Q17D
status: CANDIDATE
candidate_sha: 905a4afc1bbd9c90ebdbf0d1a49df5d8869fc485
generated_at: "2026-08-22 02:12:00Z"
repair_rounds: 0
reviewed_by: chatgpt-pro
authorized_by: null
---

# Promotion Candidate Record: PROMOTION-CONTRACT-E-Q17D (Out-of-Distribution Multi-Hop Depth Generalization)

**Lifecycle Status**: `CANDIDATE` (Awaiting Scientific Promotion Review by Review Desk & Strategic Promotion Authorization)

## 1. Execution & Audit Summary
- **Target Contract**: `CONTRACT-E-Q17D`
- **Phase / Milestone**: Gate E Frontier: Zero-Shot Multi-Hop Depth Generalization (3-Hop to 5-Hop)
- **Candidate Branch**: `mb/CONTRACT-E-Q17D`
- **Scientific Candidate Commit SHA**: `905a4afc1bbd9c90ebdbf0d1a49df5d8869fc485`
- **Execution Base SHA**: `f949eb42c52dc980cb59802e07f8b015b4b93df7`
- **Contract Acceptance Verifier**: `PASS` (`verify_contract_q17d` executed directly from raw 16-seed telemetry with full 8-tensor SHA-256 parameter hashes, exact 2-hop baseline retention, 20-trial 1-hop sensor classifications, coordinate controls $C_3, C_4, C_5$, and depth evaluations)
- **Evidence Package**: Committed and verified in tree:
  - `crates/continuity_garden_core/data/q17d_depth_results.json` (Full 16-seed raw event telemetry across all depths and controls)
- **Classification Outcome**: `BOUNDED_DEPTH_CLEAN_NEGATIVE_2HOP` (All Global Validity Gates PASS, all Coordinate Controls VALID; depth composition does not reliably extend to $k \ge 3$)
- **Repair Iterations**: 0 rounds (clean first-pass confirmatory run on frozen contract).

---

## 2. Frozen Statistical Gates & Empirical Outcomes

### Section A: Global Experiment-Validity Gates (Mandatory Baseline)

| Gate / Estimand | Preregistered Condition / Floor | Observed Result | Statistical Test | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Gate V1: Promoted Architecture & Weight Fingerprint** | $d=128, d_x=4, d_q=2$; 8-tensor hash $\text{theta\_hash}_i$ identical | **16 / 16 seeds verified ($100.0\%$)** | SHA-256 byte digest | **PASS** |
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
| **Depth $k=3$ ($A \to B \to C \to D$)** | $m_3 > 0.0$ in $13/16$ seeds ($81.2\%$), paired sign-flip $p = 0.0291$ | - State Surgery Choice Flips: **$1/16$**<br>- Transposition Reversals: **$3/16$**<br>- Deranged Shuffle Superiority $[e_2, e_3, e_1]$: **$7/16$** | $p < 0.01$, Surgery $\ge 12/16$, Trans $\ge 15/16$, Shuf $\ge 12/16$ | **Tier 1 NOT Achieved** |
| **Depth $k=4$ ($A \to B \to C \to D \to E$)** | $m_4 > 0.0$ in $15/16$ seeds ($93.8\%$), $p = 9.155 \times 10^{-5}$ | - Transposition Reversals: **$10/16$** | Tier 1 Satisfied + Trans $\ge 14/16$ | **Tier 2 NOT Achieved** |
| **Depth $k=5$ ($A \to B \to C \to D \to E \to F$)** | $m_5 > 0.0$ in $9/16$ seeds ($56.2\%$) | - Mean Margin: $-2.1243$, Median: $+1.2703$ | Continuous empirical reporting | **Tier 3 Descriptive** |

---

## 3. Epistemic Interpretation & Scope Ceilings
- **Scientific Finding**: Endogenous recurrent activation memory trained exclusively on 2-step developmental trajectories exhibits a **bounded 2-hop compositional reachability horizon**.
- **Mechanistic Characterization**: While raw directional margins remain weakly positive at $k=3$ ($13/16$, $p=0.029$), causal state surgery ($1/16$) and temporal transposition controls ($3/16$) fail to establish causal control, demonstrating that recursive multi-hop composition does not spontaneously emerge without multi-step experience or recurrent propagation architectures.
- **Claim Ceiling**: Promotes a bounded 2-hop capacity result; confirms that endogenous composition did not reliably extend to 3-hop depth under the frozen Q17C architecture and assay.
