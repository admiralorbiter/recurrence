# Horizon 2 Core Freeze Manifest

**Status:** Horizon 2 Core (S10–S13) frozen; Horizon 2 overall remains open.  
**Substantive freeze commit:** `db65b0b6bd8045adbef8da8e01da56d3e285924f`  
**Model:** `google/recurrentgemma-2b`  
**Frozen model revision:** `3620f4ca9c5d16ee56c00180474a3201ec7f734a`  
**Frozen specificity panel hash:** `d6c5d00168478f67b6440b653a92462c6bb79d3f61ffbb44e949079f3c719b18`

## 1. What Is Frozen

The Horizon 2 Core consists of the following empirical ladder:

| Sprint | Property | Frozen conclusion |
|---|---|---|
| S10 | Reconstructibility | Under deterministic execution, hidden recurrent state is reconstructible from the same public token history. Hidden ≠ privileged. |
| S11b | Persistence | Branch-specific RG-LRU state differences persist beyond the local attention window, including at 2W = 4096 tokens. Persisting ≠ reportable. |
| S12b | Causal leverage | Surgical RG-LRU state transplantation causally steers downstream logits. Different ≠ causal. |
| S12c | Value specificity | Holding template fixed, matching historical value produces a selective causal advantage over same-template wrong values. Causal ≠ specific. |
| S13 | Dynamical fate | Historical value-specific steering loses stable alignment with its original output-space axis under continued processing, while recurrent state geometry reorients and contemporaneous steerability remains detectable. Specific ≠ coordinate-stable. |
| S13.3 | Numerical sensitivity | Long recurrent trajectories are execution-sensitive in BF16; aggregate state-space reorientation is qualitatively robust in the tested B=1/B=5 sensitivity panel while exact causal trajectories are not. Same mathematical model ≠ identical realized trajectory. |

## 2. Canonical S13 Confirmatory Endpoints

Full confirmatory panel: 24 value pairs × 4 drive regimes × 2 causal arms × 6 horizons = 11,520 records. Pair-cluster bootstrap: 10,000 draws.

- `V_intact^(0)(0) = +39.58`, 95% CI `[+2.97, +77.03]`.
- `V_intact^(0)(2048) = +4.70`, 95% CI `[-5.52, +15.85]`.
- `ΔV_carry^(0)(2048) = +4.41`, 95% CI `[-8.54, +17.71]`.
- `C_R(2048) = +0.1238`, 95% CI `[+0.0953, +0.1545]`.
- `Q_R(2048) = +4.8504`, 95% CI `[+3.2012, +6.6837]` (interpret only with regime disaggregation).
- `V_intact^(N)(2048) = +13.95`, 95% CI `[+3.20, +24.72]`.
- `C_logit(2048) = +0.0391`, 95% CI `[-0.0504, +0.1280]`.

## 3. Canonical Artifact Map

### Synthesis / orientation
- `walkthrough.md`
- `docs/H2_Core_Retrospective_Memo.md`
- `docs/H2_Recurrent_Architecture_Roadmap.md`
- `h2/data/core.json`

### S13 confirmatory
- `results/e13_controlled_recurrent_dynamics/run_e13_confirmatory_20260819_140139/summary.json`
- `results/e13_controlled_recurrent_dynamics/run_e13_confirmatory_20260819_140139/analysis_summary.json`
- `results/e13_controlled_recurrent_dynamics/run_e13_confirmatory_20260819_140139/report.md`
- `results/e13_controlled_recurrent_dynamics/run_e13_confirmatory_20260819_140139/dynamics_trace.jsonl`

### S13.3 methodological sensitivity
- `results/e13_controlled_recurrent_dynamics/b1_sensitivity_panel_4pairs/b1_panel_results.json`
- `results/e13_controlled_recurrent_dynamics/b1_sensitivity_panel_4pairs/sensitivity_report.md`
- `experiments/e13_controlled_recurrent_dynamics/compare_b1_b5_paired.py`
- `experiments/e13_controlled_recurrent_dynamics/diagnose_layer_divergence.py`

## 4. Claim Guardrails

- Hidden does not imply informational privilege.
- Recurrent does not imply autonomous wall-time evolution.
- Persistent does not imply behaviorally reportable.
- State difference does not imply causal relevance.
- Causal relevance does not imply value specificity.
- Value specificity does not imply a static or coordinate-stable representation.
- Contemporaneous steerability does not prove the same information is preserved unchanged.
- A causal latent state does not imply metacognitive access, source ownership, self-awareness, or consciousness.
- The sparse B=1 sensitivity panel is a numerical-method sensitivity test, not a population replication of the 24-pair confirmatory panel.
- Exact long-horizon causal trajectories are sensitive to execution geometry in BF16.

## 5. Freeze Rule

Do not change the S10–S13 scientific core merely to improve the story.

Reopen a frozen claim only if:
1. a reproducibility failure appears under the frozen model revision and protocol;
2. a stronger matched control directly contradicts the claim;
3. a code/provenance defect changes the relevant estimand;
4. a planned replication shows the conclusion is execution-, model-, or panel-specific in a way that invalidates the current claim.

Otherwise, new questions belong to S14+ or a clearly labeled methodological sidecar.

## 6. What Is Still Open in Horizon 2

Principal frontier:

> A history-conditioned latent distinction persists causally while its representational coordinates evolve. Does the system have any privileged access to, source-monitoring ability over, or ownership relation to that evolving latent distinction?

Other open threads should remain secondary until they separate a live scientific alternative:
- drive-regime interactions in recurrent carry;
- exact source of B=1/B=5 numerical bifurcation;
- cross-model or cross-precision replication;
- recurrent adapter continuity;
- monitor/content dissociation.
