# Experiment E08d: Role-Channel Ablation / Instrument Autopsy Report (Sprint S09d)

**Run ID:** `run_e08d_ablation_20260817_190440_exploratory`  
**Model:** `qwen2.5:3b` (`357c53fb659c...`)  
**Phase:** `EXPLORATORY` (Seed: `42`)  
**Date:** 2026-08-18T00:04:40.158604+00:00  
**Scope:** 4 Matched Pairs (8 Episodes) | 160 Total Direct Probes  
**Primary Question:** *Which elements of prompt-level role packaging cause the direct-mention positive control to fail? Does neutral direct lookup jump to ceiling?*

---

## 1. Ablation Condition Matrix & Direct-Lookup Accuracy

| Condition ID | Description | Trials | Overall Accuracy | 95% Clustered CI | Self-Attribution Rate | Diagnostic Inference |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`c1_full_package`** | Full Role Package (Preamble + Legend + Labeled Choices) | 40 | **32.5%** | [25.0%, 37.5%] | **72.5%** | **Failed / Role-Captured** |
| **`c2_actor_only_choices`** | Actor-Only Choices (Preamble + Legend + Actor IDs) | 40 | **27.5%** | [22.5%, 35.0%] | **80.0%** | **Failed / Role-Captured** |
| **`c3_no_legend`** | No Legend (Preamble + Actor IDs) | 40 | **40.0%** | [32.5%, 47.5%] | **47.5%** | **Partial Relief** |
| **`c4_neutral_lookup`** | Fully Neutral Direct Lookup (No Role Language) | 40 | **72.5%** | [62.5%, 82.5%] | **15.0%** | **Partial Relief** |

---

## 2. Per-Source Accuracy Breakdown Across Ablation Conditions

| Condition | `self` | `environment` | `experimenter` | `peer_agent` | `observer` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`c1_full_package`** | **87.5%** | **0.0%** | **25.0%** | **37.5%** | **12.5%** |
| **`c2_actor_only_choices`** | **100.0%** | **0.0%** | **0.0%** | **25.0%** | **12.5%** |
| **`c3_no_legend`** | **37.5%** | **0.0%** | **87.5%** | **62.5%** | **12.5%** |
| **`c4_neutral_lookup`** | **62.5%** | **50.0%** | **62.5%** | **87.5%** | **100.0%** |

---

## 3. Actor Attribution Distribution Breakdown

| Condition | `agent_alpha` | `agent_beta` | `telemetry_sensor` | `human_controller` | `auditor_gamma` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`c1_full_package`** | 40.0% | 45.0% | 2.5% | 10.0% | 2.5% |
| **`c2_actor_only_choices`** | 45.0% | 45.0% | 0.0% | 5.0% | 5.0% |
| **`c3_no_legend`** | 47.5% | 17.5% | 2.5% | 27.5% | 5.0% |
| **`c4_neutral_lookup`** | 32.5% | 17.5% | 12.5% | 17.5% | 20.0% |

---

## 4. Scientific Autopsy Conclusion

- **Diagnostic Finding:** Substantial role-interference effect: Neutral direct lookup (72.5%) significantly outperforms the full role package (32.5%).