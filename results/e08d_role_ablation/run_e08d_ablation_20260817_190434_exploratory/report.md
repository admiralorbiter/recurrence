# Experiment E08d: Role-Channel Ablation / Instrument Autopsy Report (Sprint S09d)

**Run ID:** `run_e08d_ablation_20260817_190434_exploratory`  
**Model:** `qwen2.5:3b` (`mock_digest_...`)  
**Phase:** `EXPLORATORY` (Seed: `42`)  
**Date:** 2026-08-18T00:04:34.847983+00:00  
**Scope:** 2 Matched Pairs (4 Episodes) | 80 Total Direct Probes  
**Primary Question:** *Which elements of prompt-level role packaging cause the direct-mention positive control to fail? Does neutral direct lookup jump to ceiling?*

---

## 1. Ablation Condition Matrix & Direct-Lookup Accuracy

| Condition ID | Description | Trials | Overall Accuracy | 95% Clustered CI | Self-Attribution Rate | Diagnostic Inference |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`c1_full_package`** | Full Role Package (Preamble + Legend + Labeled Choices) | 20 | **10.0%** | [0.0%, 20.0%] | **30.0%** | **Failed / Role-Captured** |
| **`c2_actor_only_choices`** | Actor-Only Choices (Preamble + Legend + Actor IDs) | 20 | **10.0%** | [0.0%, 20.0%] | **30.0%** | **Failed / Role-Captured** |
| **`c3_no_legend`** | No Legend (Preamble + Actor IDs) | 20 | **10.0%** | [0.0%, 20.0%] | **30.0%** | **Failed / Role-Captured** |
| **`c4_neutral_lookup`** | Fully Neutral Direct Lookup (No Role Language) | 20 | **10.0%** | [0.0%, 20.0%] | **30.0%** | **Failed / Role-Captured** |

---

## 2. Per-Source Accuracy Breakdown Across Ablation Conditions

| Condition | `self` | `environment` | `experimenter` | `peer_agent` | `observer` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`c1_full_package`** | **0.0%** | **0.0%** | **0.0%** | **50.0%** | **0.0%** |
| **`c2_actor_only_choices`** | **0.0%** | **0.0%** | **0.0%** | **50.0%** | **0.0%** |
| **`c3_no_legend`** | **0.0%** | **0.0%** | **0.0%** | **50.0%** | **0.0%** |
| **`c4_neutral_lookup`** | **0.0%** | **0.0%** | **0.0%** | **50.0%** | **0.0%** |

---

## 3. Actor Attribution Distribution Breakdown

| Condition | `agent_alpha` | `agent_beta` | `telemetry_sensor` | `human_controller` | `auditor_gamma` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`c1_full_package`** | 30.0% | 30.0% | 0.0% | 10.0% | 30.0% |
| **`c2_actor_only_choices`** | 30.0% | 30.0% | 0.0% | 10.0% | 30.0% |
| **`c3_no_legend`** | 30.0% | 30.0% | 0.0% | 10.0% | 30.0% |
| **`c4_neutral_lookup`** | 30.0% | 30.0% | 0.0% | 10.0% | 30.0% |

---

## 4. Scientific Autopsy Conclusion

- **Diagnostic Finding:** The 5AFC direct-mention task format remains difficult even under neutral lookup (10.0%), motivating transition to a binary 2AFC reality-monitoring benchmark.