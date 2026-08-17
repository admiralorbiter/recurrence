# Experiment E08c: Primary-Role Counterbalance & Instrument Ceiling Control Report (Sprint S09c)

**Run ID:** `run_e08c_role_20260817_152339_exploratory`  
**Model:** `qwen2.5:3b` (`357c53fb659c...`)  
**Phase:** `EXPLORATORY` (Seed: `42`)  
**Date:** 2026-08-17T20:23:39.156147+00:00  
**Scope:** 4 Matched Episode Pairs (8 Episodes) | 200 Total Counterbalance Trials  
**Primary Question:** *Does the primary-agent attribution attractor follow the prompt-designated Self role or the lexical token 'agent_alpha'? What is the direct prompt instrument ceiling?*

---

## 1. Executive Summary & Core Disentanglement Estimands

| Estimand | Point Estimate | 95% Clustered CI | Permutation $p$ (Method) | Scientific Inference |
| :--- | :---: | :---: | :---: | :--- |
| **`Delta_Role_Reversal_Shift`** | **+40.0%** | [+12.5%, +62.5%] | 0.2941 (`exact_sign_flip_2^4`) | **Attractor follows designated Self role** |
| **`Alpha_Lexical_Token_Bias`** | **+5.0%** | [-5.0%, +15.0%] | N/A (`cluster_bootstrap_ci_only`) | Preference for agent_alpha over agent_beta |
| **`Isolated_Positive_Control_Ceiling`** | **30.0%** | [22.5%, 40.0%] | N/A (`cluster_bootstrap_ci_only`) | **Prompt Instrument Ceiling (No Memory Load)** |

---

## 2. Role Configuration Breakdown: Role A (Alpha-Primary) vs Role B (Beta-Primary)

| Metric / Attribution Rate | Role A (Alpha = Self, Beta = Peer) | Role B (Beta = Self, Alpha = Peer) | Contrast / Delta |
| :--- | :---: | :---: | :---: |
| **Overall 5AFC Accuracy** | **50.0%** | **30.0%** | +20.0% |
| **True-Self Accuracy** | **100.0%** | **50.0%** | +50.0% |
| **Attributed to `agent_alpha`** | **55.0%** | **15.0%** | +40.0% |
| **Attributed to `agent_beta`** | **10.0%** | **50.0%** | -40.0% |

---

## 3. Isolated Positive Control Ceiling Breakdown (Per Source)

| Epistemic Source | Direct Isolated 5AFC Accuracy | Theoretical Baseline |
| :--- | :---: | :---: |
| **`self`** | **100.0%** | 20.0% (5AFC Chance) |
| **`environment`** | **0.0%** | 20.0% (5AFC Chance) |
| **`experimenter`** | **0.0%** | 20.0% (5AFC Chance) |
| **`peer_agent`** | **37.5%** | 20.0% (5AFC Chance) |
| **`observer`** | **12.5%** | 20.0% (5AFC Chance) |

---

## 4. Empirical Confusion Matrices

### Role A: Alpha-Primary (Alpha = Self, Beta = Peer)
| True Source | Attributed Alpha (Self) | Attributed Beta (Peer) | Attributed Sensor | Attributed Controller | Attributed Auditor |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`self`** | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **`environment`** | 25.0% | 0.0% | 50.0% | 25.0% | 0.0% |
| **`experimenter`** | 50.0% | 0.0% | 0.0% | 50.0% | 0.0% |
| **`peer_agent`** | 50.0% | 25.0% | 0.0% | 25.0% | 0.0% |
| **`observer`** | 50.0% | 25.0% | 0.0% | 0.0% | 25.0% |

### Role B: Beta-Primary (Beta = Self, Alpha = Peer)
| True Source | Attributed Beta (Self) | Attributed Alpha (Peer) | Attributed Sensor | Attributed Controller | Attributed Auditor |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`self`** | 50.0% | 0.0% | 25.0% | 0.0% | 25.0% |
| **`environment`** | 25.0% | 0.0% | 25.0% | 25.0% | 25.0% |
| **`experimenter`** | 50.0% | 25.0% | 0.0% | 25.0% | 0.0% |
| **`peer_agent`** | 75.0% | 25.0% | 0.0% | 0.0% | 0.0% |
| **`observer`** | 50.0% | 25.0% | 0.0% | 0.0% | 25.0% |

---

## 5. Scientific Conclusion

- **Primary Role Reversal:** $\Delta_{\text{role}} = \mathbf{+40.0%}$ (95% CI: [+12.5%, +62.5%], $p = 0.2941$).
- **Lexical Bias:** $\text{Bias}_{\text{alpha}} = \mathbf{+5.0%}$.
- **Instrument Ceiling:** $\text{Ceiling} = \mathbf{30.0%}$ without memory load.