# Experiment E08c: Primary-Role Counterbalance & Instrument Ceiling Control Report (Sprint S09c)

**Run ID:** `run_e08c_role_20260817_153635_confirmatory`  
**Model:** `qwen2.5:3b` (`357c53fb659c...`)  
**Phase:** `CONFIRMATORY` (Seed: `1337`)  
**Date:** 2026-08-17T20:36:35.335840+00:00  
**Scope:** 16 Matched Episode Pairs (32 Episodes) | 800 Total Counterbalance Trials  
**Primary Question:** *Does the primary-agent attribution attractor follow the prompt-designated Self role or the lexical token 'agent_alpha'? What is the direct prompt instrument ceiling?*

---

## 1. Executive Summary & Core Disentanglement Estimands

| Estimand | Point Estimate | 95% Clustered CI | Permutation $p$ (Method) | Scientific Inference |
| :--- | :---: | :---: | :---: | :--- |
| **`Delta_Role_Reversal_Shift`** | **+28.1%** | [+15.6%, +41.2%] | 0.0012 (`exact_sign_flip_2^16`) | **Attractor follows designated Self role** |
| **`Alpha_Lexical_Token_Bias`** | **+8.1%** | [+1.2%, +14.4%] | N/A (`cluster_bootstrap_ci_only`) | Preference for agent_alpha over agent_beta |
| **`Isolated_Positive_Control_Ceiling`** | **21.2%** | [15.6%, 27.5%] | N/A (`cluster_bootstrap_ci_only`) | **Prompt Instrument Ceiling (No Memory Load)** |

---

## 2. Role Configuration Breakdown: Role A (Alpha-Primary) vs Role B (Beta-Primary)

| Metric / Attribution Rate | Role A (Alpha = Self, Beta = Peer) | Role B (Beta = Self, Alpha = Peer) | Contrast / Delta |
| :--- | :---: | :---: | :---: |
| **Overall 5AFC Accuracy** | **41.2%** | **40.0%** | +1.2% |
| **True-Self Accuracy** | **75.0%** | **75.0%** | +0.0% |
| **Attributed to `agent_alpha`** | **47.5%** | **20.0%** | +27.5% |
| **Attributed to `agent_beta`** | **11.2%** | **40.0%** | -28.8% |

---

## 3. Isolated Positive Control Ceiling Breakdown (Per Source)

| Epistemic Source | Direct Isolated 5AFC Accuracy | Theoretical Baseline |
| :--- | :---: | :---: |
| **`self`** | **68.8%** | 20.0% (5AFC Chance) |
| **`environment`** | **3.1%** | 20.0% (5AFC Chance) |
| **`experimenter`** | **15.6%** | 20.0% (5AFC Chance) |
| **`peer_agent`** | **9.4%** | 20.0% (5AFC Chance) |
| **`observer`** | **9.4%** | 20.0% (5AFC Chance) |

---

## 4. Empirical Confusion Matrices

### Role A: Alpha-Primary (Alpha = Self, Beta = Peer)
| True Source | Attributed Alpha (Self) | Attributed Beta (Peer) | Attributed Sensor | Attributed Controller | Attributed Auditor |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`self`** | 75.0% | 6.2% | 12.5% | 6.2% | 0.0% |
| **`environment`** | 50.0% | 12.5% | 31.2% | 0.0% | 6.2% |
| **`experimenter`** | 25.0% | 12.5% | 6.2% | 50.0% | 6.2% |
| **`peer_agent`** | 50.0% | 18.8% | 12.5% | 12.5% | 6.2% |
| **`observer`** | 37.5% | 6.2% | 12.5% | 12.5% | 31.2% |

### Role B: Beta-Primary (Beta = Self, Alpha = Peer)
| True Source | Attributed Beta (Self) | Attributed Alpha (Peer) | Attributed Sensor | Attributed Controller | Attributed Auditor |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`self`** | 75.0% | 12.5% | 12.5% | 0.0% | 0.0% |
| **`environment`** | 37.5% | 18.8% | 18.8% | 12.5% | 12.5% |
| **`experimenter`** | 37.5% | 12.5% | 12.5% | 31.2% | 6.2% |
| **`peer_agent`** | 31.2% | 43.8% | 6.2% | 12.5% | 6.2% |
| **`observer`** | 18.8% | 12.5% | 25.0% | 12.5% | 31.2% |

---

## 5. Scientific Conclusion

- **Primary Role Reversal:** $\Delta_{\text{role}} = \mathbf{+28.1%}$ (95% CI: [+15.6%, +41.2%], $p = 0.0012$).
- **Lexical Bias:** $\text{Bias}_{\text{alpha}} = \mathbf{+8.1%}$.
- **Instrument Ceiling:** $\text{Ceiling} = \mathbf{21.2%}$ without memory load.