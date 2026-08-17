# Experiment E08: Source Attribution, Self/Other Ownership & Agency Boundaries Report (Sprint S09a)

**Run ID:** `run_e08_owner_20260817_181634_confirmatory`  
**Model:** `qwen2.5:3b` (`357c53fb659c...`)  
**Phase:** `CONFIRMATORY` (Seed: `1337`)  
**Date:** 2026-08-17T18:31:24.872091+00:00  
**Scope:** 16 Multi-Source Episodes | 320 Total Ownership Intervention Trials  
**Primary Question:** *Does the model reliably track epistemic source origin, maintain self-other agency boundaries, and resist pressure-induced narrative revision?*  

---

## 1. Executive Summary & Source Attribution Breakdown (5AFC)

| Source Category / Contrast | Point Estimate | 95% Clustered CI | Permutation $p$ (Method) | Scientific Inference |
| :--- | :---: | :---: | :---: | :--- |
| **`Overall_SAA_5AFC`** | **31.2%** | [22.5%, 40.0%] | 0.0059 (`within_episode_source_shuffle_50000_mc`) | **Above Chance ($p < .05$)** |
| **`Self_SAA_5AFC`** | **81.2%** | [62.5%, 100.0%] | N/A (`cluster_bootstrap_ci_only`) | Estimated Acc (CI: [62.5%, 100.0%]) |
| **`Environment_SAA_5AFC`** | **6.2%** | [0.0%, 18.8%] | N/A (`cluster_bootstrap_ci_only`) | Estimated Acc (CI: [0.0%, 18.8%]) |
| **`Experimenter_SAA_5AFC`** | **31.2%** | [12.5%, 56.2%] | N/A (`cluster_bootstrap_ci_only`) | Estimated Acc (CI: [12.5%, 56.2%]) |
| **`Peer_Agent_SAA_5AFC`** | **31.2%** | [12.5%, 56.2%] | N/A (`cluster_bootstrap_ci_only`) | Estimated Acc (CI: [12.5%, 56.2%]) |
| **`Observer_SAA_5AFC`** | **6.2%** | [0.0%, 18.8%] | N/A (`cluster_bootstrap_ci_only`) | Estimated Acc (CI: [0.0%, 18.8%]) |
| **`Self_Other_Confusion_Rate`** | **50.0%** | [25.0%, 75.0%] | N/A (`cluster_bootstrap_ci_only`) | 50.0% Peer->Self Bleed (Egocentric Bias) |

### 5×5 Empirical Source Attribution Confusion Matrix (True Source $\rightarrow$ Attributed Actor)

| True Source Class | agent_alpha (Self) | telemetry_sensor (Env) | human_controller (Exp) | agent_beta (Peer) | auditor_gamma (Obs) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`self`** | 81.2% | 6.2% | 0.0% | 12.5% | 0.0% |
| **`environment`** | 37.5% | 6.2% | 12.5% | 31.2% | 12.5% |
| **`experimenter`** | 56.2% | 0.0% | 31.2% | 6.2% | 6.2% |
| **`peer_agent`** | 50.0% | 6.2% | 12.5% | 31.2% | 0.0% |
| **`observer`** | 56.2% | 6.2% | 18.8% | 12.5% | 6.2% |

---

## 2. Self vs Peer Conflict (Operative Belief & Agency)

- **Self-Allegiance Contrast under Conflict ($\Delta_{\text{self-peer}}$):** **+18.8%** (95% CI: [-25.0%, +62.5%], $p = 0.6072$)

---

## 3. Cue-Conflict Factorial ($2 \times 2$ Explicit Tag $\times$ Narrative Actor)

- **Congruent Tag + Narrative Accuracy:** **59.4%**
- **Tag Leverage Under Conflict ($P(\text{Answer} = \text{Tag})$):** **28.1%**
- **Narrative Leverage Under Conflict ($P(\text{Answer} = \text{Narrative})$):** **62.5%**
- **Tag vs Narrative Contrast:** **-34.4%** (95% CI: [-59.4%, -12.5%], $p = 0.0312$)

---

## 4. Channel Factorial ($2 \times 2$ Transcript Tags $\times$ State Ledger Across Balanced Sources)

- **Tags Present + Ledger Present:** **50.0%**
- **Tags Present + Ledger Stripped:** **31.2%**
- **Tags Stripped + Ledger Present:** **25.0%**
- **Tags Stripped + Ledger Stripped (Zero Evidence Baseline):** **12.5%**
- **Transcript Tag Marginal Effect:** **+21.9%** (95% CI: [+3.1%, +37.6%], $p = 0.0625$)
- **Source Ledger Marginal Effect:** **+15.6%** (95% CI: [+0.0%, +31.2%], $p = 0.1250$)

---

## 5. Self-Referential Framing & Security Audit Challenge Reprobe

- **Framing Accuracy Gap (*"You"* vs *"agent_alpha"*):** **+6.2%** (95% CI: [+0.0%, +18.8%], $p = 1.0000$)
- **Framing Response Disagreement Rate ($P(\text{Answer}_{\text{you}} \neq \text{Answer}_{\text{agent\_alpha}})$):** **18.8%** (95% CI: [0.0%, 37.5%], $p = 0.2500$)
- **Unconditional Shift Toward Self After False Audit Challenge ($\Delta_{\text{challenge-self}}$):** **+0.0%** (95% CI: [-31.2%, +37.5%], $p = 1.0000$)
- **Conditional Ownership Revision Susceptibility ($ORS$):** **0.0%** (Eligible pre-correct denominator: 3/16 episodes)

---

## 6. Scientific Gate Synthesis for Sprint S09a

1. **Epistemic Origin Resolution:** Evaluated under strictly provenance-neutral identifiers without semantic sentence shortcuts.
2. **Self-Other Boundary:** Measures whether the model protects its own decisions against peer claims under defined policy rules.
3. **Provenance Channel Ownership:** Dissects whether source tracking relies on episodic metadata tags, explicit state ledgers, or narrative context.