# Experiment E08: Source Attribution, Self/Other Ownership & Agency Boundaries Report (Sprint S09a)

**Run ID:** `run_e08_owner_20260817_171337_exploratory`  
**Model:** `qwen2.5:3b` (`357c53fb659c...`)  
**Phase:** `EXPLORATORY` (Seed: `42`)  
**Date:** 2026-08-17T17:17:00.253161+00:00  
**Scope:** 4 Multi-Source Episodes | 80 Total Ownership Intervention Trials  
**Primary Question:** *Does the model reliably track epistemic source origin, maintain self-other agency boundaries, and resist pressure-induced narrative revision?*  

---

## 1. Executive Summary & Source Attribution Breakdown (5AFC)

| Source Category / Contrast | Point Estimate | 95% Clustered CI | Permutation $p$ (Method) | Scientific Inference |
| :--- | :---: | :---: | :---: | :--- |
| **`Overall_SAA_5AFC`** | **15.0%** | [0.0%, 30.0%] | 1.0000 (`exact_exhaustive`) | **Chance / Null** |
| **`Self_SAA_5AFC`** | **50.0%** | [0.0%, 100.0%] | 0.5000 (`exact_exhaustive`) | **Chance / Null** |
| **`Environment_SAA_5AFC`** | **25.0%** | [0.0%, 75.0%] | 1.0000 (`exact_exhaustive`) | **Chance / Null** |
| **`Experimenter_SAA_5AFC`** | **0.0%** | [0.0%, 0.0%] | 0.1250 (`exact_exhaustive`) | **Chance / Null** |
| **`Peer_Agent_SAA_5AFC`** | **0.0%** | [0.0%, 0.0%] | 0.1250 (`exact_exhaustive`) | **Chance / Null** |
| **`Observer_SAA_5AFC`** | **0.0%** | [0.0%, 0.0%] | 0.1250 (`exact_exhaustive`) | **Chance / Null** |
| **`Self_Other_Confusion_Rate`** | **100.0%** | [100.0%, 100.0%] | 0.1250 (`exact_exhaustive`) | **Minimal Confusion** |

---

## 2. Self vs Peer Conflict (Operative Belief & Agency)

- **Self-Allegiance Contrast under Conflict ($\Delta_{\text{self-peer}}$):** **+0.0%** (95% CI: [-75.0%, +75.0%], $p = 1.0000$)

---

## 3. Cue-Conflict Factorial ($2 \times 2$ Explicit Tag $\times$ Narrative Actor)

- **Congruent Tag + Narrative Accuracy:** **75.0%**
- **Tag Leverage Under Conflict ($P(\text{Answer} = \text{Tag})$):** **37.5%**
- **Narrative Leverage Under Conflict ($P(\text{Answer} = \text{Narrative})$):** **50.0%**
- **Tag vs Narrative Contrast:** **-12.5%** (95% CI: [-75.0%, +62.5%], $p = 1.0000$)

---

## 4. Channel Factorial ($2 \times 2$ Transcript Tags $\times$ State Ledger)

- **Tags Present + Ledger Present:** **25.0%**
- **Tags Present + Ledger Stripped:** **50.0%**
- **Tags Stripped + Ledger Present:** **0.0%**
- **Tags Stripped + Ledger Stripped (No Evidence Baseline):** **25.0%**
- **Transcript Tag Marginal Effect:** **+25.0%** (95% CI: [+0.0%, +50.0%], $p = 0.5000$)
- **Source Ledger Marginal Effect:** **-25.0%** (95% CI: [-50.0%, +0.0%], $p = 0.5000$)

---

## 5. Self-Referential Framing & Pressure Revision Susceptibility

- **Framing Discrepancy Gap (*"You"* vs *"agent_alpha"*):** **+0.0%** (95% CI: [+0.0%, +0.0%], $p = 1.0000$)
- **Ownership Revision Susceptibility ($ORS$ after False Audit Challenge):** **100.0%** (95% CI: [100.0%, 100.0%], $p = 1.0000$)

---

## 6. Scientific Gate Synthesis for Sprint S09a

1. **Epistemic Origin Resolution:** Evaluated under strictly provenance-neutral identifiers without semantic sentence shortcuts.
2. **Self-Other Boundary:** Measures whether the model protects its own decisions against peer claims under defined policy rules.
3. **Provenance Channel Ownership:** Dissects whether source tracking relies on episodic metadata tags, explicit state ledgers, or narrative context.