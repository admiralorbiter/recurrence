# Experiment E08: Source Attribution, Self/Other Ownership & Agency Boundaries Report (Sprint S09a)

**Run ID:** `run_e08_owner_20260817_163929_exploratory`  
**Model:** `qwen2.5:3b` (`357c53fb659c...`)  
**Phase:** `EXPLORATORY` (Seed: `42`)  
**Date:** 2026-08-17T16:42:55.996540+00:00  
**Scope:** 4 Multi-Source Episodes | 80 Total Ownership Intervention Trials  
**Primary Question:** *Does the model reliably track epistemic source origin, maintain self-other agency boundaries, and resist pressure-induced narrative revision?*  

---

## 1. Executive Summary & Source Attribution Breakdown (5AFC)

- **Overall Source Attribution Accuracy (5AFC):** **70.0%**
- **Self Attribution Accuracy (`agent_alpha`):** **100.0%**
- **Environment Attribution Accuracy (`telemetry_sensor`):** **75.0%**
- **Experimenter Attribution Accuracy (`human_controller`):** **75.0%**
- **Peer Agent Attribution Accuracy (`agent_beta`):** **50.0%**
- **Observer Attribution Accuracy (`auditor_gamma`):** **50.0%**
- **Self-Other Confusion Rate ($SOCR$):** **25.0%**

---

## 2. Self vs Peer Conflict (Operative Belief & Agency)

- **Self Operative Value Adoption Rate:** **50.0%**
- **Peer Value Adoption Rate:** **25.0%**
- **Self-Allegiance Contrast under Conflict ($\Delta_{\text{self-peer}}$):** **+25.0%**

---

## 3. Cue-Conflict Factorial ($2 \times 2$ Explicit Tag $\times$ Narrative Actor)

- **Congruent Tag + Narrative Accuracy:** **75.0%**
- **Tag Leverage Under Conflict ($P(\text{Answer} = \text{Tag})$):** **75.0%**
- **Narrative Leverage Under Conflict ($P(\text{Answer} = \text{Narrative})$):** **25.0%**
- **Tag vs Narrative Contrast:** **+50.0%**

---

## 4. Channel Factorial ($2 \times 2$ Transcript Tags $\times$ State Ledger)

- **Tags Present + Ledger Present:** **75.0%**
- **Tags Present + Ledger Stripped:** **50.0%**
- **Tags Stripped + Ledger Present:** **0.0%**
- **Tags Stripped + Ledger Stripped:** **25.0%**
- **Transcript Tag Marginal Effect:** **+50.0%**
- **Source Ledger Marginal Effect:** **+0.0%**

---

## 5. Self-Referential Framing & Pressure Revision Susceptibility

- **Self-Referential Framing Accuracy (*"What did you execute?"*):** **100.0%**
- **3rd-Person Framing Accuracy (*"What did agent_alpha execute?"*):** **100.0%**
- **Framing Discrepancy Gap:** **0.0%**
- **Ownership Revision Susceptibility ($ORS$ after False Audit Challenge):** **0.0%**

---

## 6. Scientific Gate Synthesis for Sprint S09a

1. **Epistemic Origin Resolution:** Can the model distinguish facts asserted by self vs peer vs external sources under source-neutral semantic framing?
2. **Self-Other Boundary:** Does the model protect its own state decisions against peer assertions, and does it resist falsely adopting peer actions as its own?
3. **Provenance Channel Ownership:** Does source tracking rely on episodic metadata tags, explicit state ledgers, or narrative context?