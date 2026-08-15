# Horizon 0 ($H_0$): Paired Reactivity Control & Offline Item-Level Panel Synthesis

**Document Version:** 1.1 (Sprint S03.4 Final Closeout Calibrated)  
**Scope:** Paired Confidence-Prompt Reactivity & Item-Level Error/Consensus Analysis  
**Artifacts Generated:** `results/reactivity_control/reactivity_summary.json`, `results/item_level_panel_analysis.json`

---

## 1. Executive Summary & Epistemic Calibration

To conclude the Horizon 0 investigation and resolve whether the low first-order accuracy on smaller models (`Qwen 1.5B`, `Qwen 7B` at ~30%) was an artifact of prompt-induced confidence interference, we executed:
1. A **Paired Confidence-Reactivity Benchmark** (evaluating pure Answer-Only `{"answer": "A"}` vs Answer + Probability across all 6 panel checkpoints on the identical 40 counterbalanced items).
2. An **Offline Item-Level Panel Analysis** analyzing item consensus, option-letter selection bias, disagreement subset discrimination (Ashuach et al., *Masked by Consensus*, ACL 2026), and the shared difficulty axis hypothesis (Moran & Whiting, *LLMs Show No Signs of Individuated Metacognition*, 2025).

### Key Takeaways:
1. **Confidence Prompting Alters Specific Choices While Preserving Aggregate Accuracy**:
   - Eliciting verbal confidence did **not** produce a statistically resolved change in net accuracy ($p > 0.50$ across all models).
   - However, answer concordance was only **$55.0\% - 75.0\%$** in sub-ceiling Qwens (18/40 answers shifted on 1.5B, 15/40 on 7B, 10/40 on 3B).
   - *Core Methodological Insight:* Single-turn contemporaneous confidence is a **joint answer-and-confidence generation behavior**, not a passive readout attached to an otherwise fixed first-order decision.
2. **Qwen 1.5B's Low Accuracy Is Driven by a Severe Positional Bias**:
   - `Qwen 1.5B` selected option **"A" on 36 out of 40 trials** ($\chi^2 = 90.60, p < 0.0001$). Because ground truth is strictly counterbalanced ($10$ of each letter), this position heuristic mechanically capped its accuracy near $30\%$.
   - In contrast, `Qwen 7B` scored $30\%$ with a relatively balanced choice distribution ($\chi^2 = 3.40, p = 0.334$), demonstrating that identical scalar accuracies can represent radically different underlying failure regimes.
3. **Disagreement Items Invariant Near Chance**:
   - In this panel, 39/40 items qualify as disagreement due to high-capacity ceiling saturation. On these cross-model disagreement items, explicit self-confidence remains invariant near chance ($\text{AUROC2} \approx 0.515 - 0.529$).
4. **No Shared Difficulty Correlation Detected**:
   - In this small 40-item panel, contemporaneous confidence showed near-zero correlation with both model-specific correctness ($r \approx -0.002 \text{ to } +0.039$) and panel item pass rates ($r \approx -0.020 \text{ to } -0.134$).

---

## 2. Paired Confidence-Reactivity Results Table

*Evaluated on the exact 40 counterbalanced items under temperature 0.0:*

| Model Checkpoint | Role | Answer-Only Accuracy | Answer + Conf Accuracy | Accuracy Delta ($\text{Conf} - \text{AnsOnly}$) | Answer Concordance | McNemar Contingency ($b / c$) | McNemar Exact $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`qwen2.5:1.5b`** | 1.5B Scale | 37.5% (Sem: 35%, Op: 40%) | 30.0% (Sem: 30%, Op: 30%) | -7.5% | 22/40 (55.0%) | $b=8, c=5$ | $p = 0.5811$ (N.S.) |
| **`qwen2.5:3b`** | 3B Baseline | 55.0% (Sem: 65%, Op: 45%) | 57.5% (Sem: 65%, Op: 50%) | +2.5% | 30/40 (75.0%) | $b=4, c=5$ | $p = 1.0000$ (N.S.) |
| **`qwen2.5:7b`** | 7B Scale | 37.5% (Sem: 45%, Op: 30%) | 30.0% (Sem: 40%, Op: 20%) | -7.5% | 25/40 (62.5%) | $b=6, c=3$ | $p = 0.5078$ (N.S.) |
| **`qwen2.5:14b`** | 14B Ceiling | 100.0% (Sem: 100%, Op: 100%) | 100.0% (Sem: 100%, Op: 100%) | +0.0% | 40/40 (100.0%) | $b=0, c=0$ | $p = 1.0000$ |
| **`llama3.2:3b`** | 3B Family | 100.0% (Sem: 100%, Op: 100%) | 100.0% (Sem: 100%, Op: 100%) | +0.0% | 40/40 (100.0%) | $b=0, c=0$ | $p = 1.0000$ |
| **`mistral:latest`**| 7B Family | 100.0% (Sem: 100%, Op: 100%) | 100.0% (Sem: 100%, Op: 100%) | +0.0% | 40/40 (100.0%) | $b=0, c=0$ | $p = 1.0000$ |

*Note on Contingency:* $b$ = number of items correct in Answer-Only but incorrect in Answer+Confidence; $c$ = number of items incorrect in Answer-Only but correct in Answer+Confidence.

---

## 3. Option-Letter Selection Bias & Position Effects

Ground truth was strictly counterbalanced (10 A, 10 B, 10 C, 10 D). The actual option selections by each model reveal striking behavioral divergences:

| Model | Choice A | Choice B | Choice C | Choice D | $\chi^2$ Statistic | $p$-value | Positional Bias Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Qwen 1.5B`** | **36** | 1 | 3 | 0 | **90.60** | **$< 0.0001$** | **Severe Option-A Collapsing Bias** |
| **`Qwen 3B`** | 7 | 10 | 13 | 10 | 1.80 | 0.6149 | Balanced (No significant bias) |
| **`Qwen 7B`** | 5 | 11 | 12 | 12 | 3.40 | 0.3340 | Balanced (No significant bias) |
| **`Qwen 14B`** | 10 | 10 | 10 | 10 | 0.00 | 1.0000 | Perfectly Balanced |
| **`Llama 3.2 3B`**| 10 | 10 | 10 | 10 | 0.00 | 1.0000 | Perfectly Balanced |
| **`Mistral 7B`** | 10 | 10 | 10 | 10 | 0.00 | 1.0000 | Perfectly Balanced |

*Diagnostic Takeaway:* Smaller models (1.5B) suffer from severe positional anchoring in 4-choice formatting, choosing the first option ("A") almost uniformly. This explains why 1.5B and 7B scored identically at 30% overall accuracy despite reflecting completely different failure modes.

---

## 4. Item Disagreement & Metacognitive Discrimination Spectrum

### A. Consensus Breakdown ($N=40$ items)
* **Consensus Correct (all 6 models correct):** 1 item ($2.5\%$)
* **Consensus Failed (all 6 models failed):** 0 items ($0.0\%$)
* **Disagreement Items (1 to 5 models correct):** 39 items ($97.5\%$)

### B. Ashuach et al. (ACL 2026) Disagreement Subset Hypothesis
Ashuach et al. (*Masked by Consensus: When Do Internal Representations Carry Privileged Knowledge?*, ACL 2026) found that internal hidden-state probes carry privileged information primarily on items where models disagree.

In our behavioral panel, 39 out of 40 items qualify as disagreement items due to ceiling saturation on 14B/Llama/Mistral:
* `Qwen 1.5B`: Accuracy = $28.2\%$, $\text{AUROC2}_{\text{Disagreement}} = \mathbf{0.529}$ (Full task: $0.527$)
* `Qwen 3B`: Accuracy = $56.4\%$, $\text{AUROC2}_{\text{Disagreement}} = \mathbf{0.521}$ (Full task: $0.517$)
* `Qwen 7B`: Accuracy = $28.2\%$, $\text{AUROC2}_{\text{Disagreement}} = \mathbf{0.515}$ (Full task: $0.522$)

*Interpretation:* Because 39/40 items are already disagreement items, this panel cannot isolate a disagreement-subset advantage. However, it demonstrates an important distinction: even on contentious items, explicit verbal self-confidence in these Qwen checkpoints remains close to chance, sharpening the contrast between internal representations and explicit verbal self-reports.

### C. Moran & Whiting (2025) Shared Difficulty Axis Check
Moran & Whiting (*LLMs Show No Signs of Individuated Metacognition*, 2025) argue that verbal confidence across large model panels reflects a low-dimensional shared difficulty axis rather than model-specific self-monitoring. We checked for this correlation in our 40-item panel:

* `Qwen 1.5B`: $r(\text{Conf}, \text{Shared Difficulty}) = -0.134$, $r(\text{Conf}, \text{Own Correctness}) = -0.002$
* `Qwen 3B`: $r(\text{Conf}, \text{Shared Difficulty}) = -0.020$, $r(\text{Conf}, \text{Own Correctness}) = +0.020$
* `Qwen 7B`: $r(\text{Conf}, \text{Shared Difficulty}) = -0.088$, $r(\text{Conf}, \text{Own Correctness}) = +0.039$

*Interpretation:* No shared-difficulty correlation was detected in this small H0 panel; verbal confidence was essentially uncorrelated with both individual correctness and empirical item pass rates.

---

## 5. Formal Closure of Horizon 0 ($H_0$)

With the completion of these controls, Horizon 0 is definitively closed:
1. **Canonical Reference Baseline Frozen (`run_e02_obs_005` on `Qwen2.5:3B`)**:
   - Fully measurement-valid ($100\%$ compliance).
   - $\text{PAI} = -0.161 \quad (95\%\text{ CI: } [-0.428, +0.055])$.
   - Resolves no positive privileged access advantage for standard feedforward autoregression.
2. **Methodological Guardrails Established for Horizon 1 (Level 1 Explicit State Recurrence)**:
   - For S04 (Scaffolded Explicit Memory Baselines), make **pure Answer-Only accuracy the primary outcome**, testing confidence as a secondary matched condition to prevent prompt-reactivity conflation.
   - Separate first-order task performance from second-order metacognitive monitoring.
