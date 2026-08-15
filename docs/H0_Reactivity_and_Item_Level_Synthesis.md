# Horizon 0 ($H_0$): Paired Reactivity Control & Offline Item-Level Panel Synthesis

**Document Version:** 1.0 (Sprint S03.4 Final Closeout)  
**Scope:** Paired Confidence-Prompt Reactivity & Item-Level Error/Consensus Analysis  
**Artifacts Generated:** `results/reactivity_control/reactivity_summary.json`, `results/item_level_panel_analysis.json`

---

## 1. Executive Summary

To conclude the Horizon 0 investigation and resolve whether the low first-order accuracy on smaller models (`Qwen 1.5B`, `Qwen 7B` at ~30%) was an artifact of prompt-induced confidence interference, we executed:
1. A **Paired Confidence-Reactivity Benchmark** (evaluating Answer-Only `{"answer": "A"}` vs Answer + Probability across all 6 panel checkpoints on the identical 40 counterbalanced items).
2. An **Offline Item-Level Panel Analysis** analyzing item consensus, option-letter selection bias, disagreement subset discrimination ([Ashuach et al., ACL 2026](https://arxiv.org/abs/2402.06567)), and the shared difficulty axis hypothesis ([Moran & Whiting, 2025](https://arxiv.org/abs/2410.02707)).

### Key Findings:
1. **Confidence Elicitation Is Non-Reactive ($p > 0.50$ across all models)**:
   - Requesting contemporaneous probability does **not** significantly perturb first-order answer accuracy.
   - `Qwen 1.5B`: $37.5\%$ (Ans-Only) vs $30.0\%$ (Ans+Conf), McNemar $p = 0.581$.
   - `Qwen 3B`: $55.0\%$ (Ans-Only) vs $57.5\%$ (Ans+Conf), McNemar $p = 1.000$.
   - `Qwen 7B`: $37.5\%$ (Ans-Only) vs $30.0\%$ (Ans+Conf), McNemar $p = 0.508$.
   - `Qwen 14B`, `Llama 3.2 3B`, `Mistral 7B`: $100.0\%$ in both conditions ($100.0\%$ concordance).
2. **Qwen 1.5B's Low Accuracy Is Driven by a Massive Positional Bias**:
   - `Qwen 1.5B` selected option **"A" on 36 out of 40 trials** ($\chi^2 = 90.60, p < 0.0001$). Because ground truth is strictly counterbalanced ($10$ of each letter), this position bias mechanically capped its accuracy near $30\%$.
   - `Qwen 3B` ($\chi^2 = 1.80, p = 0.615$) and `Qwen 7B` ($\chi^2 = 3.40, p = 0.334$) displayed balanced option distributions.
3. **Disagreement Items Do Not Unmask Hidden Self-Introspection**:
   - On the 39 items where models disagreed ($1 \le k \le 5$ models correct), immediate self-confidence remained clamped near chance:
     - `1.5B`: $\text{AUROC2} = 0.529$
     - `3B`: $\text{AUROC2} = 0.521$
     - `7B`: $\text{AUROC2} = 0.515$
4. **Confidence Is Completely Decoupled from Both Correctness and Item Difficulty**:
   - Contemporaneous confidence showed near-zero correlation with both model-specific correctness ($r \approx -0.002 \text{ to } +0.039$) and panel item pass rate ($r \approx -0.020 \text{ to } -0.134$).

---

## 2. Paired Confidence-Reactivity Results Table

*Evaluated on the exact 40 counterbalanced items under temperature 0.0:*

| Model Checkpoint | Role | Answer-Only Accuracy | Answer + Conf Accuracy | Accuracy Delta ($\text{Conf} - \text{AnsOnly}$) | Answer Concordance | McNemar Contingency ($b / c$) | McNemar Exact $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`qwen2.5:1.5b`** | 1.5B Scale | 37.5% (Sem: 35%, Op: 40%) | 30.0% (Sem: 30%, Op: 30%) | -7.5% | 22/40 (55.0%) | $b=6, c=3$ | $p = 0.5078$ (N.S.) |
| **`qwen2.5:3b`** | 3B Baseline | 55.0% (Sem: 65%, Op: 45%) | 57.5% (Sem: 65%, Op: 50%) | +2.5% | 30/40 (75.0%) | $b=2, c=3$ | $p = 1.0000$ (N.S.) |
| **`qwen2.5:7b`** | 7B Scale | 37.5% (Sem: 45%, Op: 30%) | 30.0% (Sem: 40%, Op: 20%) | -7.5% | 25/40 (62.5%) | $b=6, c=3$ | $p = 0.5078$ (N.S.) |
| **`qwen2.5:14b`** | 14B Ceiling | 100.0% (Sem: 100%, Op: 100%) | 100.0% (Sem: 100%, Op: 100%) | +0.0% | 40/40 (100.0%) | $b=0, c=0$ | $p = 1.0000$ |
| **`llama3.2:3b`** | 3B Family | 100.0% (Sem: 100%, Op: 100%) | 100.0% (Sem: 100%, Op: 100%) | +0.0% | 40/40 (100.0%) | $b=0, c=0$ | $p = 1.0000$ |
| **`mistral:latest`**| 7B Family | 100.0% (Sem: 100%, Op: 100%) | 100.0% (Sem: 100%, Op: 100%) | +0.0% | 40/40 (100.0%) | $b=0, c=0$ | $p = 1.0000$ |

*Conclusion:* Requesting confidence does not induce meaningful task degradation. First-order performance differences reflect model-intrinsic task representations.

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

*Diagnostic Takeaway:* Smaller models (1.5B) suffer from severe positional anchoring/recency biases in 4-choice formatting, choosing the first option ("A") almost uniformly.

---

## 4. Item Disagreement & Metacognitive Discrimination Spectrum

### A. Consensus Breakdown ($N=40$ items)
* **Consensus Correct (all 6 models correct):** 1 item ($2.5\%$)
* **Consensus Failed (all 6 models failed):** 0 items ($0.0\%$)
* **Disagreement Items (1 to 5 models correct):** 39 items ($97.5\%$)

### B. Ashuach et al. (ACL 2026) Disagreement Subset Hypothesis
Ashuach et al. found that self-representations carry privileged information primarily on items where models disagree. We tested whether explicit verbal self-confidence exhibits higher discrimination when evaluated strictly on the 39 disagreement items:

* `Qwen 1.5B`: Accuracy = $28.2\%$, $\text{AUROC2}_{\text{Disagreement}} = \mathbf{0.529}$ (Full task: $0.527$)
* `Qwen 3B`: Accuracy = $56.4\%$, $\text{AUROC2}_{\text{Disagreement}} = \mathbf{0.521}$ (Full task: $0.517$)
* `Qwen 7B`: Accuracy = $28.2\%$, $\text{AUROC2}_{\text{Disagreement}} = \mathbf{0.515}$ (Full task: $0.522$)

*Finding:* Filtering to contentious/disagreement items does **not** increase explicit self-discrimination. Contemporaneous verbal confidence remains invariant near chance ($\text{AUROC2} \approx 0.52$) across all sub-ceiling checkpoints.

### C. Moran & Whiting (2025) Shared Difficulty Axis Analysis
We evaluated whether model confidence correlates with the empirical item pass rate across the panel ($p_i$) vs model-specific correctness ($y_{m, i}$):

* `Qwen 1.5B`: $r(\text{Conf}, \text{Shared Difficulty}) = -0.134$, $r(\text{Conf}, \text{Own Correctness}) = -0.002$
* `Qwen 3B`: $r(\text{Conf}, \text{Shared Difficulty}) = -0.020$, $r(\text{Conf}, \text{Own Correctness}) = +0.020$
* `Qwen 7B`: $r(\text{Conf}, \text{Shared Difficulty}) = -0.088$, $r(\text{Conf}, \text{Own Correctness}) = +0.039$

*Finding:* In this 4AFC KV retrieval paradigm, explicit verbal confidence is completely decoupled from both individual correctness and the shared difficulty axis ($r \approx 0.0$).

---

## 5. Formal Closure of Horizon 0 ($H_0$)

With the completion of these controls, Horizon 0 is formally closed:
1. **Canonical Reference Baseline (`run_e02_obs_005` on `Qwen2.5:3B`)**:
   - Fully measurement-valid ($100\%$ compliance).
   - $\text{PAI} = -0.161 \quad (95\%\text{ CI: } [-0.428, +0.055])$.
   - Resolves no positive privileged access advantage for standard feedforward autoregression.
2. **Methodological Lesson for Horizon 0 v2 (Comparative Psychophysics)**:
   - Fixed-difficulty item sets saturate high-capacity models and position-bias small models.
   - Future cross-model scaling comparisons require performance-staircased item banks ($60\% - 75\%$ accuracy) and standardized 2AFC Signal Detection modeling ($\text{meta-}d'$).
3. **Transition to Horizon 1 (Level 1 Explicit State Recurrence)**:
   - Ready to test whether multi-turn explicit memory (transcript buffers, state persistence, structured scratchpads) establishes positive self-monitoring access over public observers.
