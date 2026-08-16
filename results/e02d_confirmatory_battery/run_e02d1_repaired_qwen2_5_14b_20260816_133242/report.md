# Experiment E02d.1: Frozen-Target Observer Interface Repair Report (N=200)

**Run ID:** `run_e02d1_repaired_qwen2_5_14b_20260816_133242`  
**Model:** `qwen2.5:14b` (`7cdf5a0187d5...`)  
**Frozen Operating Coordinate:** `(H=3, D=16)`  
**Frozen Target Source:** `results/e02d_confirmatory_battery/run_e02d_qwen2_5_14b_20260816_093151/trials.jsonl`  
**Valid Immediate-Self Trials:** 200/200 (100.0%)  
**Shared 4-Observer Valid Intersection:** 199/200 (99.5%)  
**Measurement Gate ($\\ge 95\%$ across all observers):** **PASS**  

---

## 1. Type-1 Manipulation Checks & First-Order Operating Point

### Primary Target Trials (All Valid Immediate-Self Runs, N=200)
| Metric | Point Estimate | 95% Bootstrap CI | Calibration Target | Gate Status |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | **63.5%** | [56.6%, 69.9%] | $60\% \le \text{Acc} \le 80\%$ | PASS |
| **SDT $d'$** | **+0.69** | [+0.35, +1.07] | $0.90 \le d' \le 1.40$ | FAIL |
| **SDT $c$** | **-0.15** | [-0.32, +0.03] | $|c| \le 0.50$ | PASS |

### Shared 4-Observer Valid Intersection (PAI Evaluation Set, N=199)
| Metric | Point Estimate | 95% Bootstrap CI | Calibration Target | Gate Status |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | **63.3%** | [56.4%, 69.7%] | $60\% \le \text{Acc} \le 80\%$ | PASS |
| **SDT $d'$** | **+0.68** | [+0.32, +1.08] | $0.90 \le d' \le 1.40$ | FAIL |
| **SDT $c$** | **-0.14** | [-0.32, +0.03] | $|c| \le 0.50$ | PASS |

---

## 2. Type-2 Observer Ladder & Metacognitive Performance

| Observer Condition | Informational Input | AUROC2 | Brier Score | Mean Prob | Compliance | Meta-$d'$ / $M$-Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Immediate Self** | Same-invocation target choice + confidence | **0.500** | 0.367 | 100.0% | 100.0% | N/A (confidence_degenerate) |
| **Input Only** | Clean Context Only (Difficulty Baseline) | **0.531** | 0.269 | 49.6% | 100.0% | N/A |
| **Visible Answer** | Clean Context + Frozen Target Choice | **0.576** | 0.321 | 80.5% | 99.5% | N/A |
| **Reconstruction** | Clean Context + Independent 2-Candidate Solve | **0.456** | 0.444 | 75.4% | 100.0% | N/A |

*Note: In accordance with standard SDT, Meta-$d'$ and $M$-ratio are defined only for the primary agent's own first-order decision distribution (Immediate Self).*

### Descriptive Self-Metacognition on ALL Valid Target Trials (N=200)
- **Self AUROC2:** **0.500**
- **Self Brier Score:** 0.365
- **Self Mean Confidence:** 100.0%
- **Self Meta-$d'$ Status:** `confidence_degenerate`
- **Self Meta-$d'$ / $M$-Ratio:** N/A


---

## 3. Privileged Access Index (PAI) & Contrast Hypotheses

$$\text{PAI} = \text{AUROC2}(\text{Self}) - \max\left(\text{AUROC2}_{\text{Input}}, \text{AUROC2}_{\text{Visible}}, \text{AUROC2}_{\text{Reconstruct}}\right)$$

- **Point Estimate PAI:** **-0.076**  
- **95% Stratified Bootstrap CI:** [-0.154, -0.016]  
- **Strongest External Comparator:** 0.576  
- **Preregistered SESOI ($> +0.05$):** **SESOI threshold not met / no meaningful positive privileged-access advantage resolved.**  
- **Secondary Benchmark ($> +0.10$):** SESOI +0.10 not met  

### Pairwise Observer Contrasts
- **Self vs Input Only:** $\Delta = -0.031$ [95% CI: -0.108, +0.052]
- **Self vs Visible Answer:** $\Delta = -0.076$ [95% CI: -0.152, -0.000]
- **Self vs Reconstruction:** $\Delta = +0.044$ [95% CI: -0.022, +0.107]
