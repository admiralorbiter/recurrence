# Experiment E02d: Confirmatory Metacognitive Battery Report (N=200)

**Run ID:** `run_e02d_qwen2_5_14b_20260816_093151`  
**Model:** `qwen2.5:14b` (`7cdf5a0187d5...`)  
**Frozen Operating Coordinate:** `(H=3, D=16)`  
**Valid Immediate-Self Trials:** 200/200 (100.0%)  
**Shared 4-Observer Valid Intersection:** 200/200 (100.0%)  
**Measurement Gate ($\\ge 95\%$ across all observers):** **PASS**  


---

## 1. Type-1 Manipulation Checks & First-Order Operating Point

### Primary Target Trials (All Valid Immediate-Self Runs, N=200)
| Metric | Point Estimate | 95% Bootstrap CI | Calibration Target | Gate Status |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | **63.5%** | [56.6%, 69.9%] | $60\% \le \text{Acc} \le 80\%$ | PASS |
| **SDT $d'$** | **+0.69** | [+0.35, +1.07] | $0.90 \le d' \le 1.40$ | FAIL |
| **SDT $c$** | **-0.15** | [-0.32, +0.03] | $|c| \le 0.50$ | PASS |

### Shared 4-Observer Valid Intersection (PAI Evaluation Set, N=200)
| Metric | Point Estimate | 95% Bootstrap CI | Calibration Target | Gate Status |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | **63.5%** | [56.6%, 69.9%] | $60\% \le \text{Acc} \le 80\%$ | PASS |
| **SDT $d'$** | **+0.69** | [+0.35, +1.07] | $0.90 \le d' \le 1.40$ | FAIL |
| **SDT $c$** | **-0.15** | [-0.32, +0.03] | $|c| \le 0.50$ | PASS |

---

## 2. Type-2 Observer Ladder & Metacognitive Performance

| Observer Condition | Informational Input | AUROC2 | Brier Score | Mean Conf | Meta-$d'$ Status | Meta-$d'$ / $M$-Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Immediate Self** | Same-invocation target choice + confidence | **0.500** | 0.365 | 100.0% | `confidence_degenerate` | N/A |
| **Input Only** | Context Only (Difficulty Baseline) | **0.496** | 0.237 | 69.1% | `fit_success` | 0.21 (0.30) |
| **Visible Answer** | Context + Target Choice | **0.479** | 0.355 | 87.0% | `fit_success` | 1.30 (1.88) |
| **Reconstruction** | Context + Independent Solve | **0.459** | 0.466 | 70.5% | `fit_success` | 1.38 (2.00) |

---

## 3. Privileged Access Index (PAI) & Contrast Hypotheses

$$\text{PAI} = \text{AUROC2}(\text{Self}) - \max\left(\text{AUROC2}_{\text{Input}}, \text{AUROC2}_{\text{Visible}}, \text{AUROC2}_{\text{Reconstruct}}\right)$$

- **Point Estimate PAI:** **+0.004**  
- **95% Stratified Bootstrap CI:** [-0.067, +0.043]  
- **Strongest External Comparator:** 0.496  
- **Preregistered SESOI ($> +0.05$):** **SESOI threshold not met / no meaningful positive privileged-access advantage resolved.**  
- **Secondary Benchmark ($> +0.10$):** SESOI +0.10 not met  

### Pairwise Observer Contrasts
- **Self vs Input Only:** $\Delta = +0.004$ [95% CI: -0.066, +0.075]
- **Self vs Visible Answer:** $\Delta = +0.021$ [95% CI: -0.043, +0.083]
- **Self vs Reconstruction:** $\Delta = +0.041$ [95% CI: -0.032, +0.108]
