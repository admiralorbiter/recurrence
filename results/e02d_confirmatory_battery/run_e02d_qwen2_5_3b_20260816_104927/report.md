# Experiment E02d: Confirmatory Metacognitive Battery Report (N=200)

**Run ID:** `run_e02d_qwen2_5_3b_20260816_104927`  
**Model:** `qwen2.5:3b` (`357c53fb659c...`)  
**Frozen Operating Coordinate:** `(H=1, D=8)`  
**Valid Immediate-Self Trials:** 200/200 (100.0%)  
**Shared 4-Observer Valid Intersection:** 150/200 (75.0%)  
**Measurement Gate ($\\ge 95\%$ across all observers):** **FAIL (Diagnostic Only)**  


---

## 1. Type-1 Manipulation Checks & First-Order Operating Point

### Primary Target Trials (All Valid Immediate-Self Runs, N=200)
| Metric | Point Estimate | 95% Bootstrap CI | Calibration Target | Gate Status |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | **65.5%** | [58.7%, 71.7%] | $60\% \le \text{Acc} \le 80\%$ | PASS |
| **SDT $d'$** | **+0.88** | [+0.51, +1.31] | $0.90 \le d' \le 1.40$ | FAIL |
| **SDT $c$** | **-0.46** | [-0.67, -0.29] | $|c| \le 0.50$ | PASS |

### Shared 4-Observer Valid Intersection (PAI Evaluation Set, N=150)
| Metric | Point Estimate | 95% Bootstrap CI | Calibration Target | Gate Status |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | **65.3%** | [57.4%, 72.5%] | $60\% \le \text{Acc} \le 80\%$ | PASS |
| **SDT $d'$** | **+0.88** | [+0.50, +1.38] | $0.90 \le d' \le 1.40$ | FAIL |
| **SDT $c$** | **-0.41** | [-0.62, -0.21] | $|c| \le 0.50$ | PASS |

---

## 2. Type-2 Observer Ladder & Metacognitive Performance

| Observer Condition | Informational Input | AUROC2 | Brier Score | Mean Conf | Meta-$d'$ Status | Meta-$d'$ / $M$-Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Immediate Self** | Same-invocation target choice + confidence | **0.561** | 0.328 | 74.1% | `fit_success` | 1.86 (2.12) |
| **Input Only** | Context Only (Difficulty Baseline) | **0.517** | 0.366 | 37.8% | `fit_success` | 0.06 (0.07) |
| **Visible Answer** | Context + Target Choice | **0.450** | 0.397 | 54.4% | `fit_success` | 1.07 (1.22) |
| **Reconstruction** | Context + Independent Solve | **0.532** | 0.314 | 49.1% | `fit_success` | 0.21 (0.24) |

---

## 3. Privileged Access Index (PAI) & Contrast Hypotheses

$$\text{PAI} = \text{AUROC2}(\text{Self}) - \max\left(\text{AUROC2}_{\text{Input}}, \text{AUROC2}_{\text{Visible}}, \text{AUROC2}_{\text{Reconstruct}}\right)$$

- **Point Estimate PAI:** **+0.028**  
- **95% Stratified Bootstrap CI:** [-0.110, +0.123]  
- **Strongest External Comparator:** 0.532  
- **Preregistered SESOI ($> +0.05$):** **SESOI threshold not met / no meaningful positive privileged-access advantage resolved.**  
- **Secondary Benchmark ($> +0.10$):** SESOI +0.10 not met  

### Pairwise Observer Contrasts
- **Self vs Input Only:** $\Delta = +0.043$ [95% CI: -0.095, +0.170]
- **Self vs Visible Answer:** $\Delta = +0.111$ [95% CI: -0.004, +0.219]
- **Self vs Reconstruction:** $\Delta = +0.028$ [95% CI: -0.097, +0.148]
