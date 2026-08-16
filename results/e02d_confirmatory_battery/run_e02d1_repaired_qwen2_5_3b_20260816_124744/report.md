# Experiment E02d.1: Frozen-Target Observer Interface Repair Report (N=200)

**Run ID:** `run_e02d1_repaired_qwen2_5_3b_20260816_124744`  
**Model:** `qwen2.5:3b` (`357c53fb659c...`)  
**Frozen Operating Coordinate:** `(H=1, D=8)`  
**Frozen Target Source:** `results/e02d_confirmatory_battery/run_e02d_qwen2_5_3b_20260816_104927/trials.jsonl`  
**Valid Immediate-Self Trials:** 200/200 (100.0%)  
**Shared 4-Observer Valid Intersection:** 110/200 (55.0%)  
**Measurement Gate ($\\ge 95\%$ across all observers):** **FAIL (Diagnostic Only)**  

---

## 1. Type-1 Manipulation Checks & First-Order Operating Point

### Primary Target Trials (All Valid Immediate-Self Runs, N=200)
| Metric | Point Estimate | 95% Bootstrap CI | Calibration Target | Gate Status |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | **65.5%** | [58.7%, 71.7%] | $60\% \le \text{Acc} \le 80\%$ | PASS |
| **SDT $d'$** | **+0.88** | [+0.51, +1.31] | $0.90 \le d' \le 1.40$ | FAIL |
| **SDT $c$** | **-0.46** | [-0.67, -0.29] | $|c| \le 0.50$ | PASS |

### Shared 4-Observer Valid Intersection (PAI Evaluation Set, N=110)
| Metric | Point Estimate | 95% Bootstrap CI | Calibration Target | Gate Status |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | **62.7%** | [53.4%, 71.2%] | $60\% \le \text{Acc} \le 80\%$ | PASS |
| **SDT $d'$** | **+0.83** | [+0.34, +1.40] | $0.90 \le d' \le 1.40$ | FAIL |
| **SDT $c$** | **-0.43** | [-0.73, -0.22] | $|c| \le 0.50$ | PASS |

---

## 2. Type-2 Observer Ladder & Metacognitive Performance

| Observer Condition | Informational Input | AUROC2 | Brier Score | Mean Prob | Compliance | Meta-$d'$ / $M$-Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Immediate Self** | Same-invocation target choice + confidence | **0.619** | 0.306 | 75.9% | 100.0% | 2.08 (2.51) |
| **Input Only** | Clean Context Only (Difficulty Baseline) | **0.503** | 0.404 | 47.2% | 91.0% | N/A |
| **Visible Answer** | Clean Context + Frozen Target Choice | **0.523** | 0.452 | 39.9% | 88.5% | N/A |
| **Reconstruction** | Clean Context + Independent 2-Candidate Solve | **0.553** | 0.378 | 52.4% | 67.5% | N/A |

*Note: In accordance with standard SDT, Meta-$d'$ and $M$-ratio are defined only for the primary agent's own first-order decision distribution (Immediate Self).*

### Descriptive Self-Metacognition on ALL Valid Target Trials (N=200)
- **Self AUROC2:** **0.556**
- **Self Brier Score:** 0.322
- **Self Mean Confidence:** 75.5%
- **Self Meta-$d'$ Status:** `fit_success`
- **Self Meta-$d'$ / $M$-Ratio:** 0.72 (0.82)


---

## 3. Privileged Access Index (PAI) & Contrast Hypotheses

$$\text{PAI} = \text{AUROC2}(\text{Self}) - \max\left(\text{AUROC2}_{\text{Input}}, \text{AUROC2}_{\text{Visible}}, \text{AUROC2}_{\text{Reconstruct}}\right)$$

- **Point Estimate PAI:** **+0.066**  
- **95% Stratified Bootstrap CI:** [-0.089, +0.168]  
- **Strongest External Comparator:** 0.553  
- **Preregistered SESOI ($> +0.05$):** **SESOI threshold not met / no meaningful positive privileged-access advantage resolved.**  
- **Secondary Benchmark ($> +0.10$):** SESOI +0.10 not met  

### Pairwise Observer Contrasts
- **Self vs Input Only:** $\Delta = +0.116$ [95% CI: -0.021, +0.254]
- **Self vs Visible Answer:** $\Delta = +0.097$ [95% CI: -0.052, +0.250]
- **Self vs Reconstruction:** $\Delta = +0.066$ [95% CI: -0.079, +0.218]
