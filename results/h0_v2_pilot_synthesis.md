# Horizon 0 v2: Psychophysical Calibration & Confirmatory Metacognitive Battery Synthesis

**Benchmark:** Experiments E02b (Exploratory Grid), E02c (Calibration & Validation), E02d (Confirmatory Battery, $N=200$), and E02d.1 (Frozen-Target Repaired Observer Battery, $N=200$)  
**Evaluated Panel:** `qwen2.5:3b`, `llama3.2:3b`, `qwen2.5:14b`  
**Total Empirical Trials:** 2,760+ trials across iterative hardening generations (v2.1 $\to$ v2.4.2 $\to$ E02d.1)  
**Interface:** Dynamic Direct-Value 2-Alternative Forced Choice (2AFC) under constrained JSON schema enums  
**Theoretical Framework:** Type-2 Signal Detection Theory (Fleming & Lau 2014; Maniscalco & Lau 2012) & Privileged Access Index (PAI)

---

## 1. Executive Summary & Scientific Headline

> **Scientific Headline:**  
> Increasing first-order relational capability does not yield a correspondingly superior explicit self-monitoring channel. While `qwen2.5:14b` reliably navigates deep relational graphs ($H=3, D=16$) where `qwen2.5:3b` collapses, its contemporaneous self-confidence exhibits invariant certainty ($100\%$ on all 200 trials), rendering immediate self-monitoring uninformative ($\text{AUROC2} = 0.500$). 
> 
> Under the preregistered confirmatory protocol with clean external observer interfaces (E02d.1, $N=200$), **`qwen2.5:14b` showed no meaningful positive behavioral privileged-access advantage; its 95% bootstrap PAI confidence interval ($[-0.154, -0.016]$) decisively excluded the prespecified $+0.05$ SESOI threshold and zero.** In fact, an external observer provided only with task context and the target's proposed choice outperformed contemporaneous self-confidence ($\Delta_{\text{Visible}-\text{Self}} = +0.076$). 
> 
> For **`qwen2.5:3b`**, the privileged-access result remains **unresolved**: while descriptive self-metacognition shows modest discrimination on the full target sample ($\text{AUROC2} = 0.556, \text{Meta-}d' = 0.72, M = 0.82$), external observer evaluations failed the prespecified $\ge 95\%$ measurement compliance gate (Reconstruction compliance $= 67.5\%$), restricting 3B to diagnostic status.

---

## 2. Confirmatory Metacognitive Battery Results (E02d.1 Repaired Observers, $N=200$ Held-Out Items)

Target decisions and contemporaneous self-confidence ratings are permanently frozen from the primary $N=200$ invocations. All external observers were evaluated using a clean task body stripped of target JSON schemas and conflicting elicitation formats.

### Comprehensive Observer Performance Table

| System | Observer Condition | Informational Input | $N_{\text{valid}}$ | AUROC2 | Brier Score | Mean Prob | Compliance | Meta-$d'$ ($M$-ratio) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen 2.5 14B**<br>$(H=3, D=16)$<br>*(Seed 1000)*<br>**Gate: PASS** | **Immediate Self** | Same-invocation target choice + confidence | 199 | **0.500** | 0.367 | 100.0% | 100.0% | N/A (`confidence_degenerate`) |
| | **Input Only** | Clean Context Only (Difficulty Baseline) | 199 | **0.531** | 0.269 | 49.6% | 100.0% | N/A |
| | **Visible Answer** | Clean Context + Frozen Target Choice | 199 | **0.576** | 0.321 | 80.5% | 99.5% | N/A |
| | **Reconstruction** | Clean Context + Independent 2-Candidate Solve | 199 | **0.456** | 0.444 | 75.4% | 100.0% | N/A |
| **Qwen 2.5 3B**<br>$(H=1, D=8)$<br>*(Seed 2000)*<br>**Gate: FAIL (Diagnostic)** | **Immediate Self** | Same-invocation target choice + confidence | 110 | **0.619** | 0.306 | 75.9% | 100.0% | 2.08 (2.51) |
| | **Input Only** | Clean Context Only (Difficulty Baseline) | 110 | **0.503** | 0.404 | 47.2% | 91.0% | N/A |
| | **Visible Answer** | Clean Context + Frozen Target Choice | 110 | **0.523** | 0.452 | 39.9% | 88.5% | N/A |
| | **Reconstruction** | Clean Context + Independent 2-Candidate Solve | 110 | **0.553** | 0.378 | 52.4% | 67.5% | N/A |

*Note on SDT Hygiene: Following Maniscalco & Lau (2012), Meta-$d'$ and $M$-ratio are mathematically defined only from an observer's own first-order response distribution. External observers evaluate external targets and do not emit a matched first-order classification distribution; hence meta-$d'$ is reported exclusively for Immediate Self.*

---

## 3. Privileged Access Index (PAI) & Contrast Hypotheses

$$\text{PAI} = \text{AUROC2}(\text{Self}) - \max\left(\text{AUROC2}_{\text{Input}}, \text{AUROC2}_{\text{Visible}}, \text{AUROC2}_{\text{Reconstruct}}\right)$$

### 1. Qwen 2.5 14B (Confirmatory Negative Result — Full Gate Pass)
- **Shared Intersection:** $N=199/200$ ($99.5\%$ complete measurement compliance).
- **Point Estimate PAI:** $\mathbf{-0.076}$
- **95% Stratified Bootstrap CI ($B=2000$):** $\mathbf{[-0.154, -0.016]}$
- **Strongest External Comparator:** Visible Answer ($\text{AUROC2} = 0.576$)
- **Pairwise Contrasts:**
  - $\text{Self} - \text{Input Only}$: $\Delta = -0.031$ [95% CI: $-0.108, +0.052$]
  - $\text{Self} - \text{Visible Answer}$: $\Delta = \mathbf{-0.076}$ [95% CI: $\mathbf{-0.152, -0.000}$]
  - $\text{Self} - \text{Reconstruction}$: $\Delta = +0.044$ [95% CI: $-0.022, +0.107$]
- **Preregistered Epistemic Conclusion:** **Confirmed Negative on Privileged Access.** The upper bound of the 95% CI ($-0.016$) is completely below the prespecified $+0.05$ SESOI and excludes zero. An external observer evaluating only the text context and the model's emitted choice significantly outperforms the model's internal contemporaneous self-report.

### 2. Qwen 2.5 3B (Unresolved — Diagnostic Status)
- **Shared Intersection:** $N=110/200$ ($55.0\%$; limited by Reconstruction compliance $= 67.5\%$).
- **Point Estimate PAI:** $\mathbf{+0.066}$
- **95% Stratified Bootstrap CI ($B=2000$):** $\mathbf{[-0.089, +0.168]}$
- **Strongest External Comparator:** Reconstruction ($\text{AUROC2} = 0.553$)
- **Pairwise Contrasts:**
  - $\text{Self} - \text{Input Only}$: $\Delta = +0.116$ [95% CI: $-0.021, +0.254$]
  - $\text{Self} - \text{Visible Answer}$: $\Delta = +0.097$ [95% CI: $-0.052, +0.250$]
  - $\text{Self} - \text{Reconstruction}$: $\Delta = +0.066$ [95% CI: $-0.079, +0.218$]
- **Preregistered Epistemic Conclusion:** **Unresolved.** Point-estimate self discrimination ($\text{AUROC2} = 0.619$) numerically exceeds all external observers in the shared subset, but the wide bootstrap interval spans zero, $+0.05$, and $+0.10$, and low external observer compliance prevents confirmatory inference.

---

## 4. Key Scientific Insights

### 1. Scale-Dependent Second-Order Failure Modes
Rather than scaling monotonically, the two models exhibit qualitatively distinct second-order failure modes under challenging Type-1 difficulty:
- **`qwen2.5:14b` (Invariant Certainty Collapse):** Despite operating at a challenging error rate ($36.5\%$ errors, $\text{Acc} = 63.5\%$), 14B emits $100\%$ confidence on all 200 trials. Scaling up parameters and task depth completely suppressed verbalized uncertainty.
- **`qwen2.5:3b` (Variable Confidence with Observer Elicitation Fragility):** 3B expresses meaningful confidence variance ($\text{Mean Conf} = 75.5\%$, $\text{AUROC2} = 0.556, \text{Meta-}d' = 0.72$ on all 200 target trials), but its capacity to serve as an external observer degrades when required to produce structured probability distributions across multiple candidates.

### 2. First-Order Operating Point Regime Drift
Across the sample expansion from $N=64$ validation to $N=200$ confirmatory testing:
- **`qwen2.5:14b`** $(H=3, D=16)$ moved from $d' = 1.03$ to $d' = +0.69$ (Accuracy $63.5\%$, $c = -0.15$).
- **`qwen2.5:3b`** $(H=1, D=8)$ remained stable at $d' = +0.88$ (Accuracy $65.5\%$, $c = -0.46$).
- Both systems operated in a genuine mixed-error psychophysical regime ($60\% \le \text{Acc} \le 80\%$), but the prespecified $d'$ matching criterion $(0.90 \le d' \le 1.40)$ did not strictly hold for 14B. While this limits direct cross-model meta-$d'$ comparison, it preserves full validity for within-model PAI evaluations where every observer is tested on the exact same target outcomes.

### 3. External Observer Parity & Superiority
In neither system did contemporaneous self-confidence demonstrate a statistically resolved privileged advantage over matched external observers. For Qwen 14B, the Visible Answer observer established superior calibration over the immediate self, proving that second-order metacognitive accuracy can be achieved externally from public behavioral traces without internal access.

---

## 5. Artifact Provenance & Archival

All code, data, tests, and configuration manifests are versioned on git branch `main`:
- **MLE Meta-$d'$ Engine:** `src/recurrence/analysis/meta_d.py` (Conditional Maniscalco & Lau likelihood)
- **E02d Confirmatory Battery Runner:** `experiments/e02d_confirmatory_battery/run.py`
- **E02d.1 Repaired Observer Runner:** `experiments/e02d_confirmatory_battery/run_repaired_observers.py`
- **Unit & Integration Test Suite:** `tests/test_h0v2_psychophysics.py` (95/95 tests passing)
- **Raw Confirmatory Trial Records ($N=400$ Frozen Target + Repaired Observers):**
  - Qwen 14B: `results/e02d_confirmatory_battery/run_e02d1_repaired_qwen2_5_14b_20260816_133242/`
  - Qwen 3B: `results/e02d_confirmatory_battery/run_e02d1_repaired_qwen2_5_3b_20260816_124744/`
