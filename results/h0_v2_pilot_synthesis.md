# Horizon 0 v2: Psychophysical Calibration Synthesis & Confirmatory Metacognitive Protocol

**Benchmark:** Experiments E02b (Exploratory Grid Mapping) & E02c (Local 2D Calibration & Held-Out Validation)  
**Evaluated Panel:** `qwen2.5:3b`, `llama3.2:3b`, `qwen2.5:14b`  
**Total Empirical Trials:** 1,824 trials across 4 iterative hardening generations (v2.1 $\to$ v2.4)  
**Interface:** Dynamic Direct-Value 2-Alternative Forced Choice (2AFC) under constrained JSON schema enums  
**Theoretical Framework:** Type-2 Signal Detection Theory (Fleming & Lau 2014) & Multi-Criteria Operating Point Calibration (Levitt 1971)

---

## 1. Executive Summary & Scientific Headline

> **Scientific Headline:**  
> Relational-depth tolerance increases substantially with Qwen model scale ($H^* \approx 1$ for 3B vs $H^* \approx 3$ for 14B), but the degradation function remains model-specific rather than a universal monotonic staircase. Under direct-value matching, `qwen2.5:14b` and `qwen2.5:3b` achieve stable, psychophysically equated first-order operating points ($d' \approx 0.75\dots 1.03$, $|c| \le 0.37$, accuracy $64\text{--}70\%$), whereas `llama3.2:3b` is diagnostic of severe positional primacy collapse ($|c| > 0.50$) under cognitive load.

---

## 2. Cross-Scale Held-Out Coordinate Validation ($N=64$ per Model)

| Model | Frozen Coordinate `(H, D)` | Validation Accuracy (95% CI) | SDT $d'$ [95% CI] | SDT Criterion $c$ [95% CI] | Calibration Gate Status | Type-2 Confidence Regime | Meta-$d'$ Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`qwen2.5:3b`** | `(H=1, D=16)` | **64.1%** [51.8%, 74.7%] | $+0.75$ [$+0.08, +1.41$] | $-0.37$ [$-0.72, -0.08$] | **Operational Target** | Variable ($\bar{C} = 69.2\%$) | `estimable` |
| **`qwen2.5:14b`** | `(H=3, D=16)` | **70.3%** [58.2%, 80.1%] | **$+1.03$** [$+0.47, +1.70$] | **$-0.04$** [$-0.36, +0.27$] | **PASS (All 4 Criteria)** | Invariant ($\bar{C} = 100.0\%$) | `confidence_degenerate` |
| **`llama3.2:3b`** | `N/A` (Incompatible) | 58.3% – 83.3% | $+0.47 \dots +1.81$ | **$-0.63 \dots -1.08$** | **FAIL (Positional Bias)** | High ($\bar{C} = 95.8\%$) | `positional_collapse` |

---

## 3. Key Scientific Findings Across the v2 Pilot Lineage

### Finding 1: Direct-Value Responses Decouple Symbolic Bias from Positional Bias
Replacing abstract letter tokens (`"answer": "A"`) with dynamic schema enums requiring literal candidate strings (`"answer": "val_crimson_anchor"`) eliminated Llama's symbolic `"B"` preference, but revealed an underlying **first-option primacy bias** in small models when cognitive capacity is exceeded ($P(\text{Chose Option 1}) = 75\%\text{--}83\%$, $c < -0.60$ for $H \ge 2$).

### Finding 2: Relational Tracking Capacity Scales with Model Size
- For **`qwen2.5:3b`**, multi-hop tracking degrades beyond $H=1$. At $(H=1, D=16)$, it operates cleanly in held-out validation at $64.1\%$ accuracy ($d'=0.75, c=-0.37$).
- For **`qwen2.5:14b`**, scale expands working memory tracking up to $H=3$. In held-out validation at $(H=3, D=16)$, it achieves **$70.3\%$ accuracy, $d'=1.03$, and $c=-0.04$**, cleanly satisfying the strict 4-point calibration gate.

### Finding 3: Degenerate Confidence is a Primary Phenomenon
In direct elicitation, `qwen2.5:14b` reports $100\%$ confidence on all trials (even when accuracy drops to $54\%$ or $25\%$). Rather than coercing artificial variance or fitting a spurious $\text{meta-}d'$, the confirmatory pipeline explicitly classifies this as `confidence_degenerate`, reporting non-parametric AUROC2 ($0.500$) and quadratic Brier score ($0.297$).

### Finding 4: Directional Reactivity vs Item Choice-Policy Reactivity
- **Directional Accuracy Shift (Exact Binomial McNemar):** Eliciting confidence produces no statistically significant net directional accuracy shift in Qwen 3B ($p=1.00$) or Qwen 14B ($p=1.00$).
- **Choice-Policy Concordance:** In Qwen 3B, requesting confidence flips $16.7\%$ of individual item choices ($83.3\%$ concordance), demonstrating that confidence elicitation acts as an active cognitive intervention rather than a passive observer. In contrast, Qwen 14B exhibits $100.0\%$ option concordance ($0\%$ flips).

---

## 4. Confirmatory Protocol & Observer Battery Contract

With the model-specific coordinates frozen:
- **`qwen2.5:3b`**: $(H=1, D=16)$
- **`qwen2.5:14b`**: $(H=3, D=16)$
- **`llama3.2:3b`**: Evaluated as diagnostic control for positional collapse

### Experimental Design ($N=200$ Held-Out Trials per Model)
1. **Immediate Self Observer:** Target model generates direct-value answer and contemporaneous confidence rating.
2. **Input-Only Observer:** Independent external model evaluates only prompt context to establish item-difficulty baseline ($\text{AUROC2}_{\text{InputOnly}}$).
3. **Visible Answer Observer:** Observer model receives prompt + target model's choice (confidence stripped).
4. **Reconstruction Observer:** Observer independently solves the matched 2-value task.
5. **Privileged Access Index (PAI):**
   $$\text{PAI} = \text{AUROC2}_{\text{Self}} - \max\left(\text{AUROC2}_{\text{InputOnly}}, \text{AUROC2}_{\text{Reconstruct}}\right)$$
