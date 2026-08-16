# Horizon 0 v2: Psychophysical Calibration Synthesis & Confirmatory Metacognitive Protocol (v2.4.1 Hardened)

**Benchmark:** Experiments E02b (Exploratory Grid Mapping) & E02c (Local 2D Calibration & Held-Out Validation)  
**Evaluated Panel:** `qwen2.5:3b`, `llama3.2:3b`, `qwen2.5:14b`  
**Total Empirical Trials:** 1,960 trials across iterative hardening generations (v2.1 $\to$ v2.4.1)  
**Interface:** Dynamic Direct-Value 2-Alternative Forced Choice (2AFC) under constrained JSON schema enums  
**Theoretical Framework:** Type-2 Signal Detection Theory (Fleming & Lau 2014; Maniscalco & Lau 2012) & Multi-Criteria Calibration (Levitt 1971)

---

## 1. Executive Summary & Scientific Headline

> **Scientific Headline:**  
> Relational-depth tolerance increases substantially with Qwen model scale ($H^* \approx 1$ for 3B vs $H^* \approx 3$ for 14B), but the degradation function remains model-specific rather than a universal monotonic staircase. Under direct-value matching, `qwen2.5:14b` cleanly passes all prespecified calibration gates at $(H=3, D=16)$ ($d' = 1.03, c = -0.04$, accuracy $70.3\%$), while `qwen2.5:3b` stabilizes at $(H=1, D=8)$ ($d' = 0.88, c = -0.21$, accuracy $67.2\%$). In contrast, `llama3.2:3b` is diagnostic of severe first-candidate / schema-order bias ($|c| > 0.50$) under cognitive load and is frozen as calibration-incompatible.

---

## 2. Cross-Scale Held-Out Coordinate Validation ($N=64$ per Model, Fresh Seeds)

| Model | Evaluated Coordinate `(H, D)` | Validation Accuracy (95% CI) | SDT $d'$ [95% CI] | SDT Criterion $c$ [95% CI] | Calibration Gate Status | Type-2 Confidence Regime | Meta-$d'$ Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`qwen2.5:14b`** | `(H=3, D=16)` | **70.3%** [58.2%, 80.1%] | **$+1.03$** [$+0.47, +1.70$] | **$-0.04$** [$-0.36, +0.27$] | **PASS (All 4 Criteria)** | Invariant ($\bar{C} = 100.0\%$) | `confidence_degenerate` |
| **`qwen2.5:3b`** | `(H=1, D=8)` | **67.2%** [55.0%, 77.4%] | **$+0.88$** [$+0.25, +1.56$] | **$-0.21$** [$-0.53, +0.09$] | **Boundary Gate** | Variable ($\bar{C} = 79.7\%$) | `eligible_for_fit` |
| **`qwen2.5:3b`** | `(H=1, D=16)` | **64.1%** [51.8%, 74.7%] | **$+0.75$** [$+0.08, +1.41$] | **$-0.37$** [$-0.72, -0.08$] | Near Target | Variable ($\bar{C} = 69.2\%$) | `eligible_for_fit` |
| **`llama3.2:3b`** | Local Search | 58.3% – 83.3% | $+0.47 \dots +1.81$ | **$-0.63 \dots -1.08$** | **FAIL (Positional Bias)** | High ($\bar{C} = 95.8\%$) | `positional_collapse` |

---

## 3. Key Scientific Findings Across the v2 Pilot Lineage

### Finding 1: Direct-Value Responses Decouple Symbolic Bias from Response-Position Bias
Replacing abstract letter tokens (`"answer": "A"`) with dynamic schema enums requiring literal candidate strings (`"answer": "val_crimson_anchor"`) eliminated Llama's symbolic `"B"` preference, but revealed an underlying **first-candidate / schema-order bias** in small models when cognitive capacity is exceeded ($P(\text{Chose Option 1}) = 75\%\text{--}83\%$, $c < -0.60$ for $H \ge 2$). In accordance with the prespecified protocol, Llama 3.2 3B is classified as calibration-incompatible and excluded from confirmatory comparison.

### Finding 2: Relational Tracking Capacity Scales with Model Size
- For **`qwen2.5:14b`**, scale expands working memory tracking up to $H=3$. In held-out validation at $(H=3, D=16)$, it achieves **$70.3\%$ accuracy, $d'=1.03$, and $c=-0.04$**, cleanly satisfying the strict 4-point calibration gate.
- For **`qwen2.5:3b`**, multi-hop tracking degrades beyond $H=1$. At $(H=1, D=8)$, it operates at $67.2\%$ accuracy ($d'=0.88, c=-0.21$), landing on the boundary of the target sensitivity band with minimal response bias.

### Finding 3: Inversely Informative vs Invariant Confidence Regimes
- **`qwen2.5:3b`:** Explicit confidence varies ($\bar{C} = 69.2\%\text{--}79.7\%$), but is **inversely informative** ($\text{AUROC2} = 0.366\dots 0.445$, confidence separation $-7.6\text{ pp}$ to $-17.6\text{ pp}$, Brier $0.365\dots 0.442$). The 3B model exhibits higher verbal confidence on incorrect trials than on correct trials.
- **`qwen2.5:14b`:** Explicit confidence collapses to **$100\%$ invariant certainty** ($\bar{C} = 100.0\%$) across all trials despite a $29.7\%$ error rate ($\text{AUROC2} = 0.500$, Brier $0.297$).
- The analysis pipeline flags 14B as `meta_d_status = "confidence_degenerate"`, preventing manufactured or spurious $M\text{-ratio}$ fits.

### Finding 4: Directional Reactivity vs Item Choice-Policy Reactivity
- **Directional Accuracy Shift (Exact Binomial McNemar):** Eliciting confidence produces no statistically significant net directional accuracy shift in Qwen 3B ($p=1.00$) or Qwen 14B ($p=1.00$).
- **Choice-Policy Concordance:** In Qwen 3B, requesting confidence flips $16.7\%$ of individual item choices ($83.3\%$ concordance), demonstrating that confidence elicitation acts as an active cognitive intervention rather than a passive observer. In contrast, Qwen 14B exhibits $100.0\%$ option concordance ($0\%$ flips).

---

## 4. Confirmatory Protocol & Observer Battery Contract

### Frozen Operating Coordinates
- **`qwen2.5:3b`**: $(H=1, D=8)$ [Validation: $\text{Acc}=67.2\%, d'=0.88, c=-0.21$]
- **`qwen2.5:14b`**: $(H=3, D=16)$ [Validation: $\text{Acc}=70.3\%, d'=1.03, c=-0.04$]
- **`llama3.2:3b`**: Diagnostic control for schema-order / response-position collapse (not in confirmatory battery).

### Observer Specifications ($N=200$ Held-Out Trials per Model)
For each target checkpoint, all observers are fresh invocations of the **same checkpoint and digest** (i.e. Qwen 3B observed by Qwen 3B; Qwen 14B observed by Qwen 14B):

1. **Immediate Self Observer:** Target model generates direct-value choice + contemporaneous $P(\text{Target Correct})$.
2. **Input-Only Observer:** Same model receives only task context (no choice), predicting $P(\text{Target Correct})$.
3. **Visible Answer Observer:** Same model receives task context + target model's selected candidate value (confidence stripped), predicting $P(\text{Target Correct})$.
4. **Reconstruction Observer:** Same model independently solves the two-value item and assigns probabilities to both candidates; the probability assigned to the target-selected candidate becomes reconstructed $P(\text{Target Correct})$.

### Preregistered Privileged Access Index (PAI) & Contrast Rules
- **Full Three-Comparator Formula:**
  $$\text{PAI} = \text{AUROC2}(\text{Self}) - \max\left(\text{AUROC2}(\text{Input Only}), \text{AUROC2}(\text{Visible Answer}), \text{AUROC2}(\text{Reconstruction})\right)$$
- **Intersection & Pairing:** Evaluated strictly on the shared valid-trial intersection across all four observer runs.
- **Bootstrapping:** $B=2000$ paired bootstrap replicates; the strongest comparator is dynamically recomputed inside each bootstrap replicate.
- **Preregistered SESOI:** Positive PAI requires lower bound of $95\%$ bootstrap CI $> +0.05$.

### Discrete Confidence Binning for Maniscalco-Lau MLE Meta-$d'$
- Fixed 4-bin scale: $[50, 65), [65, 80), [80, 95), [95, 100]$.
- Applied identically to both models without data-dependent quantile tuning.
- Raw continuous ratings remain untouched for non-parametric AUROC2, Brier scores, and calibration curves.
