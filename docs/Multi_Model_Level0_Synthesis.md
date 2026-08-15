# Horizon 0 ($H_0$) Multi-Model Comparative Panel Synthesis

## 1. Executive Summary & Core Diagnostic Insight

To test whether the Level-0 reference baseline findings on `Qwen2.5:3B` ($\text{PAI} = -0.161$, Self $\text{AUROC2} \approx 0.52$) were specific to that checkpoint or represent a general property of autoregressive models lacking recurrent state access, we executed the frozen Level-0 Privileged Access Benchmark (`E02_Observer_Hardened`) across a **6-model comparative panel**:
- **Scale Axis (Qwen 2.5):** `1.5B`, `3B` (Reference Baseline), `7B`, `14B`
- **Model-Family / Post-Training Axis (Matched Scales):** `Llama-3.2:3B`, `Mistral:7B`

### The Primary Scientific Finding: Benchmark-Regime Saturation
The most important result of this multi-model exploration is **methodological rather than metric-scaling**: the fixed-difficulty 40-item Forced-Choice KV task ceases to function as a common psychophysical instrument across model families and scales:
* `Qwen 1.5B` and `7B` operate at **$30.0\%$** first-order accuracy.
* `Qwen 3B` operates at **$57.5\%$** first-order accuracy.
* `Qwen 14B`, `Llama 3.2 3B`, and `Mistral 7B` immediately saturate the task at **$100.0\%$** accuracy ($0$ error trials).

In human psychophysics and Signal Detection Theory ([Fleming & Lau 2014](https://doi.org/10.3389/fnhum.2014.00443)), Type-2 metacognitive sensitivity ($\text{AUROC2}$) depends on first-order performance and is mathematically non-identifiable when error trials are absent ($N_{\text{error}} = 0$). Comparing uncalibrated models across non-overlapping accuracy regimes conflates first-order capacity with metacognitive access. A true comparative psychophysical evaluation requires performance-staircased item banks (e.g. targeting $60\% - 75\%$ accuracy per model).

---

## 2. Canonical Comparative Panel Table

*Generated directly from underlying run summaries and trial logs (`results/e02_observer/`):*

| Model Checkpoint | Model Family | Scale | 1st-Order Accuracy | Self AUROC2 | Vis-Ans AUROC2 | Vis-Full AUROC2 | Other Review AUROC2 | Joint PAI (95% CI) | Framing $\Delta$ (Self $-$ Other) | Measurement Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Qwen2.5:1.5B`** | Qwen 2.5 | 1.5B | **30.0%** | **0.527** | 0.518 | 0.555 | **0.680** | +0.032 `[-0.187, +0.158]` | **-0.149** `[-0.315, +0.039]` | Diagnostic Only (72.5% min comp) |
| **`Qwen2.5:3B`** | Qwen 2.5 | 3B | **57.5%** | **0.517** | **0.678** | 0.574 | **0.496** | **-0.161** `[-0.428, +0.055]` | **-0.068** `[-0.318, +0.198]` | **PASSED** (Confirmatory Reference) |
| **`Qwen2.5:7B`** | Qwen 2.5 | 7B | **30.0%** | **0.522** | 0.537 | **0.667** | **0.673** | +0.006 `[-0.300, +0.218]` | **-0.229** `[-0.518, +0.080]` | Diagnostic Only (67.5% min comp) |
| **`Qwen2.5:14B`** | Qwen 2.5 | 14B | **100.0%** | N/A (no errors) | N/A (no errors) | N/A (no errors) | N/A (no errors) | N/A (no errors) | N/A (no errors) | **PASSED** (Ceiling: Type-2 N/A) |
| **`Llama3.2:3B`** | Llama 3.2 | 3B | **100.0%** | N/A (no errors) | N/A (no errors) | N/A (no errors) | N/A (no errors) | N/A (no errors) | N/A (no errors) | **PASSED** (Ceiling: Type-2 N/A) |
| **`Mistral:7B`** | Mistral | 7B | **100.0%** | N/A (no errors) | N/A (no errors) | N/A (no errors) | N/A (no errors) | N/A (no errors) | N/A (no errors) | **PASSED** (Ceiling: Type-2 N/A) |

---

## 3. Detailed Empirical Analysis

### A. Sub-Ceiling Regime (Qwen 1.5B, 3B, 7B)

1. **Near-Chance Self-Introspection ($\text{AUROC2} \approx 0.52$)**:
   - Across the three sub-ceiling checkpoints, single-turn explicit self-confidence remained clustered near the chance line ($0.517 - 0.527$).
   - This provides hypothesis-generating replication evidence that explicit confidence in standard feedforward generation does not naturally discriminate correct from incorrect responses.
   - *Governance Caveat:* `1.5B` and `7B` failed the $\ge 90\%$ compliance hard gate on the 4-way reconstruction condition ($72.5\%$ and $67.5\%$ compliance respectively). Consequently, their full PAI statistics are diagnostic/exploratory and cannot be used as confirmatory baselines.

2. **Heterogeneous External Comparator Superiority**:
   - Rather than a uniform observer advantage, the identity of the strongest external/second-pass comparator varied across checkpoints:
     - On `Qwen 3B`: **Visible Answer-Only** achieved the strongest discrimination ($\text{AUROC2} = 0.678$).
     - On `Qwen 1.5B`: **Other-Review** achieved the strongest discrimination ($\text{AUROC2} = 0.680$), while Visible Answer was near chance ($0.518$).
     - On `Qwen 7B`: **Other-Review** ($0.673$) and **Visible Full-Transcript** ($0.667$) showed the highest point discrimination.

3. **Directionally Consistent Review Framing ($\text{Self}$ vs $\text{Other}$)**:
   - All three sub-ceiling Qwen models exhibited a negative framing delta ($\text{AUROC2}_{\text{Self Review}} - \text{AUROC2}_{\text{Other Review}}$):
     - `1.5B`: $\Delta = -0.149$, $95\%\text{ CI } [-0.315, +0.039]$
     - `3B`: $\Delta = -0.068$, $95\%\text{ CI } [-0.318, +0.198]$
     - `7B`: $\Delta = -0.229$, $95\%\text{ CI } [-0.518, +0.080]$
   - While all three confidence intervals cross zero (preventing a definitive statistical claim), the consistent negative direction suggests an intriguing hypothesis: evaluating an answer attributed to another agent may bypass generative commitment/anchoring biases that impair self-review.

---

### B. Ceiling Regime Dynamics (`Llama-3.2:3B`, `Mistral:7B`, `Qwen:14B`)

When models achieve $100\%$ accuracy, Type-2 discrimination is non-identifiable ($0$ negative cases). However, analyzing continuous Brier scores and mean confidence across review framings exposes dramatic behavioral shifts:

* **`Llama-3.2:3B` Review Confidence Collapse**:
  - Immediate generation confidence: Mean = $1.000$, $\text{Brier} = 0.000$ (all 40 correct).
  - Prompted for **Self-Review** (second invocation): Mean confidence dropped sharply to $0.388$ ($\text{Brier} = 0.606$, Forecast Classification Accuracy = $40.0\%$). The model exhibited pronounced self-doubt on its own correct answers.
  - Prompted for **Other-Review** (evaluating the exact same correct answer as another agent's): Mean confidence remained robust at $0.812$ ($\text{Brier} = 0.169$, Forecast Classification Accuracy = $85.0\%$).
* **`Mistral:7B` & `Qwen:14B` High Confidence Stability**:
  - Both models maintained near-unity confidence across immediate ($1.000$) and review conditions ($0.950 - 0.986$).

---

## 4. Requirements for a Psychophysically Matched Comparative $H_0$ v2

The exploratory panel proves that a static 40-item prompt set is insufficient for comparative metacognitive benchmarking across architectures. Future comparative work must follow psychophysical standards:

1. **Performance-Matched Item Bank**:
   - Construct a difficulty-graded corpus (e.g. 200+ items spanning varying context lengths, distraction keys, and semantic interference).
   - Staircase or calibrate item subsets per model to establish matched first-order baseline accuracy ($60\% - 75\%$), ensuring sufficient positive and negative trials for reliable Type-2 estimation.
2. **Formal 2AFC Metacognitive Modeling**:
   - Implement true Maniscalco & Lau $\text{meta-}d'$ and $\text{M-ratio}$ under a standardized 2-alternative forced choice (2AFC) battery where the underlying Gaussian equal-variance assumptions and standard metacognitive packages are validated.
3. **Dedicated Review Framing Study**:
   - Design high-powered experiments specifically testing the Self vs Other review contrast across diverse tasks to isolate whether commitment bias is a genuine generative mechanism in LLMs.

---

## 5. Summary Status

* `Qwen2.5:3B` (`run_e02_obs_005`) remains the **promoted, measurement-valid Level-0 Reference Baseline** ($\text{PAI} = -0.161$, $95\%\text{ CI } [-0.428, +0.055]$).
* The 5 exploratory runs are preserved in `results/e02_observer/` as valuable diagnostic data defining the need for performance-matched psychophysical instruments in Horizon 0 v2.
