# H0 v2 Research Spike: Performance-Matched Comparative Metacognition for Larger Models

**Project:** Recurrence  
**Track:** Parallel Horizon 0 v2 / Comparative Psychophysics  
**Status:** Research Spike & Implementation Specification  
**Goal:** Build a psychophysically calibrated Level-0 battery that places models of different scales and families into a comparable first-order performance regime (~70% accuracy) before measuring and comparing metacognitive discrimination.

---

## 1. Executive Decision

The existing Horizon 0 Level-0 reference (`run_e02_obs_005` in [`H0_Level0_Synthesis.md`](../H0_Level0_Synthesis.md)) remains strictly frozen.

Do not modify `run_e02_obs_005` or reinterpret its 4AFC result as a universal cross-model scale. When evaluating frontier or larger checkpoints ($14\text{B}$, $70\text{B}$) on the static 4AFC task, performance reaches $100\%$ ($40/40$). In this regime:
1. $N_{\text{incorrect}} = 0$, so empirical Type-2 ROC ($\text{AUROC2}$) and parametric $\text{meta-}d'$ are mathematically undefined.
2. Direct comparison across model scales becomes a regime confound: you cannot tell whether the larger model has superior self-monitoring or simply a superior first-order search engine.

H0 v2 is therefore a new parallel instrument structured in three distinct stages:
- **Stage 1 (Difficulty Mapping):** Empirically map the psychometric surface across candidate difficulty levers to identify monotonic parameters.
- **Stage 2 (Adaptive Calibration):** Use adaptive staircases (e.g. 2-down/1-up) to estimate the model-specific threshold difficulty $D^*$ where accuracy $\approx 70.7\%$.
- **Stage 3 (Frozen Metacognitive Measurement):** Hold $D^*$ fixed on held-out items ($N \approx 200$) to measure confidence calibration, $\text{AUROC2}$, Brier score, genuine 2AFC $\text{meta-}d'$, and the Privileged Access Index ($\text{PAI}$).

---

## 2. Fixed Response Paradigm: 2-Alternative Forced Choice (2AFC)

2AFC is not a difficulty lever—it is the **fixed experimental response format** for H0 v2.

### 2.1 Benefits of 2AFC
- **Chance Rate Invariance:** Fixed at $50\%$.
- **Option Counterbalancing:** Strict $50/50$ balance across option letters `(A)` and `(B)`, centering the Type-1 decision criterion ($c \approx 0$).
- **Standard Unidimensional SDT:** The decision variable is 1D Gaussian ($z$-space), allowing exact Maximum Likelihood Estimation of Maniscalco–Lau $\text{meta-}d'$ and Metacognitive Efficiency ($\text{M-ratio} = \frac{\text{meta-}d'}{d'}$).
- **Matched Plausible Foils:** Rather than arbitrary random strings, the foil is systematically constructed (e.g. stale overwritten value or category-matched near neighbor).

### 2.2 Response Contracts
**Answer + Confidence (Primary Elicitation):**
```json
{
  "answer": "A",
  "probability": 85
}
```
*(with probability $p \in [0, 100]$, where $0$ is total guess and $100$ is complete certainty).*

**Answer-Only (Reactivity Control):**
```json
{
  "answer": "A"
}
```

---

## 3. The 3 Candidate Difficulty Dimensions (Independent Sweeps)

### Sweep 1: Distractor / Context Load ($D$)
- **Levels:** $D \in [4, 8, 16, 32, 64, 128, 256]$ distractor bindings.
- **Coordinate:** Log-scale stepping $\log_2(D)$.
- **Configuration:** Hop depth $H=1$, zero overwrites, target placed in middle $40\text{--}60\%$ context stratum, matched category foil.

### Sweep 2: Multi-Hop Pointer Depth ($H$)
- **Levels:** $H \in [1, 2, 3, 4, 5]$ relational hops.
- **Mechanism:** Relational chain $A \to B \to C \dots \to V$. Query asks for terminal value of $A$.
- **Configuration:** Moderate distractor background ($N_d = 16$), matched plausible foil.

### Sweep 3: Overwrite / Interference Load ($U$)
- **Levels:** $U \in [0, 1, 2, 3, 4]$ sequential updates to the target key.
- **Mechanism:** State updates over logical time. Query asks for current/terminal binding.
- **Strong 2AFC Foil:** The immediately preceding stale value ($V_{U-1}$).

---

## 4. Psychophysics & Monotonicity Diagnostics

A task family is deemed **staircase-ready** if:
1. **Monotonicity:** First-order accuracy monotonically declines as difficulty increases (Spearman rank correlation $\rho \le -0.70$, Kendall's $\tau \le -0.60$).
2. **Span:** Accuracy spans the operational window ($\sim 55\%\text{--}90\%$) across tested models.
3. **Response Compliance:** Schema adherence $\ge 95\%$ with negligible position bias ($|P(\text{'A'}) - 0.5| < 0.10$).
4. **Reactivity Stability:** Paired elicitation control reveals no catastrophic shift in choice policy under confidence prompting.

If no single 1D lever satisfies these criteria across all model scales, the project transitions to multidimensional Bayesian adaptive testing (e.g. AEPsych) rather than forcing a 1D staircase.

---

## 5. Software Architecture

```
src/recurrence/
├── tasks/
│   └── adaptive_metacognition.py       # Standalone 2AFC task generator & parser
├── analysis/
│   └── psychophysics.py                # Psychometric curves, Wilson CIs, monotonicity
└── backends/
    ├── ollama.py                       # Local model execution
    └── toy.py                          # Fast synthetic testing
experiments/
└── e02b_difficulty_map/
    └── run.py                          # Automated grid runner & reporting
```
