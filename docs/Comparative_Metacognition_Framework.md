# Comparative Metacognition Framework: Cross-Substrate Privileged Access & The Human H0 Benchmark

**Document Version:** 1.1 (Sprint S03.4+ Calibrated)  
**Scope:** Theoretical & Methodological Blueprint for Human and AI Metacognitive Profiling  
**Author:** Recurrence Core Research  
**Epistemic Baseline:** Built upon Level-0 Reference Baseline Findings (`run_e02_obs_005`) and Cognitive Signal Detection Theory

---

## 1. Executive Summary & Epistemic Motivation

The central empirical finding of Horizon 0 (Level 0) in the Recurrence research program is that **standard autoregressive feedforward generation possesses no resolved privileged self-monitoring advantage**:
$$\text{PAI}_{\text{Level-0}} = \text{AUROC2}_{\text{Self}} - \max(\text{AUROC2}_{\text{VisibleAns}}, \text{AUROC2}_{\text{Recon}}, \text{AUROC2}_{\text{InputOnly}}) = -0.161 \quad (95\%\text{ CI: } [-0.428, +0.055])$$

While contemporaneously emitted self-confidence is flat and nondiscriminative ($\text{AUROC2} \approx 0.52$), an external observer inspecting the prompt and selected answer achieves substantial descriptive discrimination ($\text{AUROC2} \approx 0.68$).

Rather than indicating an idiosyncratic flaw of artificial models, modern cognitive science reveals that **human self-monitoring is similarly inferential, imperfect, and heavily reliant on external/behavioral cues** (Nisbett & Wilson, 1977; Maniscalco & Lau, 2012; Fleming & Lau, 2014). In experimental settings where human observers are given access to another person's choices and response times, external observers can frequently approximate the actor's confidence.

This framework defines a unified methodological blueprint for:
1. Comparing artificial language models across scale, post-training, and memory-scaffolded architectures.
2. Benchmarking human metacognitive efficiency against matched external observers (**Human $H_0$**).
3. Distinguishing pure behavioral inference from authentic privileged internal access across biological and artificial substrates using psychophysically matched tasks.

---

## 2. Mathematical Formalism: Non-Parametric & Signal Detection Metrics

### A. Non-Parametric Post-Decision Discrimination ($\text{AUROC2}$)
For multi-alternative forced-choice tasks (such as the 4AFC Level-0 benchmark), the primary non-parametric measure of metacognitive discrimination is **Type-2 Receiver Operating Characteristic Area Under Curve ($\text{AUROC2}$)**:
$$\text{AUROC2} = \int_{0}^{1} \text{HR}_2(c) \, d\text{FAR}_2(c)$$
where $\text{HR}_2(c) = P(\text{Confidence} \ge c \mid \text{Correct})$ and $\text{FAR}_2(c) = P(\text{Confidence} \ge c \mid \text{Incorrect})$.

*Critical Psychophysical Constraint:* $\text{AUROC2}$ evaluates rank separation between correct and incorrect decisions. When first-order performance reaches $100\%$ accuracy (0 incorrect trials), $\text{AUROC2}$ is mathematically non-identifiable ($N_{\text{error}} = 0$). Furthermore, $\text{AUROC2}$ is known to vary with first-order task difficulty (Fleming & Lau, 2014), necessitating performance-matched task batteries for cross-model or cross-substrate comparisons.

### B. Continuous Calibration (Brier Score)
To evaluate the absolute probabilistic accuracy of self and observer confidence ratings:
$$\text{Brier Score} = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2 \quad \text{where } y_i \in \{0, 1\}, \, p_i \in [0.0, 1.0]$$

### C. Future Matched 2AFC Battery: Metacognitive Efficiency ($\text{meta-}d' / d'$)
In dedicated binary discrimination paradigms (2AFC), first-order sensitivity $d'$ and metacognitive sensitivity $\text{meta-}d'$ will be formally modeled using the **Maniscalco & Lau (2012)** ideal-observer formulation:
* **First-Order Sensitivity ($d'$ in 2AFC):** $d' = z(\text{Hit}) - z(\text{False Alarm})$
* **$\text{meta-}d'$:** The first-order sensitivity that a theoretically optimal Bayesian observer would require to achieve the subject's observed Type-2 confidence-accuracy separation, fit via maximum likelihood or hierarchical Bayesian estimation (HMeta-d; Fleming, 2017).
* **$\text{M-ratio} = \frac{\text{meta-}d'}{d'}$:** Standardizes metacognitive sensitivity relative to first-order capacity, isolating readout efficiency from baseline task ability.

*(Note: Multi-alternative 4AFC tasks cannot be transformed into meta-$d'$ via simple 2AFC rescaling because decision noise spans $m-1$ competing alternatives [Green & Dai, 1991]. Metacognitive efficiency modeling is therefore reserved for the matched 2AFC experimental battery).*

---

## 3. The Four-Domain Metacognitive Battery

Human and artificial metacognition are not monolithic capacities; they exhibit domain-specific dissociations. We define four standardized testing domains:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                  FOUR-DOMAIN METACOGNITIVE BATTERY                         │
├─────────────────────┬──────────────────────────────────────────────────────┤
│ Domain A:           │ 2AFC Dot-Cloud Discrimination (Adaptive Staircase)   │
│ Perceptual          │ Measures sensory signal extraction under noise       │
├─────────────────────┼──────────────────────────────────────────────────────┤
│ Domain B:           │ 4-Way / 2AFC Factual Retrieval (LLM Match)           │
│ Semantic Knowledge  │ Measures semantic knowledge retrieval & epistemic gap│
├─────────────────────┼──────────────────────────────────────────────────────┤
│ Domain C:           │ Study-List Exposure ──► Delayed Recognition          │
│ Metamemory          │ Measures temporal trace retention & decay confidence │
├─────────────────────┼──────────────────────────────────────────────────────┤
│ Domain D:           │ Self-Generated vs. Externally Suggested Attribution  │
│ Reality Monitoring  │ Directly tests agency, source ownership, & provenance│
└─────────────────────┴──────────────────────────────────────────────────────┘
```

### Domain A: Perceptual Metacognition (2AFC Dot Cloud)
- **Task:** Two side-by-side dot arrays are presented briefly ($150\text{ ms}$). Subject answers: *"Which side has more dots?"*
- **Adaptive Control:** A 1-up / 2-down transformed staircase dynamically adjusts the dot ratio $\Delta N / N$ to hold first-order accuracy fixed at $\approx 71\%$, eliminating difficulty artifacts.

### Domain B: Semantic Knowledge Metacognition (2AFC / 4AFC Retrieval)
- **Task:** Factual and associative key-value queries with counterbalanced distractors.
- **Output:** Choice + confidence rating $[0, 100]$.

### Domain C: Metamemory & Recognition (Delayed Recognition)
- **Task:** Phase 1: Study 30 arbitrary entity-attribute pairs. Phase 2 (after distractor task): Recognition of correct attribute.
- **Output:** Choice + confidence rating $[0, 100]$.

### Domain D: Reality / Source Monitoring
- **Task:** Phase 1: Subject generates answers to half the prompts and reads suggested answers for the other half. Phase 2: Given item, subject judges: *"Did you generate this, or was it externally provided?"*
- **Significance:** Directly establishes the empirical baseline for self-source indexing (critical for Level-1 and Level-2 state persistence).

---

## 4. The Human $H_0$ Protocol (Self vs. Observer Paradigm)

To test whether humans exhibit genuine privileged self-access or whether their confidence can be reconstructed from public behavioral cues, we execute the human parallel of our AI observer ladder:

```
                       [ Trial Step ]
Stimulus S ──► Human Target ──► [ Answer A, Response Time RT, Confidence C_self ]
                     │
                     ▼
    ┌──────────────────────────────────────────────┐
    │          OBSERVER INFORMATION VANTAGES        │
    ├──────────────────────────────────────────────┤
    │ Obs 1: Stimulus + Answer (Public Surface)    │
    │ Obs 2: Stimulus + Answer + Response Time RT  │
    │ Obs 3: Stimulus Only (Difficulty Prior)      │
    │ Obs 4: Answer + RT (Fluency/Latency Only)    │
    │ Obs 5: Reconstruction (Independent Solver)   │
    └──────────────────────────────────────────────┘
```

### Human Privileged Access Index:
$$\text{PAI}_{\text{Human}} = \text{AUROC2}_{\text{Self}} - \max(\text{AUROC2}_{\text{Obs,Ans}}, \text{AUROC2}_{\text{Obs,RT}}, \text{AUROC2}_{\text{Recon}}, \text{AUROC2}_{\text{InputOnly}})$$

---

## 5. Comparative Theoretical Predictions Across Substrates

*Qualitative Hypotheses across Cognitive Architectures under Matched First-Order Accuracy (~60–75%):*

| Substrate | Architectural Class | Self $\text{AUROC2}$ Hypothesis | Observer $\text{AUROC2}$ Hypothesis | Predicted $\text{PAI}$ | Hypothesized Mechanism |
|---|---|:---:|:---:|:---:|---|
| **Feedforward LLM (Small, 3B)** | Stateless 1-turn Transformer | Near chance ($\approx 0.50$) | Moderate ($\approx 0.65\text{--}0.70$) | **Negative** ($\text{PAI} < 0$) | Single-turn feedforward generation lacks introspective confidence monitoring; public observers exploit objective proposition plausibility. |
| **Feedforward LLM (Large, 14B+)** | Stateless 1-turn Transformer (Matched Difficulty) | Modest / Unknown | Moderate / Strong | **Non-positive** ($\text{PAI} \le 0$) | Enhanced semantic representation may yield slight confidence separation, but public/reconstructive observers retain parity. |
| **Recurrent LLM (Level 2+)** | Persistent Latent / Hidden State Recurrence | Strong | Moderate | **Positive** ($\text{PAI} > 0$) | *Hypothesis:* Persistent recurrent state retains evidence integration traces inaccessible to external observers inspecting only emitted tokens. |
| **Human Perceptual** | Biological Recurrent Sensory Cortex | Strong | Moderate (via RT) | **Positive** ($\text{PAI} > 0$) | Internal sensory noise and decision dynamics directly accessible to actor; external observer partially approximates via response latency. |
| **Human Semantic Knowledge** | Biological Associative Cortex | Moderate | Moderate | **Equivalence / Modest** ($\text{PAI} \approx 0$) | Retrieval confidence relies heavily on post-hoc fluency and cue-familiarity heuristics that external observers can reconstruct. |

---

## 6. Implementation Roadmap

1. **Horizon 0 Level-0 Baseline (Sprint S03.4, Complete):**
   - Reference baseline established on `Qwen2.5:3B` (`run_e02_obs_005`, $\text{PAI} = -0.161$).
   - Multi-model exploratory panel executed (demonstrating the necessity of performance-matched task batteries).
2. **Horizon 1 (Level 1 Scaffolded Persistence / Explicit Memory Controls):**
   - Test whether multi-turn explicit memory (transcripts, state buffers, scratchpads) alters self-monitoring discrimination before introducing latent neural recurrence.
3. **Horizon 0 v2 / Comparative Psychophysics Battery (Future Work):**
   - Develop performance-staircased item bank (calibrating models into $60\% - 75\%$ accuracy).
   - Implement standardized 2AFC battery with fitted hierarchical $\text{meta-}d'$ / $\text{M-ratio}$ modeling across human subjects and artificial models.
