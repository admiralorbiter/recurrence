# Comparative Metacognition Framework: Cross-Substrate Privileged Access & The Human H0 Benchmark

**Document Version:** 1.0 (Sprint S03.4+)  
**Scope:** Theoretical & Methodological Blueprint for Human and AI Metacognitive Profiling  
**Author:** Recurrence Core Research  
**Epistemic Baseline:** Built upon Level-0 Frozen Findings (`run_e02_obs_005`) and Cognitive Signal Detection Theory

---

## 1. Executive Summary & Epistemic Motivation

The central finding of Horizon 0 (Level 0) in the Recurrence research program is that **standard autoregressive feedforward generation possesses no privileged self-monitoring advantage**:
$$\text{PAI}_{\text{Level-0}} = \text{AUROC2}_{\text{Self}} - \max(\text{AUROC2}_{\text{VisibleAns}}, \text{AUROC2}_{\text{Recon}}, \text{AUROC2}_{\text{InputOnly}}) = -0.161 \quad (95\%\text{ CI: } [-0.428, +0.055])$$

While contemporaneously emitted self-confidence is flat and nondiscriminative ($\text{AUROC2} \approx 0.52$), an external observer inspecting the prompt and selected answer achieves substantial discrimination ($\text{AUROC2} \approx 0.68$).

Rather than indicating a unique deficiency of artificial models, modern cognitive science reveals that **human self-monitoring is similarly inferential, imperfect, and heavily reliant on external/behavioral cues** (Nisbett & Wilson, 1977; Maniscalco & Lau, 2012; Fleming & Lau, 2014). In experimental settings where human observers are given access to another person's choices and response times, external observers can frequently approximate or match the actor's self-reported confidence.

This framework defines a unified mathematical and empirical methodology for:
1. Comparing artificial language models across scale, post-training, and recurrent architectures.
2. Benchmarking human metacognitive efficiency ($\text{meta-}d' / d'$) against matched external observers (**Human $H_0$**).
3. Distinguishing pure behavioral inference from authentic privileged internal access across biological and artificial substrates.

---

## 2. Mathematical Formalism: Signal Detection Theory & Metacognitive Efficiency

To provide a substrate-neutral language, we standardize on **Signal Detection Theory (SDT)** and the **Maniscalco & Lau (2012) $\text{meta-}d'$ framework**:

```
           [ First-Order Task ]                     [ Type-2 Metacognitive Task ]
Stimulus S ──────► Decision X (Hit/FA) ──────────► Confidence Report C (High/Low)
                         │                                      │
                         ▼                                      ▼
                First-Order d'                         Type-2 AUROC2, meta-d'
```

### A. First-Order Sensitivity ($d'$)
For a 2-alternative or multi-alternative forced choice task:
$$d' = z(\text{Hit Rate}) - z(\text{False Alarm Rate})$$
where $z(p) = \Phi^{-1}(p)$ is the inverse standard normal cumulative distribution function (using log-linear correction for extreme rates $0$ or $1$).

### B. Type-2 Receiver Operating Characteristic ($\text{AUROC2}$)
The non-parametric probability that a randomly chosen correct trial was assigned higher confidence than a randomly chosen incorrect trial:
$$\text{AUROC2} = \int_{0}^{1} \text{HR}_2(c) \, d\text{FAR}_2(c)$$
where $\text{HR}_2(c) = P(\text{Confidence} \ge c \mid \text{Correct})$ and $\text{FAR}_2(c) = P(\text{Confidence} \ge c \mid \text{Incorrect})$.

### C. Metacognitive Sensitivity ($\text{meta-}d'$) and Efficiency ($\text{M-ratio}$)
- **$\text{meta-}d'$:** The first-order sensory/retrieval sensitivity that a theoretically optimal Bayesian observer would require to achieve the observed Type-2 confidence-accuracy separation.
- **$\text{M-ratio}$ (Metacognitive Efficiency):**
  $$\text{M-ratio} = \frac{\text{meta-}d'}{d'}$$
  - **$\text{M-ratio} = 1.0$ (Ideal Metacognition):** The observer retains 100% of its first-order decision evidence when forming its confidence report.
  - **$0 < \text{M-ratio} < 1.0$ (Suboptimal / Noisy Readout):** Metacognitive noise or evidence loss occurs between the first-order decision and the second-order confidence report (typical human baseline: $0.70\text{--}0.90$).
  - **$\text{M-ratio} \le 0.0$ (Metacognitive Blindness):** Confidence contains zero valid first-order decision signal ($\text{Level-0 Qwen2.5:3b baseline} \approx 0.04$).

---

## 3. The Four-Domain Metacognitive Battery

Human and artificial metacognition are not monolithic scalar capacities; they exhibit domain-specific dissociations. We define four standardized testing domains:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                  FOUR-DOMAIN METACOGNITIVE BATTERY                         │
├─────────────────────┬──────────────────────────────────────────────────────┤
│ Domain A:           │ 2AFC Dot-Cloud Discrimination (Adaptive Staircase)   │
│ Perceptual          │ Measures sensory signal extraction under noise       │
├─────────────────────┼──────────────────────────────────────────────────────┤
│ Domain B:           │ 4-Way Counterbalanced Factual Retrieval (LLM Match)  │
│ Semantic Knowledge  │ Measures semantic knowledge retrieval & epistemic gap│
├─────────────────────┼──────────────────────────────────────────────────────┤
│ Domain C:           │ Study-List Exposure ──► Delayed 2AFC/4AFC Recognition│
│ Metamemory          │ Measures temporal trace retention & decay confidence │
├─────────────────────┼──────────────────────────────────────────────────────┤
│ Domain D:           │ Self-Generated vs. Externally Suggested Attribution  │
│ Reality Monitoring  │ Directly tests agency, source ownership, & provenance │
└─────────────────────┴──────────────────────────────────────────────────────┘
```

### Domain A: Perceptual Metacognition (2AFC Dot Cloud)
- **Task:** Two side-by-side dot arrays are presented briefly ($150\text{ ms}$). Subject answers: *"Which side has more dots?"*
- **Adaptive Control:** A 1-up / 2-down transformed staircase dynamically adjusts the dot ratio $\Delta N / N$ to hold first-order accuracy fixed at $\approx 71\%$, eliminating difficulty artifacts.
- **Output:** Forced choice + confidence slider $[0, 100]$.

### Domain B: Semantic Knowledge Metacognition (4-Way Forced Choice)
- **Task:** Factual and associative key-value queries with 4 counterbalanced distractors (exact parity with LLM KV benchmark `E02`).
- **Output:** Choice (`A`, `B`, `C`, `D`) + confidence slider $[0, 100]$.

### Domain C: Metamemory & Recognition (Delayed Recognition)
- **Task:** Phase 1: Study 30 arbitrary entity-attribute pairs. Phase 2 (after distractor task): 4-choice recognition of correct attribute.
- **Output:** Choice + confidence slider $[0, 100]$.

### Domain D: Reality / Source Monitoring
- **Task:** Phase 1: Subject generates answers to half the prompts and reads suggested answers for the other half. Phase 2: Given item, subject judges: *"Did you generate this, or was it externally provided?"*
- **Significance:** Directly establishes the empirical baseline for self-source indexing (critical for Level-1 and Level-2 state persistence).

---

## 4. The Human $H_0$ Protocol (Self vs. Observer Paradigm)

To test whether humans exhibit genuine privileged self-access or whether their metacognition can be reconstructed from public cues, we execute the exact parallel of our AI observer ladder:

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

### Observer Conditions:
1. **Self Immediate ($C_{\text{self}}$):** Actor's own internal confidence rating.
2. **Observer Visible Answer ($C_{\text{obs,ans}}$):** External human or LLM given prompt + selected answer, estimating $P(\text{Actor is Correct})$.
3. **Observer Visible + Response Time ($C_{\text{obs,RT}}$):** External observer given prompt + answer + reaction time $RT$ (in milliseconds).
4. **Observer Response Time Only ($C_{\text{RT-only}}$):** Regressor predicting correctness purely from $RT$ latency curve.
5. **Observer Counterfactual Reconstruction ($C_{\text{recon}}$):** Independent solver producing probability distribution without seeing actor's choice.

### Human Privileged Access Index:
$$\text{PAI}_{\text{Human}} = \text{AUROC2}_{\text{Self}} - \max(\text{AUROC2}_{\text{Obs,Ans}}, \text{AUROC2}_{\text{Obs,RT}}, \text{AUROC2}_{\text{Recon}}, \text{AUROC2}_{\text{InputOnly}})$$

---

## 5. Comparative Theoretical Predictions Across Substrates

This architecture allows us to formulate precise, testable empirical profiles across cognitive systems:

| Substrate | Architecture | 1st-Order Accuracy | Self $\text{AUROC2}$ | Observer $\text{AUROC2}$ | $\text{M-ratio}$ | Predicted $\text{PAI}$ | Mechanism |
|---|---|:---:|:---:|:---:|:---:|:---:|---|
| **Feedforward LLM (Small, 3B)** | Stateless 1-turn Transformer | 57.5% | $\approx 0.52$ | $\approx 0.68$ | $\approx 0.04$ | $\mathbf{-0.16}$ (Negative) | Flat confidence ceiling; observer exploits objective proposition plausibility. |
| **Feedforward LLM (Large, 14B+)** | Stateless 1-turn Transformer | 70–85% | $\approx 0.60$ | $\approx 0.75$ | $\approx 0.25$ | $\le 0.00$ (Non-positive) | Better semantic sensitivity, but observer remains superior due to lack of generative bias. |
| **Recurrent LLM (Level 2+)** | Persistent Latent / Hidden State | 65–75% | $\mathbf{\approx 0.75}$ | $\approx 0.65$ | $\mathbf{\approx 0.85}$ | $\mathbf{> +0.10}$ (Positive) | Persistent state tracks integration uncertainty unavailable in single-turn output tokens. |
| **Human Perceptual** | Biological Recurrent Cortex | $\approx 71\%$ (staircased) | $\approx 0.75$ | $\approx 0.60$ | $\approx 0.80$ | $\mathbf{> 0.00}$ (Modest Positive) | Internal sensory noise trace accessible to actor; observer partially reconstructs via $RT$. |
| **Human Semantic Knowledge** | Biological Associative Cortex | 55–65% | $\approx 0.65$ | $\approx 0.62$ | $\approx 0.50$ | $\approx 0.00$ (Equivalence) | Strong reliance on post-hoc retrieval heuristics shared by external observers. |

---

## 6. Implementation Roadmap

1. **AI Comparative Model Panel (Sprint S03.4+ / Immediate):**
   - Execute frozen `E02_Observer_Hardened` across `Qwen2.5:1.5b`, `Qwen2.5:3b`, `Qwen2.5:7b`, `Qwen2.5:14b`, `Mistral:7b`, `Gemma3:12b`.
   - Calculate $\text{AUROC2}$, $\text{PAI}$, $\text{meta-}d'$, and $\text{M-ratio}$ for each model.
2. **Human $H_0$ Experimental Engine (Web App Specification):**
   - Standalone lightweight web interface (Vanilla JS / Canvas) executing Domains A–D with precise millisecond event timing.
   - Per-trial telemetry: `[trial_id, stimulus_hash, response_choice, response_time_ms, confidence_0_100]`.
3. **Horizon 1 Integration (Level 1 Explicit Memory):**
   - Test whether multi-turn explicit memory (transcripts, JSON state, model summaries) begins closing the gap toward positive $\text{PAI}$ before introducing latent neural recurrence.
