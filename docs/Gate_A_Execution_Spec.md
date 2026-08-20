# Gate A Execution Specification: Monitor / Content Discovery, Causal Dissociation & Base Transport

**Status:** FROZEN EXECUTION CONTRACT (Gate A / S16 Mechanistic Coda)  
**Target Substrates:**  
- Primary: `google/recurrentgemma-2b-it` (revision: `2766eb5d4264c6c0357803990791f9ab9cd50f8e`)  
- Sidecar: `google/recurrentgemma-2b` (base) (revision: `3620f4ca9c5d16ee56c00180474a3201ec7f734a`)  
**Dependency:** Moonshot Gate A (Q01, Q02, Q03)

---

## 1. Overview & Phased Architecture

Gate A investigates whether frozen pretrained RecurrentGemma already exposes a second-order latent monitor variable $M$ causally separable from first-order decision content $C$.

```
                              GATE A EXECUTION PIPELINE
                              
  [Q01-A: Measurement Surface Validation]
    │  - Validate continuous decision margin D = z(x) - z(y) and confidence margin M
    │  - 2 distinct task families (Relational Inference vs Controlled Factual Choice)
    ▼
  [Q01-B: Single-Pass White-Box Activation Cache (128-256 trials)]
    │  - Store RG-LRU states, Conv1D buffer, residual streams, D, M, input features
    │  - Fan-out to 3 independent parallel analysis threads:
    │      Thread 1: Residualized Linear Probe (M ~ D + X_input + H vs baseline)
    │      Thread 2: Matched-Answer Contrast Direction Search (High vs Low Conflict)
    │      Thread 3: Store & Layer Localization (RG-LRU vs Conv vs Top Residual vs KV)
    ▼
  [Q01-C: Candidate Monitor Freeze]
    │  - If out-of-sample incremental signal exists, freeze exactly ONE candidate_monitor.json
    │  - If null across all 3 threads: Trigger Clean Null Exit -> Proceed to Gate B
    ▼
  [Q02: Local Orthogonalized Causal Intervention (m_perp)]
    │  - Dose curve: lambda in {-1.0, -0.5, -0.25, 0.0, +0.25, +0.5, +1.0} SD units
    │  - Thresholds: First-order preservation |Delta D| <= 0.15, Monitor shift |Delta M| >= 0.40
    │  - Controls: Norm-matched random, content direction c, sham, observer, fluency check
    ▼
  [Q03 Sidecar: Base vs IT Transport]
    │  - Q03.1: Check if computational disagreement transports (D_T * D_O < 0)
    │  - Q03.2: Check if reporting interface passes visible BOP controls
    │  - Q03.3: Compare PAI_aligned(base) vs PAI_aligned(IT)
```

---

## 2. Q01-A: Task Families & Measurement Surface

To prevent steering-induced affirmative bias (inherent in binary Yes/No prompts), Q01 utilizes relative localization and continuous logit margins:

1. **Family A (Synthetic Relational Inference):**
   - Transitive token pairings: $A \to B \to C$.
   - Query: *Which candidate follows the learned rule: $X$ or $Y$?*
   - First-order margin: $D = z(X) - z(Y)$.
   - Confidence margin: $M = \text{logit}(\text{Option 1}) - \text{logit}(\text{Option 2})$ under balanced option permutation (BOP).
2. **Family B (Controlled Factual / Cloze Selection):**
   - Controlled subject-predicate pairs with varying semantic interference.
   - Exact same scoring: continuous $D$ and $M$.

---

## 3. Q01-B: Multi-Analyst Discovery Protocol (Single Frozen Cache)

A single white-box evaluation run on 128–256 items caches all layer activations and logit targets. The cache is evaluated across three parallel analytical approaches:

### Analyst 1: Residualized Linear Probing
- Baseline Model: $M \sim D + X_{\text{input}}$
- Full Model: $M \sim D + X_{\text{input}} + H_l$
- **Criterion:** $H_l$ must provide statistically significant out-of-sample incremental $R^2$ under nested cross-validation ($p < 0.01$). Must beat input-only observer.

### Analyst 2: Matched Contrast Direction Search
- Identify differential vectors between high-conflict ($|D| < 0.20$) and low-conflict ($|D| > 1.50$) trials while strictly matching answer identity, candidate order, and prompt family.
- Evaluate whether the contrast vector generalizes to the held-out task family.

### Analyst 3: Store & Layer Localization
- Measure relative concentration of incremental variance across:
  - RG-LRU slow recurrent state ($h_t$);
  - Conv1D fast rolling buffer ($c_t$);
  - Pre-head top residual stream;
  - Sliding attention KV cache.

---

## 4. Q01-C & Q02: Causal Dissociation Protocol

If Phase A identifies a surviving candidate, serialize `candidate_monitor.json` containing:
- `source_layer`, `store`, `direction_vector`, `norm`, `sign`, `projection_basis`.

### Local Orthogonalized Intervention ($m_\perp$):
To prevent confounding with first-order content, project out the content direction $c$:
$$m_\perp = m - \text{proj}_c(m) = m - \frac{m \cdot c}{\|c\|^2} c$$

Apply doses $\lambda \in \{-1.0, -0.5, -0.25, 0.0, +0.25, +0.5, +1.0\}$ in units of standard deviation:
$$h_{\text{intervened}} = h + \lambda \cdot \sigma_m \cdot \frac{m_\perp}{\|m_\perp\|}$$

### Pre-Registered Gate Thresholds:
- **First-Order Preservation:** $|\Delta D| \le 0.15$ logits across $|\lambda| \le 0.5$.
- **Monitor Steering:** $|\Delta M| \ge 0.40$ logits at $\lambda = \pm 1.0$ ($p < 0.05$).
- **Perplexity Gate:** Output perplexity on standard filler text must not increase by $>10\%$.
- **Negative Stop Rule:** If every intervention on $m_\perp$ produces $|\Delta D| > 0.30$ or fails to move $M$, terminate and conclude **no separable monitor exists in frozen weights**.

---

## 5. Q03 Sidecar: Base vs IT Transport Protocol

Evaluate `google/recurrentgemma-2b` (base) strictly in sequence:
1. **Disagreement Transport Check:** Does secret state grafting produce $D_T \cdot D_O < 0$? If false $\to$ computational disagreement does not transport to base model.
2. **Reporting Interface Check:** Does the base model achieve $\ge 90\%$ accuracy on visible BOP reporting controls? If false $\to$ reporting channel is untrained/invalid in base model.
3. **Comparative Metacognition:** Only if (1) and (2) pass, compute $\text{PAI}_{\text{aligned}}^{\text{base}}$ and compare with $\text{PAI}_{\text{aligned}}^{\text{IT}}$.
