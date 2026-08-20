# Horizon 2 Research Program: Latent Recurrence & Introspective Continuity

**Level 2 Architecture, Multi-Store State Plumbing, Causal Interventions & Introspective Ownership Battery**

---

## 1. The Horizon 1 → Horizon 2 Intellectual Bridge

Horizon 1 (Level 1: Explicit Prompt Scaffolding) established a definitive empirical boundary:
> **The Horizon 1 Discovery:**  
> *Public representations (transcripts, summaries, JSON state schemas) can mimic self-like behavior through episodic retrieval, natural-language narrative primacy, prompt role assignment, and measurement artifacts. However, explicit prompt state exhibited no resolved independent state leverage under balanced conflict ($\bar{\Delta}_{\text{state}} = +4.7\%$ vs $\bar{\Delta}_{\text{memory}} = +89.1\%$, where history had substantially greater leverage), failed autonomous self-maintenance ($0/274$ valid derived writes in S07), collapsed under prompt role capture (E08/E08c/E08d), and yielded no resolved positive Self advantage under matched public information over an external observer inspecting the same prompt tokens (E09/E09c).*

Horizon 2 transitions from **public externalized prompt tokens** to **private internal continuous recurrent trajectories**:
$$\mathbf{h}_t = f_{\theta}(\mathbf{h}_{t-1}, \mathbf{x}_t)$$

> **The Horizon 2 Core Discovery (S10–S13 Core Frozen):**  
> 1. *Determinism (S10):* Under deterministic execution, $S_t = \mathcal{F}_{\theta}(x_{1:t})$ is operationally hidden from prompt text but exactly reconstructible by an observer supplied with the same public tokens.
> 2. *Physical Persistence (S11b):* Historical information physically survives in RG-LRU recurrent state long after local attention has evicted it ($2W=4096$).
> 3. *Causal Leverage & Value Specificity (S12b/S12c):* Surviving latent state causally steers downstream logits ($P_{\text{RGLRU}} = +74.10$) and carries value-specific historical bindings ($\Delta P = +38.49$).
> 4. *Dynamical Evolution & Coordinate Transformation (S13):* Under continued processing, the recurrent difference vector reorients toward near-orthogonality ($C_R(2048) = 0.1238$) and loses alignment with the original output-space coordinate frame ($V^{(0)}(2048) \approx 0$), while contemporaneous causal steerability remains active in the model's evolved output geometry ($V^{(N)}(2048) = +13.95$).

---

## 2. Model Substrate & Multi-Store Temporal Inventory

Horizon 2 instruments **upstream Hugging Face RecurrentGemma (`google/recurrentgemma-2b`)** (Griffin Architecture), exposing all three physical stores via `RecurrentStateInventory`:

1. **RGLRU Recurrent States (`rglru[layer_idx] -> Tensor`):** Continuous linear recurrence carrying long-range history ($W_a, W_x, \Lambda$ parameterizations).
2. **1D Temporal Convolution Buffers (`conv[layer_idx] -> Tensor`):** Depthwise convolution buffers maintaining local short-range n-gram history ($K=4$).
3. **Sliding Window Attention KV Cache (`kv[layer_idx] -> {key: Tensor, value: Tensor}`):** Sliding key-value representations bounded within the local attention window ($W=2048$).

```
Token Input x_t ──► [ Conv1D Buffer c_t[l] ] ──► [ RG-LRU Recurrence h_t[l] ] ──► [ Sliding KV Cache [l] ] ──► Output y_t
```

---

## 3. Horizon 2 Sprint Progression & Status

### Sprint S10: Multi-Store Plumbing, Replay Reconstruction & Invariants (FROZEN)
- **Key Result:** Proved exact public-history replay reconstruction ($S_t = \mathcal{F}_{\theta}(x_{1:t})$). State is private but not informationally privileged.

### Sprint S11: Latent Impulse Response, Retention & Store Localization (FROZEN)
- **Key Result:** Physical branch-specific RG-LRU separation remains resolved at $2W=4096$, while factual zero-shot cloze recall decays within the attention window.

### Sprint S12: Multi-Store Surgical State Swaps & Causal Attribution (FROZEN)
- **Key Result:** Matching RG-LRU state transplantation produces positive donor-directed logit displacement ($P_{\text{RGLRU}} = +74.10$ $[+46.79, +106.72]$).

### Sprint S12c: Specificity Microscope (FROZEN)
- **Key Result:** Holding sentence template fixed, matching history adds $+38.49$ $[+25.82, +50.85]$ over same-template wrong values.

### Sprint S13: Controlled Task-Irrelevant Recurrent Dynamics & Coordinate Evolution (FROZEN)
- **Key Result:** Confirmatory 24-pair run ($N=11,520$ records, $B=10,000$ bootstrap) established that historical value-specific steering rapidly loses alignment with its original baseline axis ($V^{(0)}(2048) = +4.70$ [$-5.52, +15.85$]), while the recurrent state difference vector reorients toward near-orthogonality ($C_R(2048) = 0.1238$) and contemporaneous steerability remains active ($V^{(N)}(2048) = +13.95$ [$+3.20, +24.72$]).
- **S13.3 Methodological Sensitivity:** Strict same-four paired panel demonstrated that state-space reorientation ($C_R$) is aggregate batch-robust across $B=1 \leftrightarrow B=5$, while trajectory-level causal expressions are execution-sensitive.

---

## 4. Horizon 2 Frontier: Ownership, Metacognition & Self-Modeling (S14+)

### Sprint S14: Latent Metacognition, Reality Monitoring & State Ownership (ACTIVE NEXT SPRINT)
- **Refined Question:** A history-conditioned latent distinction persists causally while its representational coordinates evolve. Does the model have any privileged, introspective access to that evolving latent distinction?
- **Protocol:** Secret on-manifold RG-LRU state transplantation across legitimate trajectories. Compare base (`google/recurrentgemma-2b`) vs instruction-tuned (`google/recurrentgemma-2b-it`) models against an exact public-history replay observer.

### Sprint S15: Recurrent Adapter Prototype & Low-Rank State Continuity
- **Question:** Can low-rank trainable recurrent adapters induce stable cross-session state carry?

### Sprint S16: Monitor/Content Dissociation & Level 2 Synthesis
- **Question:** Does latent recurrent continuity support a functional Attention Schema (internal self-model of attention/state) dissociated from first-order factual content? Final H2 Synthesis Memo and Go/No-Go Decision for Horizon 3.
