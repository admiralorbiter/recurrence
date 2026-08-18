# Horizon 2 Research Program: Latent Recurrence & Introspective Continuity

**Level 2 Architecture, Multi-Store State Plumbing, Causal Interventions & Introspective Ownership Battery**

---

## 1. The Horizon 1 → Horizon 2 Intellectual Bridge

Horizon 1 (Level 1: Explicit Prompt Scaffolding) established a definitive empirical boundary:
> **The Horizon 1 Discovery:**  
> *Public representations (transcripts, summaries, JSON state schemas) can mimic self-like behavior through episodic retrieval, natural-language narrative primacy, prompt role assignment, and measurement artifacts. However, explicit prompt state exhibited no resolved independent state leverage under balanced conflict ($\bar{\Delta}_{\text{state}} = +4.7\%$ vs $\bar{\Delta}_{\text{memory}} = +89.1\%$, where history had substantially greater leverage), failed autonomous self-maintenance ($0/274$ valid derived writes in S07), collapsed under prompt role capture (E08/E08c/E08d), and yielded no resolved positive Self advantage under matched public information over an external observer inspecting the same prompt tokens (E09/E09c).*

Horizon 2 transitions from **public externalized prompt tokens** to **private internal continuous recurrent trajectories**:
$$\mathbf{h}_t = f_{\theta}(\mathbf{h}_{t-1}, \mathbf{x}_t)$$

> **The Horizon 2 Question:**  
> *Is native recurrent state merely hidden, or is it unreconstructible from public history? Does a private recurrent trajectory contain information with selective causal consequences that public prompt explanations cannot reconstruct?*

```
Level 1 (Prompt-Level Memory & Scaffolding)          Level 2 (Latent Recurrent Continuity)
┌────────────────────────────────────────┐          ┌────────────────────────────────────────┐
│ Public Token Context                   │          │ Private Latent State Vectors           │
│  - Raw Transcript [Tick 01..12]        │          │  - Layer-indexed RGLRU State h_t[l]    │
│  - Structured JSON State Schema        │          │  - Layer-indexed Conv1D Buffer c_t[l]  │
│  - Public Role Legend & Preamble       │          │  - Layer-indexed Sliding KV Cache [l]  │
│                                        │          │                                        │
│ Causal Mechanism:                      │          │ Causal Mechanism:                      │
│  - Attention re-reads prompt tokens    │          │  - Native state transition function    │
│  - History dominates state             │          │  - Vector snapshot, zero, swap, inject │
│  - Matched Observer reads same prompt  │          │  - Observer replays public history     │
└────────────────────────────────────────┘          └────────────────────────────────────────┘
```

---

## 2. Model Substrate & Multi-Store Temporal Inventory

Horizon 2 instruments the **pinned upstream Hugging Face RecurrentGemma (`google/recurrentgemma-2b`)** (Griffin Architecture), wrapping the official model modules rather than rebuilding layer dynamics from scratch.

### The Layer-Indexed Multi-Store Architecture
Recurrent blocks maintain internal state inside module layers, while attention layers write K/V representations to the cache. The `RecurrentStateInventory` exposes this cleanly through layer-indexed mappings:

1. **RGLRU Recurrent States (`rglru[layer_idx] -> Tensor`):** Continuous linear recurrence carry across unrolled token sequences.
2. **1D Temporal Convolution Buffers (`conv[layer_idx] -> Tensor`):** Depthwise convolution buffers maintaining local short-range n-gram history.
3. **Local Attention Sliding KV Cache (`kv[layer_idx] -> {key: Tensor, value: Tensor}`):** Sliding key-value representations bounded within the local attention window.

```
Token Input x_t ──► [ Conv1D Buffer c_t[l] ] ──► [ RG-LRU Recurrence h_t[l] ] ──► [ Sliding KV Cache [l] ] ──► Output y_t
```

**Experimental Invariant:** No cognitive or self-related probe may be executed before a formal state-inventory API can independently snapshot, serialize, clone, swap, zero, and restore each of these three stores in layer-wise isolation.

---

## 3. Horizon 2 Sprint Progression

### Sprint S10: Multi-Store Plumbing, Replay Reconstruction & Invariants
- **Goal:** Instrument upstream RecurrentGemma-2B, build `RecurrentStateInventory`, implement explicit token stepping, and verify core invariants.
- **Core Deliverables:**
  - `RecurrentGemmaAdapter`: Explicit single-token stepping adapter around upstream `model.forward()`.
  - `RecurrentStateInventory`: Layer-indexed container with device-local deep cloning and canonical CPU serialization.
  - **Invariance Test 1 (Snapshot $\to$ Restore):** Verifies restored state reproduces exact next-step logits (deterministic reference) and greedy tokens.
  - **Invariance Test 2 (Store Isolation):** Zeroing RGLRU changes no conv/KV tensor; zeroing conv changes no RGLRU/KV tensor; KV surgery changes neither recurrent store.
  - **Invariance Test 3 (One-Step RG-LRU Equation Parity):** Captures real $(x_t, r_t, i_t, h_{t-1}, h_t)$ from a live step and independently verifies Griffin Equation 4:
    $$r_t = \sigma(W_a x_t + b_a), \quad i_t = \sigma(W_x x_t + b_x), \quad a_t = a_c^{r_t}, \quad h_t = a_t \odot h_{t-1} + \sqrt{1 - a_t^2} \odot (i_t \odot x_t)$$
  - **Invariance Test 4 (Mandatory Public-History Replay Reconstruction):**
    Process token history $x_{1:t} \to S_t$. Reset model to canonical zero state and replay identical tokens $x_{1:t} \to S_t'$. Compare every RGLRU state, conv buffer, KV cache, and output logit to establish whether native recurrent state is merely operationally private or informationally privileged.
  - **Invariance Test 5 (Cloned Branch Independence):** Mutations on branch $B$ do not alter branch $A$.

### Sprint S11: Latent State Capacity, Store Localization & Natural Decay
- **Goal:** Characterize the empirical information retention limits across RGLRU, Conv1D, and KV cache without injecting synthetic off-manifold vectors.
- **Protocol:** Expose the real model to arbitrary neutral factual bindings through normal token processing, snapshot naturally resulting states, and measure retention across sequence lengths $T \in \{10, 50, 200, 1000\}$ and distractor intervals.

### Sprint S12: Multi-Store Causal Factorials (State $\times$ Memory & Component Decompositions)
- **Goal:** Execute causal interventions in continuous latent state using a scout $\to$ establish methodology.
- **Protocol (Two Stages):**
  1. **Stage 1 (Whole-State Causal Contrast):** Test entire temporal state $\mathbf{S}^A$ vs $\mathbf{S}^B$ under identical subsequent inputs to measure baseline state leverage.
  2. **Stage 2 (Component Decomposition):** Perform one-store swaps against matched baselines:
     $$\text{RGLRU}^A / \text{RGLRU}^B \quad \times \quad \text{Conv}^A / \text{Conv}^B \quad \times \quad \text{KV}^A / \text{KV}^B$$
     Isolate which physical temporal mechanism carries which causal history.

### Sprint S13: Reality Monitoring, Prefill Introspection & Reconstructed Observers
- **Goal:** Execute the introspective ownership battery using validated paradigms and reconstructed observer baselines.
- **Protocol:**
  1. **Binary Reality-Monitoring Benchmark (2AFC):** Model generates token $A$; environment supplies matched token $B$. Immediate positive control must achieve $\ge 90\%$ accuracy. Delayed source probe evaluates retention.
  2. **Reconstructed Observer Baseline:** Observer is a fresh matched RecurrentGemma initialized at zero state that reconstructs its trajectory $R(x_{1:t})$ from the public transcript. Privileged access ($PAI$) evaluates what the carried target trajectory $\mathbf{S}_t$ knows that the strongest replay observer cannot recover.
  3. **Forced-Prefill Introspection (Lindsey Paradigm):** Snapshot pre-output state $\mathbf{S}_t$, force conflicting token $y_t^{\text{forced}} \neq y_t^*$, and probe whether pre-activation state encoded intended vs forced action.

---

## 4. Targeted Cross-Model Sentinel Panel

| Model Candidate | Architectural Contrast | Tested Sentinel Tasks |
| :--- | :--- | :--- |
| **`Qwen2.5:3B-Instruct` (FP16)** | Full-precision baseline vs Q4_K_M | Direct-mention control, E08c role counterbalance |
| **`Qwen2.5:7B-Instruct`** | Scale contrast within family | S07 quiet derivation, S08 state-memory conflict |
| **`Gemma-3-4B-IT`** | Cross-family dense transformer baseline | S08 conflict, 2AFC reality monitoring |
| **`RecurrentGemma-2B` (Base vs IT)** | Pinned recurrent hybrid substrate | S10 plumbing, S12 continuous causal factorial |
