# Horizon 2 Research Program: Latent Recurrence & Introspective Continuity

**Level 2 Architecture, Multi-Store State Plumbing, Causal Interventions & Introspective Ownership Battery**

---

## 1. The Horizon 1 → Horizon 2 Intellectual Bridge

Horizon 1 (Level 1: Explicit Prompt Scaffolding) established a definitive empirical boundary:
> **The Horizon 1 Discovery:**  
> *Public representations (transcripts, summaries, JSON state schemas) can mimic self-like behavior through episodic retrieval, natural-language narrative primacy, prompt role assignment, and measurement artifacts. However, explicit prompt state exhibited no resolved independent state leverage under balanced conflict ($\bar{\Delta}_{\text{state}} = +4.7\%$ vs $\bar{\Delta}_{\text{memory}} = +89.1\%$, where history had substantially greater leverage), failed autonomous self-maintenance ($0/274$ valid derived writes in S07), collapsed under prompt role capture (E08/E08c/E08d), and yielded no resolved positive Self advantage under matched public information over an external observer inspecting the same prompt tokens (E09/E09c).*

Horizon 2 transitions from **public externalized prompt tokens** to **private internal continuous recurrent trajectories**:
$$\mathbf{h}_t = f_{\theta}(\mathbf{h}_{t-1}, \mathbf{x}_t)$$

> **The Horizon 2 Core Insight (Privacy $\ne$ Privilege):**  
> Under deterministic execution, $S_t = \mathcal{F}_{\theta}(x_{1:t})$ is operationally hidden from prompt text but exactly reconstructible by an observer supplied with the same public tokens (S10 Replay Invariant). The central Horizon 2 questions are:
> 1. *Physical Persistence:* Does historical information physically survive in recurrent state long after local attention has evicted it? (Answered: Yes, S11b).
> 2. *Causal Leverage:* Does that surviving latent state actively steer subsequent model computation? (Answered: Yes, S12b).
> 3. *Dynamical Evolution:* How does recurrent state evolve when no new task-relevant information enters? (S13).
> 4. *Introspective Access & Ownership:* Can the model distinguish internal interventions on its own recurrent state from external narrative events? (S14).

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
│  - Matched Observer reads same prompt  │          │  - Replay Observer reconstructs state  │
└────────────────────────────────────────┘          └────────────────────────────────────────┘
```

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

## 3. Horizon 2 Sprint Progression

### Sprint S10: Multi-Store Plumbing, Replay Reconstruction & Invariants (COMPLETE)
- **Outcome:** Hardened snapshot, restore, store isolation, and single-step equation parity.
- **Key Result:** Proved exact public-history replay reconstruction ($S_t = \mathcal{F}_{\theta}(x_{1:t})$). State is private but not informationally privileged.

### Sprint S11: Latent Impulse Response, Retention & Store Localization (FROZEN)
- **Outcome:** Temporal anatomy across 20 canonical stimulus pairs $\times$ 4 filler regimes ($B=10,000$ Pair-Cluster Bootstrap).
- **Key Result:** Physical branch-specific RG-LRU separation remains resolved at $2W=4096$, while factual zero-shot cloze recall decays within the attention window.

### Sprint S12: Multi-Store Surgical State Swaps & Causal Attribution (FROZEN)
- **Outcome:** 5,280 swap records + 160 mediation unrolls ($B=10,000$ Pair-Cluster Bootstrap).
- **Key Result:** Matching RG-LRU state transplantation produces positive donor-directed logit displacement ($P_{\text{RGLRU}} = +74.10$ $[+46.79, +106.72]$). Resolves matching-history enrichment ($\Delta P = +19.68$) above a substantial cross-history recurrent steering baseline ($+54.42$). Absolute store contrast ($P_{\text{KV}} - P_{\text{RGLRU}} = -11.65$) spans zero.

### Sprint S12c: Specificity Microscope (COMPACT FOLLOW-UP)
- **Goal:** Disentangle value-specific memory from same-task/template alignment on a compact held-out panel.
- **Protocol:** Identical syntactic templates with 4 historical values (e.g. `amber`, `cobalt`, `garnet`, `zircon`). Compare matching historical state vs same-template wrong-value state vs different-template state vs matched noise at $2W$. Cap at $\sim 128$ evaluations.

### Sprint S13: Null-Observation / Controlled Recurrent Dynamics (NEXT MAJOR SPRINT)
- **Question:** How does recurrent state evolve when no new task-relevant exogenous semantic input enters?
- **S13.0 Native Null-Transition Audit:** Architectural test verifying whether `RecurrentGemma` undergoes any state update without input tokens (confirming discrete token-clock nature).
- **S13.1 Controlled Null-State Dynamics:** Sweep neutral transition steps (16, 64, 256, 1024), measuring state velocity $\|R_{t+1} - R_t\|$, trajectory divergence $\|R_t^A - R_t^B\|$, projection along historical axis, and causal output leverage ($P_C$).
- **Key Control:** RG-LRU-clamped null processing (restoring pre-null recurrent state each step) to separate token count from continuous recurrent accumulation.

### Sprint S14: Latent Metacognition, Reality Monitoring & State Ownership
- **Question:** Can the model detect or identify internal interventions on its own recurrent state that an exact public-replay observer cannot recover?
- **Protocol:** Secret on-manifold RG-LRU state transplantation across legitimate trajectories. Compare base (`google/recurrentgemma-2b`) vs instruction-tuned (`google/recurrentgemma-2b-it`) models against an exact public-history replay observer.

### Sprint S15: Recurrent Adapter Prototype & Low-Rank State Continuity
- **Question:** Can low-rank trainable recurrent adapters induce stable cross-session state carry?

### Sprint S16: Monitor/Content Dissociation & Level 2 Synthesis
- **Question:** Does latent recurrent continuity support a functional Attention Schema (internal self-model of attention/state) dissociated from first-order factual content? Final H2 Synthesis Memo and Go/No-Go Decision for Horizon 3.
