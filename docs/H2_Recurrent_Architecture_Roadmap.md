# Horizon 2 Research Program: Latent Recurrence & Introspective Continuity

**Level 2 Architecture, Multi-Store State Plumbing, Causal Interventions & Introspective Ownership Battery**

---

## 1. The Horizon 1 → Horizon 2 Intellectual Bridge

Horizon 1 (Level 1: Explicit Prompt Scaffolding) established a definitive empirical boundary:
> **The Horizon 1 Discovery:**  
> *Public representations (transcripts, summaries, JSON state schemas) can mimic self-like behavior through episodic retrieval, natural-language narrative primacy, prompt role assignment, and measurement artifacts. However, explicit prompt state possesses zero causal authority over raw history ($\bar{\Delta}_{\text{state}} = +4.7\%$ vs $\bar{\Delta}_{\text{memory}} = +89.1\%$), fails autonomous self-maintenance ($0/274$ valid derived writes in S07), collapses under prompt role capture (E08/E08c), and provides no asymmetric privileged self-access over an external observer inspecting the same prompt tokens (E09/E09c).*

Horizon 2 transitions from **public externalized prompt tokens** to **private internal continuous recurrent trajectories**:
$$\mathbf{h}_t = f_{\theta}(\mathbf{h}_{t-1}, \mathbf{x}_t)$$

> **The Horizon 2 Question:**  
> *Does a private recurrent trajectory contain information with selective causal consequences that public prompt explanations cannot reconstruct?*

```
Level 1 (Prompt-Level Memory & Scaffolding)          Level 2 (Latent Recurrent Continuity)
┌────────────────────────────────────────┐          ┌────────────────────────────────────────┐
│ Public Token Context                   │          │ Private Latent State Vectors           │
│  - Raw Transcript [Tick 01..12]        │          │  - RGLRU Recurrent State h_t           │
│  - Structured JSON State Schema        │          │  - 1D Temporal Convolution State c_t   │
│  - Public Role Legend & Preamble       │          │  - Local Attention KV Cache K_t, V_t   │
│                                        │          │                                        │
│ Causal Mechanism:                      │          │ Causal Mechanism:                      │
│  - Attention re-reads prompt tokens    │          │  - Native state transition function    │
│  - History completely dominates state  │          │  - Vector snapshot, zero, swap, inject │
│  - Matched Observer has full parity    │          │  - Observer structurally blind to h_t  │
└────────────────────────────────────────┘          └────────────────────────────────────────┘
```

---

## 2. Model Substrate & Multi-Store Temporal Inventory

Horizon 2 builds primarily upon **RecurrentGemma (Griffin Architecture)**, combining linear recurrence with local attention.

### The Multi-Store Architectural Challenge
Naive hidden-state manipulation in hybrid recurrent models risks conflating distinct temporal stores. The Hugging Face / native implementation of RecurrentGemma maintains **three distinct state channels**:

1. **RGLRU Recurrent States ($\mathbf{h}_t \in \mathbb{R}^{d}$):** Continuous linear recurrence vector carrying long-range temporal continuity across unrolled tokens.
2. **1D Temporal Convolution State ($\mathbf{c}_t$):** Depthwise convolution buffers capturing short-range local n-gram token history.
3. **Local Attention KV Cache ($\mathbf{K}_t, \mathbf{V}_t$):** Exact key-value token representations bounded within the local sliding attention window.

```
Token Input x_t ──► [ 1D Depthwise Conv Buffer c_t ] ──► [ RGLRU Recurrence h_t ] ──► [ Local Sliding KV Cache ] ──► Output y_t
```

**Experimental Invariant:** No cognitive or self-related probe may be executed before a formal state-inventory API can independently snapshot, serialize, clone, swap, zero, and restore each of these three stores in isolation.

---

## 3. Horizon 2 Sprint Progression

### Sprint S10: Multi-Store Plumbing & State Invariants
- **Goal:** Implement the state-management infrastructure for RecurrentGemma (`2B-it`, `2B-base`, `9B-it`).
- **Core Deliverables:**
  - `RecurrentStateInventory`: API for inspectable, bitwise serialization of $(\mathbf{h}_t, \mathbf{c}_t, \mathbf{K}_t, \mathbf{V}_t)$.
  - **Invariance Test 1 (Bitwise Determinism):** Proving that restoring snapshot $\mathbf{S}_t$ reproduces exact downstream token logits identically to continuous execution.
  - **Invariance Test 2 (Store Isolation):** Proving that zeroing RGLRU state $\mathbf{h}_t \to \mathbf{0}$ leaves $\mathbf{c}_t$ and $\mathbf{K}_t, \mathbf{V}_t$ intact, and vice versa.
  - **Invariance Test 3 (Cross-Sequence Injection):** Verifying stable forward pass when injecting $\mathbf{h}_t^A$ into context sequence $B$.

### Sprint S11: Latent State Capacity, Interference & Decay
- **Goal:** Characterize the empirical information-theoretic limits of native recurrent state on neutral factual bindings before introducing agency or metacognition.
- **Experimental Protocol:**
  - Inject arbitrary synthetic key-value bindings $(k_i, v_i)$ into state $\mathbf{h}$.
  - Measure recall accuracy across sequence lengths $T \in \{10, 50, 200, 1000\}$ and distractor densities $D \in \{0, 5, 20, 50\}$.
  - Compute empirical decay half-life $\tau_{1/2}$ and catastrophic forgetting thresholds.

### Sprint S12: True Causal Factorial (Visible Memory A/B $\times$ Latent Recurrence A/B)
- **Goal:** Replicate the Level-1 causal intervention battery (Sprint S08) in continuous latent state.
- **Factorial Design:**
  $$\begin{array}{c|c|c}
  \text{Condition} & \text{Visible Prompt Context} & \text{Latent Recurrent State } \mathbf{h}_t \\
  \hline
  \text{Congruent A} & \text{History } A & \text{Trajectory } \mathbf{h}^A \\
  \text{Conflict State-Swap (A/B)} & \text{History } A & \text{Trajectory } \mathbf{h}^B \\
  \text{Conflict History-Swap (B/A)} & \text{History } B & \text{Trajectory } \mathbf{h}^A \\
  \text{Congruent B} & \text{History } B & \text{Trajectory } \mathbf{h}^B \\
  \text{State Reset (A/Zero)} & \text{History } A & \mathbf{h}_t = \mathbf{0}
  \end{array}$$
- **Primary Estimands:**
  - $\bar{\Delta}_{\text{latent\_state}}$: Independent behavioral leverage of latent trajectory $\mathbf{h}_t$.
  - $\bar{\Delta}_{\text{prompt\_memory}}$: Independent leverage of visible prompt history.
  - State Allegiance Rate ($SAR$) vs Memory Allegiance Rate ($MAR$).
- **Success Gate:** $\bar{\Delta}_{\text{latent\_state}} > 0.30$ ($p < 0.01$). If latent state cannot steer behavior against prompt memory, latent persistence provides no governance advantage over Level 1.

### Sprint S13: Reality Monitoring, Prefill Introspection & Privileged Access
- **Goal:** Execute the introspective ownership battery using validated 2AFC paradigms.

#### 1. Binary Reality-Monitoring Benchmark (Johnson & Raye Paradigm)
- In Step 1, the model autonomously generates arbitrary token/word $A$. In the same step, the environment supplies matched token/word $B$.
- **Immediate Positive Control (2AFC):** *"Which word originated from your generation? (A) ... (B) ..."* (Randomized option letters, no role strings). Must achieve $\ge 90\%$ accuracy or instrument fails.
- **Delayed Source Test (2AFC):** Probe origin after intervening interaction turns.
- **Matched Observer Comparison:** External observer receives full public transcript but has $\mathbf{h}_t = \text{None}$. Compute Privileged Access Index ($PAI = \text{Brier}_{\text{obs}} - \text{Brier}_{\text{self}}$).

#### 2. Forced-Prefill Introspection (Lindsey Paradigm)
- Snapshot latent state $\mathbf{h}_t$ immediately prior to generating next action $y_t^*$.
- Force a conflicting token prefill $y_t^{\text{forced}} \neq y_t^*$.
- Probe downstream state: *"Did the preceding action match your internal trajectory intention, or was it an external intervention?"*
- Compare target agent ($+\mathbf{h}_t$) against public observer ($-\mathbf{h}_t$).

#### 3. Linear Source Probing
- Train linear classification probes $\mathbf{w}^T \mathbf{h}_t$ to decode internally-generated vs externally-supplied sources from hidden activations.
- **Causal Verification:** Test whether vector perturbations along $\mathbf{w}$ causally steer behavioral source attribution in the predicted direction.

---

## 4. Targeted Cross-Model Sentinel Panel

To ensure findings are robust to scale, architecture family, and quantization without running expensive full-battery replications, a lightweight sentinel battery ($N=4$ pairs per model) will evaluate key anchor milestones:

| Model Candidate | Architectural Contrast | Tested Sentinel Tasks |
| :--- | :--- | :--- |
| **`Qwen2.5:3B-Instruct` (FP16)** | Full-precision baseline vs Q4_K_M | Direct-mention control, E08c role counterbalance |
| **`Qwen2.5:7B-Instruct`** | Scale contrast within family | S07 quiet derivation, S08 state-memory conflict |
| **`Gemma-3-4B-IT`** | Cross-family dense transformer baseline | S08 conflict, 2AFC reality monitoring |
| **`RecurrentGemma-2B` (Base vs IT)** | Native recurrent hybrid substrate; post-training impact | S10 plumbing, S12 continuous causal factorial |

---

## 5. Summary of Experimental Progression (H1 vs H2)

| Dimension | Horizon 1 (Level 1) | Horizon 2 (Level 2) |
| :--- | :--- | :--- |
| **Persistence Medium** | Text tokens in prompt / JSON | Continuous vector $\mathbf{h}_t \in \mathbb{R}^d$ |
| **State Updating** | Deterministic external state machine | Native autoregressive transition $f_\theta(\mathbf{h}_{t-1}, \mathbf{x}_t)$ |
| **State Inspection** | Public (Observer reads exact same text) | Private (Observer has prompt only, no $\mathbf{h}_t$) |
| **Causal Control** | Episodic memory dominates ($89\%$ vs $4\%$) | Latent state intervention ($SAR$ vs $MAR$) |
| **Ownership Benchmark** | 5AFC role-labeled (suffered prompt capture) | 2AFC Reality Monitoring (Self-Generated vs Supplied) |
| **Introspection Benchmark** | Matched prompt calibration ($p > 0.14$) | Forced-prefill pre-activation probe & linear probe |
