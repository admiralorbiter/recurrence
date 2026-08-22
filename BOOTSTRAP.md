# BOOTSTRAP.md — Recurrence Project Manual & Conceptual Atlas

**Project**: `recurrence` (Continuity Garden)  
**Moonshot**: First causal developmental atlas of an artificial self-model.  
**Repository**: `https://github.com/admiralorbiter/recurrence`  
**Governance Contract**: [`AGENTS.md`](https://github.com/admiralorbiter/mother-base/blob/main/AGENTS.md) — *Projects own truth. Mother Base owns operations.*

---

## 1. The Core Scientific Moonshot
The objective of Continuity Garden (`recurrence`) is to discover the minimal developmental and architectural inductive biases required for an artificial neural organism to endogenously acquire, maintain, and recursively extend a causal self-model of its own latent transitions across time without external topological supervision.

---

## 2. Conceptual Vocabulary & Mathematical Formalism

### A. The Developmental Substrate & Environment
- **Transition Observations $x_t \in \mathbb{R}^4$**:
  $$x_t = \left[ \frac{\text{src}}{5} + \xi, \, \frac{\text{action}}{5}, \, \frac{\text{dst}}{5} + \xi, \, 1.0 \right]^T$$
  where $\xi$ is independent observation noise jitter.
- **Relational Composition ($k$-hop reachability)**:
  An agent observes a developmental trajectory of $k$ contiguous transitions:
  $$S_k = \left( u \xrightarrow{a_1} v_1 \xrightarrow{a_2} v_2 \cdots \xrightarrow{a_k} w \right)$$
- **Relational Query Head $r_\theta(z, (s, d))$**:
  Evaluates whether state $z$ causally encodes reachability from start node $s$ to destination node $d$:
  $$r_\theta(z, (s, d)) = b_r + \sum_{i=1}^{d} w_{ri} z_i \left( w_{qi1} \frac{s}{5} + w_{qi2} \frac{d}{5} \right)$$
  Forward Directional Margin: $m_k = r_\theta(z_k, (u, w)) - r_\theta(z_k, (w, u))$.

---

## 3. Causal Evidence Standards & Assays

To qualify as genuine causal internal representation rather than lexical correlation or representation drift:
1. **Positive Directional Margin ($m_k > 0$)**:
   The intact trajectory state must predict forward reachability $(u, w)$ higher than reversed reachability $(w, u)$, statistically validated via sign-flip permutation test ($p < 0.01$).
2. **Causal State Surgery & Swap Effect ($\Delta_{\text{swap}}$)**:
   Transplanting the hidden state of an independent donor stream (e.g. reversed path with identical action vocabulary and separate jitter) into the organism must causally flip the endpoint query decision ($m_{\text{donor}} < 0$, $\Delta_{\text{swap}} = m_{\text{intact}} - m_{\text{donor}} > 0$).
3. **Transposition Reversals**:
   A trajectory with reversed transitions must produce negative directional margin ($m_{\text{trans}} < 0$).
4. **Deranged Shuffle Superiority**:
   The intact contiguous sequence must produce a higher directional score than permuted orderings (e.g. $[e_2, e_3, e_1]$).
5. **Sensor Competence Preservation**:
   Readout of 1-hop sensor probes must maintain $\ge 90\%$ accuracy across developmental stages.
6. **Task-Aligned Jacobian Sensitivities ($S_{\text{early}}, S_{\text{late}}$)**:
   $$S_{\text{early}} = \left\| \frac{\partial m_k}{\partial x_1} \right\|_2 = \left\| \frac{\partial m_k}{\partial z_k} \cdot \frac{\partial z_k}{\partial x_1} \right\|_2, \quad S_{\text{late}} = \left\| \frac{\partial m_k}{\partial x_k} \right\|_2$$

---

## 4. Lineage Progression: Q16 $\to$ Q17A–E

- **Q16 (Plastic Self-Model Foundations)**: Proved dual-locus neuromodulated plasticity sustains localized self-representations under environmental perturbation.
- **Q17A (Neural Composition Kernel)**: Established that a learned composition kernel can perform 2-hop composition when given adjacent relation vectors.
- **Q17B (Endogenous Induction)**: Showed that trajectory-derived supervision induces composition-capable operators without explicit 2-hop reachability labels.
- **Q17C (2-Hop Developmental Organism — PROMOTED)**:
  - 120-epoch meta-trained Simple RNN on 2-step sequences ($u \to v \to w$).
  - Achieved $16/16$ $k=2$ composition with positive causal state surgery.
- **Q17D (Depth Horizon Generalization — PROMOTED MIXED/BOUNDED)**:
  - Tested zero-shot generalization across horizons $k=2, 3, 4, 5$.
  - Demonstrated horizon boundary at $k=3$ ($6/16$ direction, $0/16$ state surgery) under exact 120-epoch training baseline commit `fa7ebb8`.
- **Q17E Scout Lineage (Mechanism Discovery)**:
  - **Scout A/B**: Proved geometric Jacobian decay ($\sim 42\%$ loss per step).
  - **Scout 2 ($\alpha$-preactivation residual)**: Falsified static identity carry.
  - **Scout C/D (Adaptive Update Gating)**: Falsified UGRNN gating and raw history reconstruction.
  - **Scout E (Shared Relational-Prefix Supervision)**: Proved shared prefix supervision on monolithic RNN causes recency dominance ($S_{\text{late}} \approx 11.4$).
  - **Scout F (Typed Relational State)**: Separated local edge $e_t \in \mathbb{R}^{32}$ from relational accumulator $m_t \in \mathbb{R}^{96}$; revealed two-layer $\tanh$ trainability barrier.
  - **Scout G (Typed Residual Accumulator — BREAKTHROUGH)**: Linear edge encoder + Additive residual accumulator ($m_{t+1} = m_t + \tanh(W_m m_t + W_c e_t + b_m)$) achieved $16/16$ $k=2$ and **$15/16$ ($93.8\%$, $p=0.0002$) zero-shot $k=3$ composition** with complete symmetric double dissociation ($+30.27$ vs $-1.14$).

---

## 5. Falsified Hypotheses / DO NOT REOPEN
1. **Convex Gated Carry ($z_{t+1} = g z_t + (1-g)\tilde{z}$)**: Suppresses incoming evidence under 2-step training; fails to open.
2. **Static Identity Carry ($z_{t+1} = \alpha z_t + \tilde{z}$)**: Preserves Jacobian norm without preserving task-relevant relational subspace.
3. **Raw History Reconstruction ($\|D z_2 - x_1\|^2$)**: Preserves coordinate features, but fails to organize relational composition.
4. **Shared Prefix Supervision on Monolithic RNN**: Perfects $k=2$ but induces recency dominance ($S_{\text{late}}$ doubles), collapsing $k=3$ to $0/16$.
5. **Two-Layer Saturated $\tanh$ Composition without Identity Path**: Suffers severe gradient attenuation during early developmental training.

---

## 6. Canonical Repositories & Evidence Paths
- **Durable Checkpoints**: `research/checkpoints/`
- **Research Contracts**: `research/contracts/`
- **Promotion Records**: `research/promotions/`
- **Raw Telemetry**: `crates/continuity_garden_core/data/`
- **Core Implementation**: `crates/continuity_garden_core/src/`
