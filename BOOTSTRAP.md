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

## 3. PROMOTED / ESTABLISHED (Durable Scientific Baseline)

- **Q16 (Plastic Self-Model Foundations)**: Dual-locus neuromodulated plasticity sustains localized self-representations under environmental perturbation.
- **Q17A (Neural Composition Kernel)**: A learned composition kernel can compose 2-hop relations when provided adjacent relation vectors.
- **Q17B (Endogenous Induction)**: Trajectory-derived supervision induces composition-capable operators without explicit multi-hop topological labels.
- **Q17C (2-Hop Developmental Organism — PROMOTED)**:
  - 120-epoch meta-trained Simple RNN on 2-step sequences ($u \to v \to w$).
  - Achieved $16/16$ $k=2$ composition with positive causal state surgery.
- **Q17D (Depth Horizon Generalization — PROMOTED MIXED/ANOMALOUS)**:
  - Evaluated zero-shot generalization across horizons $k=2, 3, 4, 5$ on candidate commit `fa7ebb8`.
  - Established that standard Simple RNN exhibits mixed/anomalous behavior beyond trained depth ($k=3$: $13/16$ positive direction, but failed sign-flip significance $p=0.029$, state surgery only $1/16$, shuffle superiority only $7/16$).

---

## 4. CANDIDATE (Executed Confirmatory Benchmark Awaiting Promotion)
*(No candidate benchmark currently active. Q17D is promoted; Q17E is in pre-contract scout phase).*

---

## 5. SCOUT / DEVELOPMENT EVIDENCE (Exploratory / Cannot Become Premise)

- **Scout A/B**: Quantified geometric state-Jacobian sensitivity decay ($\sim 42\%$ loss per recurrent step) in standard RNNs.
- **Scout 2 ($\alpha$-preactivation residual)**: Proved that preserving 100% Jacobian norm via identity carry does not preserve task-relevant relational subspaces ($0/16$ direction).
- **Scout C/D (Gating & Reconstruction)**: Showed that UGRNN convex gating and raw coordinate reconstruction fail to induce recursive composition.
- **Scout E (Shared Prefix Supervision)**: Forced shared relational supervision across steps; revealed that monolithic RNN states collapse into a "recency attractor" ($S_{\text{late}} \approx 11.4$, $k=3 \to 0/16$).
- **Scout F (Typed Relational State)**: Separated local edge $e \in \mathbb{R}^{32}$ from relational accumulator $m \in \mathbb{R}^{96}$; identified double-$\tanh$ saturation bottleneck.
- **Scout G (Additive Residual Accumulator)**: Linear edge encoder + Additive residual accumulator ($\eta=1.00$, $m \in \mathbb{R}^{128}$) restored $k=2$ to $16/16$, achieved $15/16$ $k=3$ direction ($p=0.0002$), and showed strong history swap drop ($+30.27$).
- **Scout H (Final-Edge Necessity & Compositional Binding Assay)**:
  - Discovered that Scout G's $m_2$ state predicts $A \to D$ reachability before observing edge 3 ($+3.96$ pre-edge margin).
  - Swapping wrong source edge $X \to D$ where $X \ne C$ produces $+2.00$ margin (no drop), proving that the model exhibits an **unbound reachability shortcut** rather than verified variable-binding composition $(A \to C) \circ (C \to D)$.

---

## 6. FALSIFIED BY PROMOTED EVIDENCE
1. **Monolithic Simple RNN Horizon Scalability**: Monolithic Simple RNNs trained on $k=2$ do not zero-shot scale to $k=3+$ causal composition.

---

## 7. DEVELOPMENT LINES CURRENTLY DEPRIORITIZED (Scout Evidence)
1. **Convex Update Gating ($z_{t+1} = g z_t + (1-g)\tilde{z}$)**: Suppresses incoming evidence under short-horizon training.
2. **Static Preactivation Residuals ($z_{t+1} = \alpha z_t + \tilde{z}$)**: Preserves scalar norm without task-relevant geometry.
3. **Coordinate Reconstruction Loss ($\|Dz - x\|^2$)**: Improves raw feature retention but fails to organize relational operators.
4. **Monolithic Shared-Prefix RNNs**: Induces recency dominance.

---

## 8. Canonical Repositories & Evidence Paths
- **Durable Checkpoints**: `research/checkpoints/`
- **Research Contracts**: `research/contracts/`
- **Promotion Records**: `research/promotions/`
- **Raw Telemetry**: `crates/continuity_garden_core/data/`
- **Core Implementation**: `crates/continuity_garden_core/src/`
