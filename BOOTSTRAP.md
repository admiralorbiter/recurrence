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

- **CHECKPOINT-E-Q17A-R1 (Research Checkpoint: Q17A Endogenous 2-Hop Transitive Composition (Promoted))**:
  - **Checkpoint**: [`CHECKPOINT-E-Q17A-R1.md`](research/checkpoints/CHECKPOINT-E-Q17A-R1.md)
  - **Candidate SHA**: `efc2d9941bb546a28fc01ff634211e79070a5bae`
  - **Hypothesis Confirmed**: Endogenous neural composition kernels $f_\theta(e_{AB}, e_{BC}) \to a_{AC}$ generalize to withheld multi-hop causal endpoints without explicit graph traversal.
  - **Empirical Baseline**:
  - **Next Frontier**: Q17B — Self-Supervised Composition Discovery (learning composition without explicit auxiliary two-hop targets).

- **CHECKPOINT-E-Q17B (Checkpoint Record: CHECKPOINT-E-Q17B (Self-Supervised Endogenous Composition))**:
  - **Checkpoint**: [`CHECKPOINT-E-Q17B.md`](research/checkpoints/CHECKPOINT-E-Q17B.md)
  - **Candidate SHA**: `da925179bbe769d9da544239c6db9604fcbad243`
  - **Contract Promoted**: `CONTRACT-E-Q17B`
  - **Promotion Artifact**: [`research/promotions/PROMOTION-CONTRACT-E-Q17B.md`](../promotions/PROMOTION-CONTRACT-E-Q17B.md)
  - **Verified Code Baseline**: `da925179bbe769d9da544239c6db9604fcbad243`
  - **Empirical Confirmation**:

- **CHECKPOINT-E-Q17C (Checkpoint Record: CHECKPOINT-E-Q17C (Endogenous Recurrent Causal History & State Surgery))**:
  - **Checkpoint**: [`CHECKPOINT-E-Q17C.md`](research/checkpoints/CHECKPOINT-E-Q17C.md)
  - **Candidate SHA**: `b0af2e13e4118564c72b0d004b7e2d54170657d2`
  - **Contract Promoted**: `CONTRACT-E-Q17C`
  - **Promotion Artifact**: [`research/promotions/PROMOTION-CONTRACT-E-Q17C.md`](../promotions/PROMOTION-CONTRACT-E-Q17C.md)
  - **Verified Code Baseline (`candidate_sha`)**: `b0af2e13e4118564c72b0d004b7e2d54170657d2`
  - **Empirical Confirmation**:

- **CHECKPOINT-E-Q17D (Checkpoint Record: CHECKPOINT-E-Q17D (Multi-Hop Depth Dissociation & Endpoint Extrapolation))**:
  - **Checkpoint**: [`CHECKPOINT-E-Q17D.md`](research/checkpoints/CHECKPOINT-E-Q17D.md)
  - **Candidate SHA**: `fa7ebb857187429188e404b9015c7e8a9394602f`
  - **Contract Promoted**: `CONTRACT-E-Q17D`
  - **Promotion Artifact**: [`research/promotions/PROMOTION-CONTRACT-E-Q17D.md`](../promotions/PROMOTION-CONTRACT-E-Q17D.md)
  - **Verified Code Baseline (`candidate_sha`)**: `fa7ebb857187429188e404b9015c7e8a9394602f`
  - **Empirical Confirmation**:

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
- **Scout H / H-R1 (Final-Edge Necessity & Exact Replay Assay)**:
  - Exact replay on winning Scout-G organisms confirmed that $m_2$ state predicts $A \to D$ reachability before observing edge 3 ($+32.81$ pre-edge margin).
  - Swapping wrong source edge $X \to D$ ($X \ne C$) produces $+16.81$ margin, and zero edge produces $+48.95$, proving that the model exhibits an **unbound reachability shortcut** (propagation without intermediate variable unification) under arithmetic/fixed curricula.

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
