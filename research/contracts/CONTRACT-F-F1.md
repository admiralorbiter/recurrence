---
contract_id: CONTRACT-F-F1
status: DRAFT
base_sha: 752a47f4413762cab4a2f041e19ff75dd0f5e825
created_at: "2026-08-22 13:12:00Z"
proposed_by: antigravity
design_review: PENDING
reviewed_by: chatgpt-pro
authorized_by: null
resource_class: cpu
long_running: false
exclusive_gpu: false
interruptible: true
---

# Research Contract: CONTRACT-F-F1 (Endogenous Causal Self-Modeling: Latent State Partitioning & Counterfactual Interventional Probing)

**Lifecycle Status**: `DRAFT` (Awaiting Scientific Review Desk Audit & Human Strategic Authorization)

---

## 1. Executive Summary & Epistemic Motivation

Gate E (Contracts `CONTRACT-E-Q17A` through `CONTRACT-E-Q17E`) certified that relational composition is an algebraic closure property, proving that two-step developmental evidence reliably selects canonical tensor contractions $O_1(R \cdot E)$ that generalize to multi-hop reachability at $k=3$.

With external relational graph composition closed, Continuity Garden (`recurrence`) now opens its central moonshot horizon:
$$\textbf{The Causal Developmental Atlas of an Artificial Self-Model}$$

### The Theoretical Problem: What is a "Self-Model"?
In cognitive science and robotics, a system possesses a genuine *self-model* (as opposed to an ungrounded state buffer) if and only if it endogenously maintains a structured internal representation of its own agency, causal commitments, and counterfactual future trajectories that is causally dissociated from external environmental state dynamics.

$$\text{Latent State: } z_t = \left[ z_t^{\text{self}}, \, z_t^{\text{env}} \right] \in \mathbb{R}^{d_{\text{self}} + d_{\text{env}}}$$

**CONTRACT-F-F1** formalizes the first developmental experiment to test whether self-supervised temporal prediction under motor commitments spontaneously induces an orthogonalized self-vs-environment state factorization.

---

## 2. Mathematical Formalism & Interventional Probes

### A. The Agent State & Dynamic Transition
At time $t$, the recurrent agent receives observation $x_t \in \mathbb{R}^4$ (environmental state + action commitment) and updates its recurrent state $z_t$:
$$z_t = \Phi_\theta(z_{t-1}, x_t)$$

### B. Interventional Causal Probing (The Pearl do-Calculus Invariant)
To prevent trivial shortcuts (e.g. passive observation buffering), we subject $z_t$ to two orthogonal interventional assays:

1. **Self-Intervention Assay ($\text{do}(z^{\text{self}})$)**:
   We intervene directly on the self-subspace:
   $$\tilde{z}_t = \left[ z_t^{\text{self}} + \delta, \, z_t^{\text{env}} \right]$$
   We measure the forward counterfactual action prediction shift ($\Delta_{\text{action}}$) and environmental invariant shift ($\Delta_{\text{env}}$).
   **Causal Dissociation Invariant**: $\Delta_{\text{action}} \ge +0.50$ while $\Delta_{\text{env}} \le 0.15$.

2. **Environmental Noise Invariance Assay ($\xi_{\text{env}}$)**:
   We subject the environmental observation channel to independent Gaussian jitter $\xi \sim \mathcal{N}(0, \sigma^2)$.
   **Self-Stability Invariant**: The self-model's internal trajectory coherence must not degrade by more than $5\%$ under environmental sensory perturbation.

3. **Negative Control (Scrambled Self-State)**:
   Randomly shuffling $z_t^{\text{self}}$ across the batch dimension must collapse self-trajectory prediction to chance ($\le 2/16$ pass rate).

---

## 3. Statistical Acceptance Gates ($N=16$ Independent Seeds)

All evaluations execute across $N=16$ independent random seeds ($\text{MasterSeed}(i) = 222000 + 777 \times i$).

| Gate ID | Target Metric & Statistical Boundary | Preregistered Floor | Evaluation Method |
| :--- | :--- | :--- | :--- |
| **Gate 1** | Baseline Environment Reconstruction | $\ge 15/16$ ($93.8\%$) seeds with $R^2 \ge 0.90$ | Linear probe on $z_t^{\text{env}} \to x_{t+1}^{\text{env}}$ |
| **Gate 2** | Endogenous Self/Env Subspace Orthogonality | $\ge 14/16$ ($87.5\%$) seeds with $\text{sim}(W_{\text{self}}, W_{\text{env}}) \le 0.10$ | Cosine similarity between principal subspace bases |
| **Gate 3** | Forward Self-Trajectory Prediction Margin | $\ge 14/16$ ($87.5\%$) seeds with $\text{Margin}_{\text{self}} \ge +0.60$ | Internal action sequence prediction from $z_t^{\text{self}}$ |
| **Gate 4** | Causal Interventional Specificity | $\ge 14/16$ ($87.5\%$) seeds ($\Delta_{\text{self}} \ge +0.50 \land \Delta_{\text{env}} \le 0.15$) | Pearl $\text{do}(z^{\text{self}})$ causal perturbation assay |
| **Gate 5** | Environmental Sensory Noise Invariance | $\ge 14/16$ ($87.5\%$) seeds retain $\text{Margin}_{\text{self}} \ge 95\%$ baseline | Robustness under $\sigma = 0.20$ observation noise |
| **Gate 6 (Negative Control)** | Scrambled Self-State Falsification | $\le 2/16$ ($12.5\%$) seeds pass | Batch-shuffled $z_t^{\text{self}}$ ablation assay |

---

## 4. Scientific Claim Ceiling

### Certified Finding Authorized Upon Promotion
> Under self-supervised recurrent dynamics, developmental exposure to action-conditioned transitions spontaneously induces an orthogonal self-vs-environment state partition ($z = [z^{\text{self}}, z^{\text{env}}]$) where internal self-representations causally steer future action predictions without corrupting external sensory environment tracking.

### Explicit Exclusions
This contract does **NOT** authorize claims of conscious phenomenal awareness, open-world embodied navigation, or social theory-of-mind.
