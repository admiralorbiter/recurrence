---
contract_id: CONTRACT-F-F1
status: DRAFT
base_sha: 752a47f4413762cab4a2f041e19ff75dd0f5e825
created_at: "2026-08-22 13:15:00Z"
revised_at: "2026-08-22 18:43:00Z"
revision_note: "Revised per independent Codex CLI audit (FIX verdict). Operationally sealed all metrics, added missing controls, strengthened negative control stratification, narrowed epistemic language, rephrased sealing as pre-freeze requirements."
proposed_by: antigravity
design_review: CHANGES_REQUESTED
reviewed_by: codex-cli
authorized_by: null
resource_class: cpu
long_running: false
exclusive_gpu: false
interruptible: true
---

# Research Contract: CONTRACT-F-F1 (Endogenous Latent Self-State Factorization: Partitioned Recurrent Dynamics & Counterfactual Interventional Probing)

**Lifecycle Status**: `DRAFT` (Codex Audit: CHANGES_REQUESTED → Revisions Applied; Awaiting Human Strategic Authorization)

---

## 1. Executive Summary & Epistemic Motivation

Gate E (Contracts `CONTRACT-E-Q17A` through `CONTRACT-E-Q17E`) certified that relational composition is an algebraic closure property, proving that two-step developmental evidence reliably selects canonical tensor contractions $O_1(R \cdot E)$ that generalize to multi-hop reachability at $k=3$.

With external relational graph composition closed, Continuity Garden (`recurrence`) now opens its central research horizon:

$$\textbf{Developmental Evidence for Action-Conditioned Latent Self-State Factorization}$$

### The Research Question
Under self-supervised recurrent temporal prediction with motor commitment signals, does a developmental training regime produce a latent state partition where one subspace correlates specifically with future action predictions while the complementary subspace tracks external environmental dynamics?

This is a **weaker, empirically testable** claim than "genuine self-model possession." The contract tests whether a specific architectural and training configuration produces evidence *consistent with* an action-conditioned latent self-state factorization — not whether the system possesses phenomenal self-awareness or cognitive self-models in the philosophical sense.

$$\text{Observation: } x_t \in \mathbb{R}^4, \quad \text{Latent State: } z_t = \left[ z_t^{\text{self}}, \, z_t^{\text{env}} \right] \in \mathbb{R}^8 \quad (d_{\text{self}} = 4, \, d_{\text{env}} = 4)$$

---

## 2. Mathematical Formalism & Interventional Probes

### A. The Agent State & Dynamic Transition
At time $t$, the recurrent agent receives observation $x_t \in \mathbb{R}^4$ (environmental state + action commitment) and updates its partitioned recurrent state $z_t \in \mathbb{R}^8$:
$$z_t = \Phi_\theta(z_{t-1}, x_t) = \left[ z_t^{\text{self}}, \, z_t^{\text{env}} \right]$$

### B. Interventional Causal Probing

#### B.1 Self-Intervention Assay ($\text{do}(z^{\text{self}})$)
We intervene directly on the self-subspace with a fixed perturbation:
$$\tilde{z}_t = \left[ z_t^{\text{self}} + \delta, \, z_t^{\text{env}} \right], \quad \delta \sim \mathcal{N}(0, 0.50^2 \cdot I_{d_{\text{self}}})$$

We measure two metrics over a forward rollout of $H=5$ steps:

**$\Delta_{\text{action}}$** (Action Prediction Shift):
$$\Delta_{\text{action}} = \frac{1}{H} \sum_{h=1}^{H} \| \hat{a}_{t+h}(\tilde{z}) - \hat{a}_{t+h}(z) \|_2$$
where $\hat{a}_{t+h}$ is the action prediction from a frozen linear probe $W_a \in \mathbb{R}^{d_{\text{action}} \times d_{\text{self}}}$ trained on held-out trajectories (80/20 train/test split by seed).

**$\Delta_{\text{env}}$** (Environmental Invariance):
$$\Delta_{\text{env}} = \frac{1}{H} \sum_{h=1}^{H} \| \hat{x}_{t+h}^{\text{env}}(\tilde{z}) - \hat{x}_{t+h}^{\text{env}}(z) \|_2$$
where $\hat{x}^{\text{env}}$ is the environment reconstruction from a frozen linear probe $W_e \in \mathbb{R}^{d_{\text{obs}} \times d_{\text{env}}}$.

**Causal Dissociation Criterion**: $\Delta_{\text{action}} \ge +0.50$ AND $\Delta_{\text{env}} \le 0.15$.

#### B.2 Environmental Noise Invariance Assay ($\xi_{\text{env}}$)
Independent Gaussian jitter on the observation channel:
$$\tilde{x}_t = x_t + \xi, \quad \xi \sim \mathcal{N}(0, 0.20^2 \cdot I_{d_{\text{obs}}})$$

**Self-Stability Criterion**: $\text{Margin}_{\text{self}}(\tilde{x}) \ge 0.95 \times \text{Margin}_{\text{self}}(x)$.

#### B.3 Action-Channel Ablation (New: Codex Audit Requirement)
Zero out action commitment signals in the observation before encoding:
$$x_t^{\text{no-action}} = [x_t^{\text{env-only}}, \, \mathbf{0}]$$

**Ablation Criterion**: Without action commitment, the self-subspace must lose predictive power: $\text{Margin}_{\text{self}}(\text{no-action}) < 0.40$ for $\ge 12/16$ seeds. This confirms the self-subspace depends on motor commitment, not passive observation buffering.

#### B.4 Environment-Only Intervention (New: Codex Audit Requirement)
Intervene on $z^{\text{env}}$ while holding $z^{\text{self}}$ fixed:
$$\tilde{z}_t^{\text{env-only}} = \left[ z_t^{\text{self}}, \, z_t^{\text{env}} + \delta_{\text{env}} \right], \quad \delta_{\text{env}} \sim \mathcal{N}(0, 0.50^2 \cdot I_{d_{\text{env}}})$$

**Symmetric Dissociation Criterion**: $\Delta_{\text{env}}^{\text{env-probe}} \ge +0.50$ AND $\Delta_{\text{action}}^{\text{env-probe}} \le 0.15$ for $\ge 14/16$ seeds. This confirms the partition is bidirectionally clean, not just one-directional.

#### B.5 Negative Control (Scrambled Self-State)
Randomly shuffle $z_t^{\text{self}}$ across the batch dimension using **cross-trajectory, cross-time, cross-seed stratified permutation**: each shuffled $z_t^{\text{self}}$ is drawn from a different trajectory AND different timestep AND different seed than its paired $z_t^{\text{env}}$. This prevents information leakage from shared task phase, action distribution, or environment structure.

**Falsification Criterion**: $\text{Margin}_{\text{self}} < +0.60$ for shuffled inputs, with $\le 2/16$ seeds passing Gate 3.

### C. Operationally Sealed Metric Definitions

**$\text{Margin}_{\text{self}}$** (Forward Self-Trajectory Prediction Margin):
$$\text{Margin}_{\text{self}} = \text{Acc}_{\text{self-probe}}(z^{\text{self}} \to a_{t+1:t+H}) - \text{Acc}_{\text{chance}}$$

where $\text{Acc}_{\text{self-probe}}$ is the top-1 classification accuracy of a frozen single-layer linear probe ($W_{\text{self}} \in \mathbb{R}^{|\mathcal{A}| \times d_{\text{self}}}$, no bias, no nonlinearity) predicting the next $H=5$ discrete actions from $z_t^{\text{self}}$. $\text{Acc}_{\text{chance}} = 1/|\mathcal{A}|$.

**Probe Protocol**: Train linear probes on 80% of trajectories (by seed), evaluate on held-out 20%. Each seed generates $T=200$ timesteps. Probes are frozen after training and reused across all interventional assays.

**Subspace Orthogonality**: Cosine similarity between principal subspace bases $W_{\text{self}}$ and $W_{\text{env}}$:
$$\text{sim}(W_{\text{self}}, W_{\text{env}}) = \frac{|\text{tr}(W_{\text{self}}^T W_{\text{env}})|}{\|W_{\text{self}}\|_F \|W_{\text{env}}\|_F}$$

---

## 3. Statistical Acceptance Gates ($N=16$ Independent Seeds)

All evaluations execute across $N=16$ independent random seeds ($\text{MasterSeed}(i) = 222000 + 777 \times i$).

| Gate ID | Target Metric & Statistical Boundary | Preregistered Floor | Evaluation Method |
| :--- | :--- | :--- | :--- |
| **Gate 1** | Baseline Environment Reconstruction | $\ge 15/16$ ($93.8\%$) seeds with $R^2 \ge 0.90$ | Linear probe on $z_t^{\text{env}} \to x_{t+1}^{\text{env}}$ |
| **Gate 2** | Subspace Orthogonality | $\ge 14/16$ ($87.5\%$) seeds with $\text{sim}(W_{\text{self}}, W_{\text{env}}) \le 0.10$ | Cosine similarity of principal subspace bases |
| **Gate 3** | Forward Self-Trajectory Prediction Margin | $\ge 14/16$ ($87.5\%$) seeds with $\text{Margin}_{\text{self}} \ge +0.60$ | Frozen linear probe accuracy minus chance |
| **Gate 4** | Self-Intervention Causal Dissociation | $\ge 14/16$ ($87.5\%$) seeds ($\Delta_{\text{action}} \ge +0.50 \land \Delta_{\text{env}} \le 0.15$) | $\text{do}(z^{\text{self}})$ perturbation assay |
| **Gate 5** | Environmental Noise Invariance | $\ge 14/16$ ($87.5\%$) seeds with $\ge 95\%$ baseline margin retained | Observation jitter ($\sigma = 0.20$) |
| **Gate 6** | Scrambled Self-State Falsification (Negative Control) | $\le 2/16$ ($12.5\%$) seeds pass Gate 3 | Cross-trajectory/time/seed stratified shuffling |
| **Gate 7** | Action-Channel Ablation (New) | $\ge 12/16$ ($75.0\%$) seeds with $\text{Margin}_{\text{self}} < 0.40$ | Zero-action observation ablation |
| **Gate 8** | Symmetric Env-Intervention Dissociation (New) | $\ge 14/16$ ($87.5\%$) seeds ($\Delta_{\text{env}}^{\text{env}} \ge +0.50 \land \Delta_{\text{action}}^{\text{env}} \le 0.15$) | $\text{do}(z^{\text{env}})$ perturbation assay |

---

## 4. Sealing & Execution Protocol (Pre-Freeze Requirements)

The following assets **must be created and sealed before** transitioning this contract from `DRAFT` to `FROZEN`:

- **Test Harness** (to be created at `crates/continuity_garden_core/tests/confirmatory_f1.rs` or equivalent)
- **Sealing Manifest** (to be created at `research/contracts/SEALING_MANIFEST-F-F1.json`): SHA-256 hashes of contract, test harness, configuration, and seed schedule
- **Verification Entrypoint**: `cargo test -p continuity_garden_core --test confirmatory_f1`
- **Freeze Condition**: All sealing manifest hashes must match the exact frozen contract and assets at `execution_base_sha`

---

## 5. Scientific Claim Ceiling

### Certified Finding Authorized Upon Promotion
> Under a specific developmental training regime with action-conditioned recurrent dynamics, the trained model produces a partitioned latent state ($z = [z^{\text{self}}, z^{\text{env}}] \in \mathbb{R}^8$) where one subspace correlates with shifts in future action predictions under direct perturbation ($\Delta_{\text{action}} \ge +0.50$) without disrupting environmental state tracking ($\Delta_{\text{env}} \le 0.15$), and where removing action commitment signals from observations collapses self-subspace predictive power. This constitutes evidence consistent with an action-conditioned latent self-state factorization under the tested conditions.

### Explicit Exclusions
This contract does **NOT** authorize claims of:
- Conscious phenomenal awareness or subjective experience
- Genuine self-model possession in the philosophical or cognitive science sense
- Open-world embodied navigation or real-world robotic self-modeling
- Social theory-of-mind or other-agent modeling
- Generalization beyond the specific architecture, training regime, and observation dimensionality tested
- Causal sufficiency (the probes test correlational dissociation under intervention, not full causal identification)
- Model-independent conclusions (results are specific to the tested recurrent architecture)
