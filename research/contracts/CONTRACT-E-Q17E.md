---
contract_id: CONTRACT-E-Q17E
status: DRAFT
proposed_by: antigravity
design_review: null
reviewed_by: chatgpt-pro
authorized_by: null
base_sha: a56b1d8e1e786b24bb43a0d176767512803b9bdf
execution_base_sha: null
resource_class: cpu
long_running: false
exclusive_gpu: false
interruptible: true
---

# Research Contract Proposal: CONTRACT-E-Q17E (Preservation of Causal History Under Recurrent Depth)

## Title
Gate E Confirmatory Frontier: Preservation of Causal History Under Recurrent Depth via Minimal Temporal Residual Carry

## 1. Context & Research Problem
In `CHECKPOINT-E-Q17D`, Recurrence established a key scientific dissociation:
> Endpoint-directional scores extrapolate positively beyond two-step training ($13/16$ at $k=3, 15/16$ at $k=4$), but fail causal state-surgery transfer ($1/16$), temporal reversal controls ($3/16$), and deranged shuffle superiority ($7/16$).

Diagnostic Scout `Q17D-B` isolated the underlying mechanistic cause:
- Zero static bias in the readout ($\equiv 0.0000$).
- Low saturation ($5.3\%$), with states operating in the moderate-curvature regime ($1 - z_t^2 \approx 0.65$).
- Compound Jacobian sensitivity decay $\left\|\frac{\partial z_k}{\partial x_1}\right\|_F$ losing $\approx 40-50\%$ per recurrent step, causing the initial developmental transition $A \to B$ to lose causal leverage over the final recurrent state $z_k$.

### Core Research Hypothesis
Residual carry creates an explicit identity-mediated route by which earlier state can influence later state, reducing dependence on repeated multiplication solely through $D_t W_z$:
$$\tilde{z}_{t+1} = \tanh(W_z z_t + W_x x_t + b_z)$$
$$z_{t+1} = \lambda z_t + (1 - \lambda)\tilde{z}_{t+1}$$
$$\mathbf{A}_t = \frac{\partial z_{t+1}}{\partial z_t} = \lambda \mathbf{I} + (1 - \lambda) D_{t+1} W_z$$

Developmental parameter scouting across $\lambda \in [0.0..0.9]$ identified $\lambda^* = 0.08$ as the optimal carry strength that enhances temporal order and transposition specificity while preserving last-edge plasticity ($\|d(z_3)/d(x_3)\|_F \ge 2.70$) and $100\%$ baseline retention at $k=2$.

---

## 2. Experimental Design: Paired Baseline vs Residual Carry

The confirmatory experiment evaluates 16 fresh seeds on sealed test worlds in a strict paired configuration:

$$\text{For each seed } i \in \{1 \dots 16\}:$$
$$\text{Baseline Organism } (\lambda = 0.0) \quad \text{vs} \quad \text{Residual Organism } (\lambda = 0.08)$$

Both organisms share:
1. Identical parameter initialization seeds.
2. Identical 2-step-only meta-training streams and learning hyperparameters.
3. Identical frozen readout architecture $r_\theta(z_t, q)$.
4. Identical sealed evaluation trajectories for $k=2, k=3, k=4$.

---

## 3. Confirmatory Acceptance Gates

| Gate / Estimand | Metric / Definition | Verification Method | Preregistered Floor |
| :--- | :--- | :--- | :--- |
| **Gate V1: Architecture & Weight Fingerprint** | $d=128, d_x=4, d_q=2$, full 8-tensor SHA-256 parameter hash | Cryptographic byte check | $\mathbf{16 / 16}$ verified ($100.0\%$) |
| **Gate V2: Baseline 2-Hop Retention ($k=2$)** | Directional margin $m_2 > 0$ under $\lambda=0.08$ | Exact binomial | $\ge \mathbf{15 / 16}$ seeds ($93.8\%$) |
| **Gate V3: Contemporaneous Sensor Competence** | 20-trial 1-hop sensor classification accuracy | vs Gold labels | $\ge \mathbf{90.0\%}$ in $16/16$ seeds |
| **Gate V4: Structural Zero-Sidecar Invariant** | External transition ledger reads $\equiv 0$ | Direct API assertion | $\equiv \mathbf{100.0\%}$ |
| **Gate V5: Last-Edge Plasticity Retention** | Last-step sensitivity $\|d(z_3)/d(x_3)\|_F$ | Analytical Jacobian | $\ge \mathbf{2.00}$ in $16/16$ seeds |
| **Gate 1: Depth-3 Positive Generalization** | Directional margin $m_3 > 0$ under $\lambda=0.08$ | Paired sign-flip test | $\ge \mathbf{12 / 16}$ seeds, $p < \mathbf{0.01}$ |
| **Gate 2: Depth-3 Causal State Surgery** | Donor-aligned choice flip under latent state transplant | Twin transplant replay | $\ge \mathbf{12 / 16}$ seeds ($75.0\%$) |
| **Gate 3: Depth-3 Temporal Transposition Reversal** | Reversal collapse $m_{3, \text{trans}} < 0$ | Transposed stream | $\ge \mathbf{14 / 16}$ seeds ($87.5\%$) |
| **Gate 4: Depth-3 Deranged Shuffle Superiority** | Intact margin exceeds deranged shuffle $[e_2, e_3, e_1]$ | Paired margin comparison | $\ge \mathbf{12 / 16}$ seeds ($75.0\%$) |
| **Gate 5: Paired Residual Advantage** | $\Delta_{\text{surgery}} = \text{surgery}(\lambda=0.08) - \text{surgery}(\lambda=0.0)$ | Paired seed difference | $\Delta_{\text{surgery}} \ge \mathbf{+4 / 16}$ seeds |

---

## 4. Epistemic Scope Ceilings
- **Claim Ceiling**: Claims that an explicit temporal residual carry path preserves early developmental causal leverage and restores causally grounded 3-hop composition without multi-hop training.
- **Exclusions**: Does NOT claim full arbitary $N$-hop reasoning ($N \ge 5$). Does NOT claim optimal architecture over gated or HiPPO memory models (reserved for future milestones).
