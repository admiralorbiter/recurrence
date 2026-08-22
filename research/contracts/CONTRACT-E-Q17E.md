---
contract_id: CONTRACT-E-Q17E
status: DRAFT
proposed_by: antigravity
design_review: null
reviewed_by: chatgpt-pro
authorized_by: null
base_sha: 1b69bba4e73bb93be589c36bbd922ec9e0cb7031
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
In `CHECKPOINT-E-Q17D` (corrected under the frozen 120-epoch Q17C training baseline), Recurrence established the following empirical profile:
- Global validity ($16/16$) and coordinate-OOD controls $C_3, C_4, C_5$ ($16/16$) pass completely.
- Endpoint directional margins extrapolate positively beyond 2-step training ($12/16$ at $k=4, p = 6.88 \times 10^{-3}$), but fail causal state-surgery transfer ($0/16$), reversal collapse ($6/16$ at $k=3$), and temporal shuffle superiority ($12/16$).

Diagnostic Scout `Q17D-B` isolated the underlying geometric mechanics under 120 training epochs:
- Zero static bias in the readout ($\equiv 0.0000$).
- Low saturation ($0.4\%$) with states operating in the responsive moderate-curvature regime ($1 - z_t^2 \approx 0.766$).
- Compound Jacobian sensitivity decay $\left\|\frac{\partial z_k}{\partial x_1}\right\|_F$ losing $\approx 37.5\%$ per recurrent step ($100\% \to 62.5\% \to 32.7\% \to 18.6\%$).
- Readout forward vs reverse alignment difference is only $\Delta \cos \approx 0.114$, explaining why fine-grained causal state surgery and temporal reversal fail despite positive endpoint score margins.

### Core Research Hypothesis
Residual carry introduces an explicit identity-mediated path $\lambda \mathbf{I}$ into the local state transition Jacobian:
$$\tilde{z}_{t+1} = \tanh(W_z z_t + W_x x_t + b_z)$$
$$z_{t+1} = \lambda z_t + (1 - \lambda)\tilde{z}_{t+1}$$
$$\mathbf{A}_t = \frac{\partial z_{t+1}}{\partial z_t} = \lambda \mathbf{I} + (1 - \lambda) D_{t+1} W_z$$

The empirical hypothesis is:
> By providing an identity carry path, residual carry enhances directional and temporal specificity (donor swap separation $\Delta_{\text{swap}}$ and deranged shuffle discrimination) without requiring multi-hop training or sacrificing new-edge plasticity.

Developmental parameter scouting across $\lambda \in [0.00..0.50]$ under exact 120-epoch training identified **$\lambda^* = 0.04$** as the optimal carry strength:
- Increases mean donor-state swap separation $\Delta_{\text{swap}}$ from $-0.023$ ($\lambda=0.0$) to $+1.370$ ($\lambda=0.04$).
- Increases deranged shuffle superiority from $12/16$ to $16/16$.
- Preserves $>92\%$ of baseline last-edge plasticity ($\|d(z_3)/d(x_3)\|_F = 2.657$ vs $2.868$ baseline) and maintains high $k=2$ retention.

---

## 2. Experimental Design: Strict Paired Comparison

The confirmatory experiment evaluates 16 fresh seeds on sealed test worlds in a strict paired configuration:

$$\text{For each seed } i \in \{1 \dots 16\}:$$
$$\text{Baseline Organism } (\lambda = 0.0) \quad \text{vs} \quad \text{Residual Organism } (\lambda = 0.04)$$

### Paired Parameter Provenance
For each seed $i$:
1. `initial_hash_i`: Cryptographic SHA-256 digest of pre-training initialization $\theta_{i, 0}$ (strictly identical across both arms).
2. `posttrain_hash_i_lambda0`: Immutable SHA-256 digest of post-120-epoch trained weights under $\lambda = 0.0$.
3. `posttrain_hash_i_residual`: Immutable SHA-256 digest of post-120-epoch trained weights under $\lambda = 0.04$.
4. Identical 2-step auxiliary training streams ($120$ epochs, $64$ batches/epoch, $\text{lr} = 0.030$).
5. Identical frozen readout architecture $r_\theta(z_t, q)$.
6. Identical sealed evaluation trajectories for $k=2, k=3, k=4$.

---

## 3. Independent Cloned-Twin Donor Transplant & Acceptance Gates

| Gate / Estimand | Metric / Definition | Verification Method | Preregistered Floor |
| :--- | :--- | :--- | :--- |
| **Gate V1: Parameter Provenance & 120-Epoch Training Verification** | `initial_hash_i` match across arms; valid 64-char `posttrain_hash` per arm; epochs $=120$, lr $=0.030$, batches $=64$ | Cryptographic byte check | $\mathbf{16 / 16}$ verified ($100.0\%$) |
| **Gate V2: Baseline 2-Hop Retention ($k=2$)** | Directional margin $m_2 > 0$ under both $\lambda=0.0$ and $\lambda=0.04$ | Exact binomial per arm | $\ge \mathbf{14 / 16}$ in **both** arms |
| **Gate V3: Contemporaneous Sensor Competence** | 20-trial 1-hop sensor classification accuracy under both arms | vs Gold labels | $\ge \mathbf{90.0\%}$ in $16/16$ in **both** arms |
| **Gate V4: Structural Zero-Sidecar Invariant** | External transition ledger reads $\equiv 0$ | Direct API assertion | $\equiv \mathbf{100.0\%}$ |
| **Gate V5: Last-Edge Plasticity Retention Floor** | Last-step sensitivity $\|d(z_3)/d(x_3)\|_F$ under $\lambda=0.04$ | Analytical Jacobian | $\ge \mathbf{2.20}$ in $16/16$ seeds |
| **Gate 1: Depth-3 Directional Margin** | Directional margin $m_3 > 0$ under $\lambda=0.04$ | Paired sign-flip test | $\ge \mathbf{12 / 16}$ seeds, $p < \mathbf{0.01}$ |
| **Gate 2: Independent Cloned-Twin State Surgery** | $m_3(z_3^{\text{twin1}}) > 0 \land m_3(z_3^{\text{donor}}) < 0$ under independent twin donor transplant | Cloned twin replay | $\ge \mathbf{12 / 16}$ seeds ($75.0\%$) |
| **Gate 3: Independent Transposition Reversal** | Reversal collapse $m_{3, \text{trans}} < 0$ on $D \to C \to B \to A$ | Transposed stream | $\ge \mathbf{14 / 16}$ seeds ($87.5\%$) |
| **Gate 4: Deranged Shuffle Superiority** | Intact margin exceeds deranged shuffle $[e_2, e_3, e_1]$ | Paired margin comparison | $\ge \mathbf{14 / 16}$ seeds ($87.5\%$) |
| **Gate 5: Paired Continuous Swap Advantage** | $\Delta_{\text{paired}} = \Delta_{\text{swap}}(\lambda=0.04) - \Delta_{\text{swap}}(\lambda=0.0) > 0$ | Exact paired sign-flip test | Mean $\Delta_{\text{paired}} > 0, p < \mathbf{0.01}$ |

---

## 4. Epistemic Scope Ceilings
- **Claim Ceiling**: Claims that an explicit temporal residual carry path introduces directional and temporal specificity into recurrent hidden states, restoring causally grounded 3-hop composition without multi-hop training.
- **Exclusions**:
  - Does NOT claim inflation of scalar Jacobian Frobenius norms.
  - Does NOT claim general $N$-hop reasoning for $N \ge 5$.
  - Does NOT claim optimal architecture over gated (GRU/LSTM) or structured state-space models.
