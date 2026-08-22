---
scout_id: SCOUT-E-Q17D-B
status: DRAFT
proposed_by: antigravity
reviewed_by: chatgpt-pro
authorized_by: null
base_sha: 905a4afc1bbd9c90ebdbf0d1a49df5d8869fc485
resource_class: cpu
long_running: false
exclusive_gpu: false
interruptible: true
---

# Diagnostic Scout Proposal: SCOUT-E-Q17D-B (Multi-Hop Causal Dissociation & Query Geometry Diagnostics)

## Title
Gate E Diagnostic Scout: Geometric and Attenuation Decomposition of Multi-Hop Extrapolation Dissociation

## 1. Context & Research Problem
In `CHECKPOINT-E-Q17D`, Recurrence established an important scientific finding:
> Under the frozen Q17C architecture, endpoint-directional margins extrapolate positively beyond two-step training ($13/16$ at $k=3$, $15/16$ at $k=4, p \approx 9.2 \times 10^{-5}$), but fail causal state-surgery transfer ($1/16$), reversal collapse ($3/16$), and temporal shuffle controls ($7/16$).

Because 2-hop coordinate controls $C_3, C_4, C_5$ passed with $100\%$ precision ($16/16$), this dissociation is not an out-of-distribution coordinate artifact.

Diagnostic Scout `Q17D-B` isolates the underlying mechanism across four targeted probes:

```
DIAGNOSTIC ARCHITECTURE:
Probe 1: Zero-History Baseline     z_0 = 0 ────────────────────────► Query (A, D/E/F) [Static Geometry Bias]
Probe 2: Unrelated-History Control z_unrelated (X->Y->Z) ──────────► Query (A, D/E/F) [History Specificity]
Probe 3: Initial-Step Jacobian     || d(z_k) / d(x_1) || ──────────► Attenuation vs Depth k in {2,3,4,5}
Probe 4: Readout Decomposition     Score = <W_r (x) z_t, W_q (u,v)> ► Query Alignment vs Recurrent Alignment
```

---

## 2. Experimental Probes & Metrics

### Probe 1: Zero-History / Static Query Bias
- Measure directional score margin $m_{\text{static}}(u, v) = r_\theta(\mathbf{0}, (u, v)) - r_\theta(\mathbf{0}, (v, u))$ for queries $(A, C), (A, D), (A, E), (A, F)$.
- *Hypothesis*: If $m_{\text{static}}(A, D) > 0$ with no history, the associative query projection $W_q$ introduces an inductive directional coordinate bias that persists regardless of recurrent state.

### Probe 2: Unrelated-History Control
- Feed 3 unrelated transitions $[(X, 1, Y), (Y, 2, Z), (Z, 1, W)]$ where $X, Y, Z, W \notin \{A, B, C, D\}$.
- Evaluate query $(A, D)$ under $z_{\text{unrelated}}$.
- *Metric*: $\Delta_{\text{spec}} = m_{\text{intact}}(A, D) - m_{\text{unrelated}}(A, D)$.

### Probe 3: Initial-Step Jacobian Attenuation
- Compute the Frobenius norm of the Jacobian $\mathbf{J}_k = \frac{\partial z_k}{\partial x_1}$ for depths $k \in \{2, 3, 4, 5\}$:
  $$\mathbf{J}_k = \left( \prod_{t=2}^k \text{diag}(1 - z_t^2) W_z \right) \text{diag}(1 - z_1^2) W_x$$
- *Metric*: Attenuation ratio $\rho(k) = \frac{\|\mathbf{J}_k\|_F}{\|\mathbf{J}_2\|_F}$.

### Probe 4: Readout Projection vs Recurrent State Alignment
- Decompose $r_\theta(z_t, q) = \sum_{i=0}^{127} W_r[i] \cdot z_t[i] \cdot e_q[i]$ into cosine similarity $\cos(z_t, W_r \odot e_q)$ vs magnitude $\|z_t\|$.

---

## 3. Deliverables & Claim Ceiling
- **Deliverable**: Lightweight Rust scout binary `src/bin/scout_q17d_b_diagnostics.rs` reporting analytical distributions across all 16 seeds.
- **Claim Ceiling**: Diagnostic only; does not promote a new operational checkpoint. Informs architectural design for next-stage multi-step propagation contracts.
