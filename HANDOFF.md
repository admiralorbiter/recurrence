# Project Handoff: Recurrence (Continuity Garden)

**Generated**: 2026-08-22 03:35:00Z  
**Project**: `recurrence`  
**Verified HEAD**: `87ed71e21b56930fcb7e28c464858b1ee3f8373b` (`main`)  
**Last Promoted Checkpoint**: `CHECKPOINT-E-Q17D` (`fa7ebb8`)  
**Operational State**: `IDLE` (Scout Lineage complete; ready for Q17E contract authoring)

---

## 1. Moonshot
First causal developmental atlas of an artificial self-model.

---

## 2. Current Scientific State

### Promoted Baseline
- **Q17C**: 2-hop developmental baseline ($16/16$, commit `fa7ebb8`).
- **Q17D**: Depth horizon boundary at $k=3$ ($6/16$ direction, $0/16$ surgery, candidate commit `fa7ebb8`).

### Scout Evidence Lineage (Mechanistic Resolution)
- **Scout A–E**: Falsified convex gating, static residual carry, history reconstruction, and monolithic prefix supervision.
- **Scout F**: Discovered that typed state separation ($e_t \in \mathbb{R}^{32}, m_t \in \mathbb{R}^{96}$) functionally decouples local edge and relational memory, but two-layer $\tanh$ limits $k=2$ base trainability ($4/16$).
- **Scout G (BREAKTHROUGH)**:
  - Architecture: Linear Edge Encoder ($e_t = W_e x_t + b_e \in \mathbb{R}^{32}$) + Full Additive Residual Accumulator ($m_{t+1} = m_t + \tanh(W_m m_t + W_c e_t + b_m) \in \mathbb{R}^{128}$).
  - $k=2$ Baseline Retention: **$16/16$ ($100.0\%$)**
  - Zero-Shot $k=3$ Direction: **$15/16$ ($93.8\%$, $p = 0.0002$)**
  - Symmetric Causal Double Dissociation:
    - Relational State Swap $\mathcal{C}_\theta(m_2^{\text{donor}}, e_3^{\text{intact}})$: Mean Swap $= \mathbf{+30.2744}$ ($15/16$ flips)
    - Local Edge Swap $\mathcal{C}_\theta(m_2^{\text{intact}}, e_3^{\text{donor}})$: Mean Swap $= \mathbf{-1.1370}$ ($0/16$ flips)
  - $k=3$ Transposition Reversals: **$16/16$ ($100.0\%$)**
  - $k=3$ Deranged Shuffle Superiority: **$16/16$ ($100.0\%$)**
  - Task-Aligned Sensitivities: $S_{\text{early}} = 60.7049$, $S_{\text{late}} = 7.8081$.

---

## 3. Falsified Lines / DO NOT REOPEN
- Convex update gating ($z_{t+1} = g z_t + (1-g)\tilde{z}$).
- Static identity residual carry ($z_{t+1} = \alpha z_t + \tilde{z}$).
- Auxiliary history reconstruction loss ($\|D z_2 - x_1\|^2$).
- Shared prefix supervision on monolithic unseparated RNN.

---

## 4. Current Frontier & Next Immediate Actions

### Frontier
Drafting formal research contract `CONTRACT-E-Q17E.md` for Confirmatory Benchmark on Typed Relational Additive Residual Accumulators across $k=2, 3, 4, 5$ depth horizons.

### First Actions on Resume
1. Review [`BOOTSTRAP.md`](file:///c:/Users/admir/Github/recurrence/BOOTSTRAP.md) and [`HANDOFF.md`](file:///c:/Users/admir/Github/recurrence/HANDOFF.md).
2. Verify git status and compile test: `cargo test --workspace`.
3. Draft `CONTRACT-E-Q17E.md` embodying the linear edge + additive residual accumulator ($\eta=1.00$) mechanism.
4. Present draft contract to Scientific Review Desk for Design Review.

---

## 5. Canonical Evidence & Artifacts
- **Scout G Telemetry**: [`crates/continuity_garden_core/data/q17e_g_typed_trainability_results.json`](file:///c:/Users/admir/Github/recurrence/crates/continuity_garden_core/data/q17e_g_typed_trainability_results.json)
- **Scout G Binary**: [`crates/continuity_garden_core/src/bin/scout_q17e_g_typed_trainability.rs`](file:///c:/Users/admir/Github/recurrence/crates/continuity_garden_core/src/bin/scout_q17e_g_typed_trainability.rs)
- **Verified Commit**: `87ed71e21b56930fcb7e28c464858b1ee3f8373b`
