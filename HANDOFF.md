# Project Handoff: Recurrence (Continuity Garden)

**Generated**: 2026-08-22 03:50:00Z  
**Project**: `recurrence`  
**generated_from_sha**: `fc8cbd02047ff96d5b058a5e848bc95dd7ff07df`  
**Last Promoted Checkpoint**: `CHECKPOINT-E-Q17D` (`fa7ebb8`)  
**Operational State**: `IDLE` (Scout H completed; diagnoses unbound reachability shortcut)

---

## 1. Moonshot
First causal developmental atlas of an artificial self-model.

---

## 2. Epistemic State Classification

### PROMOTED / ESTABLISHED (Durable Baseline)
- **Q17C**: 2-hop developmental baseline ($16/16$, commit `fa7ebb8`).
- **Q17D**: Depth horizon boundary at $k=3$ (mixed/anomalous depth result at candidate commit `fa7ebb8`).

### SCOUT / DEVELOPMENT EVIDENCE (Exploratory)
- **Scout G**: Linear Edge Encoder + Additive Residual Accumulator ($\eta=1.00$) achieved $16/16$ $k=2$, $15/16$ zero-shot $k=3$ direction ($p=0.0002$), and $+30.27$ historical transplant drop.
- **Scout H (Crucial Diagnostic)**:
  - Interrogated whether incoming edge $C \to D$ is causally bound to history $m_2$ ($A \to B \to C$).
  - Results show pre-edge margin on $m_2$ is already $+3.96$, and giving an unbound transition $X \to D$ ($X \ne C$) scores $+2.00$ (no penalty).
  - **Mechanistic Finding**: Scout G solved directional history persistence, but the query readout operates via an unbound reachability shortcut (associating $A$ with general forward flow and checking if $D$ is the final destination, without enforcing $C == C$ source-destination binding).

### DEPRIORITIZED DEVELOPMENT LINES (Scout Falsifications)
- Convex update gating ($z_{t+1} = g z_t + (1-g)\tilde{z}$).
- Static identity residual carry ($z_{t+1} = \alpha z_t + \tilde{z}$).
- Auxiliary history reconstruction loss ($\|D z_2 - x_1\|^2$).
- Shared prefix supervision on monolithic unseparated RNN.

---

## 3. Current Frontier & Next Steps

### Frontier
Diagnosing how to enforce compositional variable binding (requiring edge $t+1$ source to match edge $t$ destination) so that $m_t$ computes true relational chaining $(A \to C) \circ (C \to D) \implies A \to D$ rather than unbound forward-flow heuristics.

### First Actions on Resume
1. Review [`BOOTSTRAP.md`](file:///c:/Users/admir/Github/recurrence/BOOTSTRAP.md) and [`HANDOFF.md`](file:///c:/Users/admir/Github/recurrence/HANDOFF.md).
2. Consult Scientific Review Desk on variable-binding inductive biases (e.g. key-value relational matching or explicit intermediate node consistency in $\mathcal{C}_\theta$).
3. Draft Scout I to test binding-enforcing mechanisms before formalizing `CONTRACT-E-Q17E`.

---

## 4. Canonical Evidence & Artifacts
- **Scout H Telemetry**: [`crates/continuity_garden_core/data/q17e_h_final_edge_results.json`](file:///c:/Users/admir/Github/recurrence/crates/continuity_garden_core/data/q17e_h_final_edge_results.json)
- **Scout G Telemetry**: [`crates/continuity_garden_core/data/q17e_g_typed_trainability_results.json`](file:///c:/Users/admir/Github/recurrence/crates/continuity_garden_core/data/q17e_g_typed_trainability_results.json)
- **Verified Commit**: `fc8cbd02047ff96d5b058a5e848bc95dd7ff07df`
