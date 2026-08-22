# Project Handoff: Recurrence (Continuity Garden)

**Generated**: 2026-08-22 04:46:00Z  
**Project**: `recurrence`  
**generated_from_sha**: `5bc4e78`  
**Last Promoted Checkpoint**: `CHECKPOINT-E-Q17D` (`fa7ebb857187429188e404b9015c7e8a9394602f`)  
**Operational State**: `IDLE` (Scout J-R1 Frozen-State Addressability & Probing completed)

---

## 1. Moonshot
First causal developmental atlas of an artificial self-model.

---

## 2. Epistemic State Classification

### PROMOTED / ESTABLISHED (Durable Baseline)
- **Q17C**: 2-hop developmental baseline ($16/16$, candidate `b0af2e13e4118564c72b0d004b7e2d54170657d2`).
- **Q17D**: Depth horizon boundary at $k=3$ (candidate `fa7ebb857187429188e404b9015c7e8a9394602f`, classified `NON_MONOTONIC_ANOMALOUS_DEPTH_PROFILE`).

### SCOUT / DEVELOPMENT EVIDENCE (Exploratory)
- **Scout J-A (Readout Degeneracy Theorem)**:
  - Analytically proved and verified with $1,000$-trial oracle that the historical linear query head $r(m, (s, d)) = b_r + \frac{s}{5}A(m) + \frac{d}{5}B(m)$ forces $M(m; s, d) = \frac{s-d}{5}[A(m)-B(m)]$, meaning $M(1, 4) = 1.5 M(1, 3)$ identically (`OLD_QUERY_HEAD_PAIR_SPECIFICITY = IMPOSSIBLE`).
- **Scout J-R1 (Frozen-State Relational Addressability & Causal Diagnostic)**:
  - **Exact State Probing**: Linear probes on exact frozen Scout-I $m_2$ states decode terminal node $w$ at **$82.2\%$** (chance = $16.7\%$, up to $98\%$), intermediate node $v$ at $69.8\%$, and origin $u$ at $61.1\%$.
  - **$k=2$ Endpoint Addressability**: Pair-specific diagnostic decoder $W_{\text{pair}} \in \mathbb{R}^{36 \times 128}$ on frozen states achieves **$16/16$ ($100.0\%$)** endpoint selectivity (target $+0.98$, reverse $-4.96$, distractors $-1.49$, selectivity margin $+1.04$).
  - **Zero-Shot $k=3$ Battery**: Source grounding passes $16/16$ ($100.0\%$, $X \to D$ drops score from $+0.36 \to -2.07$), but destination grounding fails ($3/16$, $C \to E$ fails to write new target $(A, E)$ at $-1.32$).
  - **Mechanistic Double Dissociation**:
    $$\text{Representation Exists (82.2\%)} \quad \land \quad \text{Reporting Works (16/16 k2)} \quad \land \quad \text{Recursive State Rewriting Lacks Variable Binding}$$

### DEPRIORITIZED DEVELOPMENT LINES (Scout Falsifications)
- Monolithic unseparated RNN / static identity residual carry / auxiliary history reconstruction.
- Small-factor multiplicative query parameterizations without balanced conditioning (J-C prior collapse).

---

## 3. Current Frontier & Next Steps

### Frontier
**Scout K (Keyed / Unification-Aware Recurrent Composition)**:
With addressable reporting certified and state representation established, designing a minimal variable-unification recurrent update operator (e.g. Tensor-Product / NBFNet-style product composition) so that $m_3 = \text{Compose}(m_2, e_3)$ unifies the incoming edge source with the intermediate terminal and binds the new destination.

### First Actions on Resume
1. Review [`BOOTSTRAP.md`](file:///c:/Users/admir/Github/recurrence/BOOTSTRAP.md) and [`HANDOFF.md`](file:///c:/Users/admir/Github/recurrence/HANDOFF.md).
2. Review [`research/reviews/current/REVIEW_PACKET.md`](file:///c:/Users/admir/Github/recurrence/research/reviews/current/REVIEW_PACKET.md).
3. Design Scout K (Variable-Unification Recurrent Update) to resolve the destination-grounding rewrite failure.

---

## 4. Canonical Evidence & Artifacts
- **Scout J-R1 Telemetry**: [`crates/continuity_garden_core/data/q17e_j_r1_frozen_addressability_results.json`](file:///c:/Users/admir/Github/recurrence/crates/continuity_garden_core/data/q17e_j_r1_frozen_addressability_results.json)
- **Scout J Telemetry**: [`crates/continuity_garden_core/data/q17e_j_relational_addressability_results.json`](file:///c:/Users/admir/Github/recurrence/crates/continuity_garden_core/data/q17e_j_relational_addressability_results.json)
- **Scout I Telemetry**: [`crates/continuity_garden_core/data/q17e_i_broken_joins_results.json`](file:///c:/Users/admir/Github/recurrence/crates/continuity_garden_core/data/q17e_i_broken_joins_results.json)
- **Verified Commit**: `5bc4e78`
