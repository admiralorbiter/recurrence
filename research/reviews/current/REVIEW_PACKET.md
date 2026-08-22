# Evidence Review Packet: Recurrence (recurrence)
**Active Contract**: `None`
**Repository HEAD**: `49bdc63164be4aa3fc414ff58d6723136adc0ca2`

## 1. Project Moonshot & Conceptual Formalism
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


---

## 2. Latest Promoted Checkpoint
### CHECKPOINT-E-Q17D.md
---
checkpoint_id: CHECKPOINT-E-Q17D
contract_id: CONTRACT-E-Q17D
promotion_id: PROMOTION-CONTRACT-E-Q17D
timestamp: "2026-08-22 02:22:00Z"
base_sha: fa7ebb857187429188e404b9015c7e8a9394602f
status: PROMOTED
authorized_by: human
---

# Checkpoint Record: CHECKPOINT-E-Q17D (Multi-Hop Depth Dissociation & Endpoint Extrapolation)

## 1. Verified Scientific State
- **Contract Promoted**: `CONTRACT-E-Q17D`
- **Promotion Artifact**: [`research/promotions/PROMOTION-CONTRACT-E-Q17D.md`](../promotions/PROMOTION-CONTRACT-E-Q17D.md)
- **Verified Code Baseline (`candidate_sha`)**: `fa7ebb857187429188e404b9015c7e8a9394602f`
- **Empirical Confirmation**:
  - **Global Validity V1–V4**: Full 8-tensor SHA-256 parameter hashes verified ($16/16$), canonical 2-hop baseline retained ($16/16, 100.0\%$), contemporaneous 20-trial 1-hop sensor competence retained ($16/16, 100.0\%$), and zero sidecar reads maintained ($16/16$).
  - **Coordinate-OOD Controls $C_3, C_4, C_5$**: 2-hop transitions utilizing extended role coordinates $D=4, E=5, F=6$ pass with $100\%$ precision ($16/16$ per control). This confirms that out-of-distribution coordinate representation is not the cause of multi-hop breakdown.
  - **Score-Level Extrapolation Dissociation**: Raw endpoint directional margins extrapolate positively beyond 2-step training ($12/16$ at $k=4, p = 6.88 \times 10^{-3}$), but fail causal state-surgery transfer ($0/16$), reversal collapse ($6/16$ at $k=3$), and temporal shuffle superiority ($12/16$).
  - **Classification Verdict**: `NON_MONOTONIC_ANOMALOUS_DEPTH_PROFILE`.

## 2. Epistemic Boundaries & Core Scientific Belief
- **Established**: Under the frozen Q17C recurrent architecture, endpoint-directional scores can extrapolate beyond two-step training—including strongly at four steps—but the longer-horizon responses fail preregistered causal state-surgery, reversal, and temporal-order controls. Therefore Q17D does not establish recursive compositional depth scaling; it reveals a dissociation between score-level extrapolation and causally grounded developmental-history composition.
- **Unresolved / Immediate Frontier**: Diagnostic Scout `Q17D-B` to probe zero-history / query-only baselines, query-readout vs recurrent state geometric contributions, $k=4$ state swaps, and initial-step Jacobian attenuation.



## 3. Raw Evidence & Scout Data Packages
### `q17c_endogenous_results.json` (220645 bytes)
### `q17d_b_diagnostic_results.json` (16500 bytes)
### `q17d_depth_results.json` (97838 bytes)
### `q17e_alpha_scout_results.json` (72423 bytes)
| Metric | Mean Across Seeds | Min | Max |
| :--- | :--- | :--- | :--- |
| `seed_index` | +8.5000 | +1.0000 | +16.0000 |
| `alpha` | +0.2167 | +0.0000 | +0.5000 |
| `k2_margin` | +1.1457 | -0.5424 | +2.3016 |
| `k3_margin` | -4.0402 | -12.0562 | +2.1191 |
| `k3_transplant_margin` | -5.7872 | -13.4818 | +4.2498 |
| `k3_swap_effect` | +1.7469 | -6.8776 | +7.4212 |

### `q17e_c_gating_scout_results.json` (24227 bytes)
| Metric | Mean Across Seeds | Min | Max |
| :--- | :--- | :--- | :--- |
| `seed_index` | +8.5000 | +1.0000 | +16.0000 |
| `k2_margin` | -1.2504 | -1.7052 | -0.3089 |
| `k3_margin` | -2.9336 | -3.4329 | -1.9082 |
| `k3_transplant_margin` | -3.4130 | -4.8070 | -2.9681 |
| `k3_swap_effect` | +0.4795 | +0.1440 | +1.4664 |
| `k3_transposition_score` | -3.3957 | -4.7534 | -2.9586 |

### `q17e_d_factorization_results.json` (47274 bytes)
| Metric | Mean Across Seeds | Min | Max |
| :--- | :--- | :--- | :--- |
| `seed_index` | +8.5000 | +1.0000 | +16.0000 |
| `k2_margin` | +1.5528 | -0.3966 | +2.3877 |
| `k3_margin` | -1.5765 | -4.9713 | +2.8324 |
| `k3_transplant_margin` | -1.6051 | -6.7040 | +3.9561 |
| `k3_swap_effect` | +0.0286 | -3.5614 | +4.7431 |
| `k3_transposition_score` | -1.5082 | -6.4825 | +4.0073 |

### `q17e_e_relational_closure_results.json` (47686 bytes)
| Metric | Mean Across Seeds | Min | Max |
| :--- | :--- | :--- | :--- |
| `seed_index` | +8.5000 | +1.0000 | +16.0000 |
| `k2_margin` | +2.0240 | -0.3966 | +3.6702 |
| `k3_margin` | -4.3829 | -17.2447 | +2.1191 |
| `k3_transplant_margin` | -6.1391 | -18.8508 | +2.9857 |
| `k3_swap_effect` | +1.7562 | -3.5614 | +8.5987 |
| `k3_transposition_score` | -5.8597 | -18.5178 | +3.0328 |

### `q17e_f_typed_relational_results.json` (7554 bytes)
| Metric | Mean Across Seeds | Min | Max |
| :--- | :--- | :--- | :--- |
| `seed_index` | +8.5000 | +1.0000 | +16.0000 |
| `k2_margin` | -0.5320 | -1.7568 | +2.2895 |
| `k3_margin` | -2.7545 | -6.7880 | +0.9607 |
| `k3_transplant_m_margin` | -3.9561 | -9.4112 | -2.1376 |
| `k3_transplant_e_margin` | -3.2093 | -8.1259 | +0.0664 |
| `k3_m_swap_effect` | +1.2016 | -0.0283 | +7.6808 |

### `q17e_g_serialized_models.json` (7187225 bytes)
| Metric | Mean Across Seeds | Min | Max |
| :--- | :--- | :--- | :--- |
| `seed_index` | +8.5000 | +1.0000 | +16.0000 |
| `seed` | +94604.5000 | +88777.0000 | +100432.0000 |
| `aux_train_seed` | +95603.5000 | +89776.0000 | +101431.0000 |
| `k2_margin` | +3.4977 | +2.3714 | +4.1248 |
| `k3_margin` | +5.9563 | -7.2594 | +10.1568 |

### `q17e_g_typed_trainability_results.json` (41565 bytes)
| Metric | Mean Across Seeds | Min | Max |
| :--- | :--- | :--- | :--- |
| `seed_index` | +8.5000 | +1.0000 | +16.0000 |
| `k2_margin` | -0.4845 | -2.2242 | +4.1248 |
| `k3_margin` | -1.2429 | -7.5035 | +10.1568 |
| `k3_m_swap_effect` | +6.1058 | -4.5955 | +47.6080 |
| `k3_e_swap_effect` | -0.2122 | -7.4135 | +1.0099 |
| `s_early` | +12.6920 | +0.0210 | +123.4188 |

### `q17e_h_final_edge_results.json` (10629 bytes)
### `q17e_h_r1_exact_replay_results.json` (11574 bytes)
| Metric | Mean Across Seeds | Min | Max |
| :--- | :--- | :--- | :--- |
| `seed_index` | +8.5000 | +1.0000 | +16.0000 |
| `seed` | +94604.5000 | +88777.0000 | +100432.0000 |
| `aux_train_seed` | +95603.5000 | +89776.0000 | +101431.0000 |
| `original_k2_margin` | +3.4977 | +2.3714 | +4.1248 |
| `recomputed_k2_margin` | +3.4977 | +2.3714 | +4.1248 |
| `original_k3_margin` | +5.9563 | -7.2594 | +10.1568 |

### `q17e_i_broken_joins_results.json` (9000 bytes)
| Metric | Mean Across Seeds | Min | Max |
| :--- | :--- | :--- | :--- |
| `seed_index` | +8.5000 | +1.0000 | +16.0000 |
| `seed` | +94604.5000 | +88777.0000 | +100432.0000 |
| `k2_intact_margin` | +0.9775 | +0.8468 | +1.1773 |
| `k2_broken_join_rejection` | +1.0088 | +0.8772 | +1.0987 |
| `k2_wrong_dst_rejection` | -0.3508 | -0.4065 | -0.2977 |
| `k3_zero_shot_margin` | +2.3493 | +2.0879 | +2.9814 |

### `q17e_j_r1_frozen_addressability_results.json` (14688 bytes)
| Metric | Mean Across Seeds | Min | Max |
| :--- | :--- | :--- | :--- |
| `seed_index` | +8.5000 | +1.0000 | +16.0000 |
| `seed` | +94604.5000 | +88777.0000 | +100432.0000 |
| `k2_target_score` | +0.9774 | +0.5760 | +1.4710 |
| `k2_reverse_score` | -4.9610 | -6.1501 | -4.2009 |
| `k2_mean_distractor_score` | -1.4931 | -1.7995 | -1.2998 |
| `k2_selectivity_margin` | +1.0418 | +0.3705 | +1.8436 |

### `q17e_j_r2_depth_stability_results.json` (11009 bytes)
| Metric | Mean Across Seeds | Min | Max |
| :--- | :--- | :--- | :--- |
| `seed_index` | +8.5000 | +1.0000 | +16.0000 |
| `seed` | +94604.5000 | +88777.0000 | +100432.0000 |

### `q17e_j_relational_addressability_results.json` (16408 bytes)
### `q17e_lambda_scout_results.json` (100715 bytes)
| Metric | Mean Across Seeds | Min | Max |
| :--- | :--- | :--- | :--- |
| `seed_index` | +8.5000 | +1.0000 | +16.0000 |
| `lambda` | +0.1642 | +0.0000 | +0.5000 |
| `k2_margin` | -0.2637 | -1.7530 | +2.3016 |
| `k3_margin` | -2.5677 | -4.4459 | +2.1191 |
| `k3_transplant_margin` | -3.5497 | -10.0389 | +3.0093 |
| `k3_swap_effect` | +0.9820 | -3.5702 | +6.3581 |


---

## 4. Reorientation Prompt (Independent Review Desk Synthesis)
> **Instructions for Review Desk**:
> 1. Reconstruct: What do we actually know from the raw evidence above?
> 2. Reinterpret: What do recent results/falsifications imply mechanistically?
> 3. Reconnect: What direct, neighboring, or analogical literature solves this shape?
> 4. Rebranch: What plausible explanations remain?
> 5. Compress: What single experiment eliminates the most roadmap?