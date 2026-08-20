# Gate B Environment Contract: The Continuity Garden v0 ("Hidden Switchboard")

**Status:** FROZEN ENVIRONMENT SPECIFICATION (Gate B Baseline)  
**Target Organism:** Small Recurrent Organism (64–128 unit GRU, 20K–250K params)  
**Primary Questions:** Q04 (Temporally Extended POMDP), Q06 (Construct Separation), Q05 (Interruption & Reconstruction)

---

## 1. Environment Architecture: "Hidden Switchboard" POMDP

The Continuity Garden v0 is a minimal, fully deterministic, partially observable environment designed to test whether persistent latent state confers a selective advantage over explicit memory and feedforward baselines.

```
                     EPISODE TIMELINE & INFORMATION FLOW
                     
  t = 0 (Cue Step):
    Ground Truth:  z in {0, 1} chosen uniformly.
    Observation:   Symbolic cue s_cue = (z + 1).
    
  t = 1 .. T_delay (Distractor Delay Steps):
    Ground Truth:  z maintained internally in environment state.
    Observation:   Distractor symbol s_distractor = 0 (or noise token).
    Delay Length:  Short: [8, 16] steps; Long: [32, 64] steps.
    
  t = T_delay + 1 .. T_total (Query Steps):
    Ground Truth:  Query bit x_t in {0, 1}.
    Target Action: a_t* = x_t XOR z.
    Observation:   Query symbol s_query = (x_t + 3).
    Consequence:   Action produces observable sensor feedback on step t+1.
```

---

## 2. Q06: Strict Three-Layer Construct Separation

To prevent construct leakage into observations, the environment strictly separates:

```
  ┌────────────────────────────────────────────────────────┐
  │ 1. Ground Truth State (Environment-Internal Only)     │
  │    - hidden_mode: int (z)                              │
  │    - true_source: int (self vs environment)            │
  │    - resource_integrity: float                         │
  └──────────────────────────┬─────────────────────────────┘
                             │ SensorTransform (Noise / Masking)
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ 2. Agent Observation (Visible to Organism)             │
  │    - symbol: int (s_t in {0, 1, 2, 3, 4})              │
  │    - action_feedback: int | None                       │
  │    - NO EXPLICIT LABELS (no 'uncertainty', 'self_state')│
  └──────────────────────────┬─────────────────────────────┘
                             │ Policy / Recurrence Update
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ 3. Organism Learned Estimate (Internal to Organism)    │
  │    - latent_state: h_t in R^d                          │
  │    - predicted_action: a_hat_t in {0, 1}               │
  │    - internal_uncertainty: decoded estimate            │
  └────────────────────────────────────────────────────────┘
```

**Unit Test Invariant:** `test_no_target_construct_leakage` must inspect every observation field and assert that ground truth variables (`hidden_mode`, `true_source`, `uncertainty`) never appear in `AgentObservation`.

---

## 3. Model Benchmark Set (Q04)

Every experiment evaluates four standard conditions:

| Model / Condition | Description & Capacity | Target Role |
|---|---|---|
| **Oracle Belief-State** | Receives $z$ directly at query time. | **Ceiling Performance** ($\ge 0.95$) |
| **Current-Input MLP** | 2-layer MLP receiving only current observation $o_t$. | **No-Memory Baseline** ($\approx 0.50$) |
| **History-Window MLP** | MLP receiving sliding window of last $K=4$ observations. | **Explicit Finite-Memory Baseline** |
| **GRU Organism** | 1-layer 64–128 unit GRU ($h_t = \text{GRU}(h_{t-1}, o_t)$). | **Latent Recurrence Condition** ($\ge 0.85$) |

### Surgical State Interventions (Q04 / Q05):
1. **Recurrent State Reset:** $h_{t^*} \leftarrow 0$ immediately after the cue step ($t^* = 1$). Must cause performance to collapse to chance ($\approx 0.50$).
2. **Sham Buffer Reset:** Reset an irrelevant dummy buffer; GRU performance must remain $\ge 0.85$.
3. **Interruption & Reconstruction (Q05):**
   - Branch A: Uninterrupted baseline.
   - Branch B: Explicit memory restored + Latent state reset.
   - Branch C: Latent state preserved + Explicit memory cleared.
   - Branch D: Compute-matched replay.

---

## 4. Pre-Registered Q04 Acceptance Gates

1. **Oracle Accuracy:** $\ge 0.95$.
2. **GRU Held-Out Accuracy:** $\ge 0.85$ across short ($8-16$) and long ($32-64$) delay regimes.
3. **Feedforward Ceiling:** $\le 0.55$ (confirms environment does not leak information).
4. **Recurrent Margin:** $\Delta(\text{GRU} - \text{FF}) \ge 0.30$ ($30$ percentage points).
5. **Causal Reset Collapse:** $\text{GRU}_{\text{reset}} \le 0.55$ (proves latent state is causally necessary).
