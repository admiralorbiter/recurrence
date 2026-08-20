# Developmental Analysis & Manifest Contract

**Status:** FROZEN METHODOLOGICAL CONTRACT  
**Scope:** Lineage Tracking, Multi-Seed Evaluation, Causal Abstraction, and Reporting Standards  
**Applicability:** All Moonshot Experiments (Gate A through Gate I)

---

## 1. Unified Experiment Manifest Schema

Every experimental run (whether RecurrentGemma white-box cache or Continuity Garden developmental lineage) must serialize a standardized JSON manifest:

```json
{
  "run_id": "run_YYYYMMDD_HHMMSS_uuid",
  "git_sha": "40-character-hex-sha",
  "experiment_id": "Q04_hidden_switchboard_baseline",
  "gate": "GATE_B",
  "model_revision": "pinned-sha-or-architecture-spec",
  "environment_revision": "v0.1.0",
  "seed": 42,
  "lineage": {
    "lineage_id": "lineage_org_A17",
    "parent_lineage_id": "lineage_org_A",
    "fork_step": 381,
    "event_hash": "sha256-of-prior-trajectory"
  },
  "execution": {
    "device": "cpu",
    "precision": "fp32",
    "batch_size": 32,
    "torch_version": "2.x.x"
  },
  "condition": {
    "name": "gru_recurrent_delay64",
    "manipulation_type": "state_intervention",
    "intervention_target": "h_t"
  },
  "metrics": {
    "oracle_accuracy": 1.0,
    "held_out_accuracy": 0.942,
    "reset_collapse_accuracy": 0.501
  }
}
```

---

## 2. Lineage & Branching Invariants

To support exact developmental twin experiments (Gates G & H), the lineage system must guarantee:

1. **Deterministic Cloning:** Cloning an agent at step $t^*$ and running copies $A_1$ and $A_2$ through identical future observations must produce **bitwise identical** actions and internal states.
2. **Exact Event Branching:** Modifying exactly one observation or action at step $t^*$ causes trajectories to diverge starting at step $t^*$, while perfectly matching prior history ($t < t^*$).
3. **State Snapshot & Restoration:** `organism.snapshot()` serializes full parameters and recurrent buffers; `organism.restore(snapshot)` recovers exact computational state.

---

## 3. Statistical & Multi-Seed Philosophy

To prevent cherry-picking or single-seed variance traps:

- **Sanity / Development:** 1–2 seeds.
- **Scout / Gate Screen:** 5–8 independent seeds.
- **Confirmatory / Landmark Battery:** $\ge 20$ seeds with reported bootstrap confidence intervals ($B=10,000$).
- **Aggregates:** Report Interquartile Mean (IQM) and empirical distributions alongside standard means and standard deviations.

---

## 4. Standardized Reporting Template (The Synchronization Packet)

Every gate result submitted to the synchronization review must adhere to this 10-point format:

```text
================================================================================
SYNCHRONIZATION REPORT: [GATE ID / QUESTION ID]
================================================================================
1. QUESTION:                  [Exact scientific question]
2. WHAT WAS FROZEN:           [Pre-registered protocol, parameters, seeds]
3. WHAT WAS RUN:              [Number of trials, seeds, compute budget]
4. PRIMARY ESTIMAND:          [Formula and empirical target]
5. RESULT + UNCERTAINTY:      [Point estimate + 95% CI / IQM]
6. CONTROL RESULTS:           [Observer, random-direction, sham, fluency]
7. FAILURES / INVALID CELLS:  [Broken assumptions, uncooperative items]
8. STRONGEST ALTERNATIVE:     [Competing non-self / non-monitor explanation]
9. CLAIM CEILING:             [Exact epistemic boundary of the finding]
10. DECISION:                 [PROMOTE / REPAIR / KILL]
================================================================================
```
