---
title: Experiment Card — E__
experiment_id: E__
version: 0.1.0
status: idea
owner:
created:
updated:
---

# E__ — Experiment Name

## One-sentence question

> What is the smallest question this experiment answers?

## Why this experiment now

- **Gap addressed:**
- **Prerequisites:**
- **Expensive future step it could prevent:**
- **Connection to the field map:**

## System boundary

Check and specify all included components.

- [ ] frozen model weights
- [ ] current activations/KV state
- [ ] persistent latent state
- [ ] external memory
- [ ] workspace/self-state
- [ ] scheduler/clock
- [ ] environment
- [ ] online parameter learning
- [ ] tools/actions

**Exact object under study:**

## Construct and claim ceiling

- **Primary construct:**
- **Operational definition:**
- **Evidence level sought:** behavior / representation / causal role / mechanism / cross-system analogy
- **Maximum warranted claim even under a positive result:**
- **Claims this experiment cannot support:**

## Hypotheses

### H1 — Target hypothesis

### H0 — Null

### A1 — Strongest alternative explanation

### A2 — Additional alternative

## Predictions table

| Outcome/condition | H1 predicts | H0 predicts | A1 predicts | Discriminating observation |
|---|---|---|---|---|
| | | | | |

## Independent variables

| Variable | Levels | Manipulated or observed | Randomized? |
|---|---|---|---|
| | | | |

## Dependent variables

### Primary

- metric:
- unit:
- direction interpreted as support:

### Secondary

### Diagnostic only

## Conditions and controls

- [ ] fresh/no-state baseline
- [ ] explicit-memory match
- [ ] compute/token-budget match
- [ ] observer/reconstruction control
- [ ] sham intervention
- [ ] norm-matched random intervention
- [ ] timing control
- [ ] input-anomaly control
- [ ] state-damage/perplexity control
- [ ] prompt paraphrase/order control
- [ ] base/instruct or other matched-model control

**Control matrix:**

| Condition | Visible information | Compute | Explicit memory | Latent state | Intervention |
|---|---:|---:|---:|---:|---|
| | | | | | |

## Stimuli and tasks

- **Generator/version:**
- **Scout size:**
- **Establish size:**
- **Held-out/confirmatory set:**
- **Leakage checks:**
- **Answer format:**
- **Scoring:** exact / parser / judge

## Model and compute

- **Model ID/revision:**
- **Base/instruct:**
- **dtype/quantization:**
- **Backend:**
- **GPU estimate:**
- **Expected wall time:**
- **Expected storage:**

## Procedure

1.
2.
3.

## State and intervention details

- **State variable:**
- **Capture timing:**
- **Restore/reset/swap operation:**
- **Intervention target:**
- **Strength selection:**
- **Specificity controls:**

## Analysis plan

- **Primary statistical model/test:**
- **Random effects/grouping:**
- **Effect size:**
- **Uncertainty interval:**
- **Multiple comparisons:**
- **Sequential stopping rule, if any:**
- **Exclusions:**
- **Missing/invalid outputs:**

## Scout plan

The scout should be capable of killing or redesigning the experiment.

- **Minimum implementation:**
- **Number of conditions/items:**
- **What failure looks like:**
- **What would justify expansion:**

## Go / pivot / stop rules

### Proceed to establish if

### Redesign if

### Retire if

### Escalate to causal/mechanistic phase if

## Interpretation branches

| Result | Narrow conclusion | Remaining alternative | Next experiment |
|---|---|---|---|
| robust positive | | | |
| null with precision | | | |
| prompt-sensitive | | | |
| observer matches self | | | |
| state intervention changes both task and damage metric | | | |
| unexpected result | | | |

## Ethics and safety card

- **Autonomy stage:** A0/A1/A2/A3+
- **Online learning:**
- **External access:**
- **Negative regulatory state:** none/low/moderate/high
- **Human/private data:**
- **Irreversible changes:**
- **Stopping condition:**
- **Checkpoint/deletion policy:**

## Reproducibility requirements

- [ ] frozen config
- [ ] model revision recorded
- [ ] complete run manifest
- [ ] raw outputs preserved
- [ ] state snapshot tier specified
- [ ] RNG/environment state recorded
- [ ] code tests pass
- [ ] source tables generated

## Open questions

## Links

- preregistration:
- code:
- runs:
- analysis:
- result report:
- decision record:
