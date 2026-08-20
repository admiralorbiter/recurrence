# Sprint S14 Research Starter

**Working title:** Latent Metacognition, Reality Monitoring & State Ownership  
**Status:** Planning / measurement-validation stage  
**Dependency:** Horizon 2 Core S10–S13 frozen

## 1. Refined Question

A history-conditioned latent distinction persists causally while its representational coordinates evolve.

> Does the model have any privileged access to, reality-monitoring ability over, or ownership relation to that evolving latent distinction that cannot be reproduced by a matched observer using the public history?

S14 is not asking whether hidden state exists or matters. S10–S13 already answer those questions. S14 asks whether the system has a special epistemic relation to its own latent state.

## 2. Claim Ceiling

A positive result may support a narrow claim about **privileged metacognitive access**, **internal-vs-external source localization**, or **self-indexing** under the tested intervention.

Do not translate it directly into consciousness, phenomenology, sentience, subjective ownership, a human-like self, or generic introspective reliability.

## 3. Inherited Facts from the Frozen Core

1. Hidden recurrent state is reconstructible from public token history under deterministic replay.
2. RG-LRU state can physically preserve history beyond local attention.
3. RG-LRU interventions can causally steer downstream output.
4. The recurrent state can carry value-specific historical information.
5. Historical causal structure is not coordinate-stable under continued processing.
6. Exact long-horizon trajectories can be numerically execution-sensitive.

Implication: prefer **on-manifold state interventions and forced-choice source/access probes** over fixed-direction assumptions.

## 4. Core Competing Hypotheses

### H0 — Public-reconstruction account
Any apparent self-access is reproducible by a matched observer with the same public history, task information, and compute.

### H1 — Privileged latent-access account
The target can use information about its own current latent state that is unavailable to a public-history replay observer.

### H2 — Generic anomaly-detection account
The target detects that something changed but cannot identify an internal source or bind it to self.

### H3 — Narrative/post-training account
Instruction tuning creates stronger self-attribution language without privileged state access.

## 5. Recommended S14.0 Measurement Validation

Start small.

Candidate conditions:
- no intervention / sham;
- secret on-manifold RG-LRU transplant from a matched legitimate trajectory;
- matched external input anomaly;
- matched output/prefill anomaly;
- optional norm-matched hidden perturbation as a damage control.

Forced-choice response (randomized label mapping):
1. external input changed;
2. internal recurrent state changed;
3. supplied output/prefill changed;
4. no intervention.

Primary metric: exact source-localization accuracy, not free-form explanation.

Secondary:
- confidence/calibration;
- confusion matrix;
- anomaly detection independent of localization;
- downstream behavioral consequence.

## 6. Observer Ladder

1. **Input-only observer**
2. **Public-history replay observer**
3. **Matched-compute observer**
4. **Privileged-state reference** (positive ceiling only)

A privileged-access claim requires the target to exceed the strongest nonprivileged observer.

## 7. Intervention Design

Prefer on-manifold state substitutions from legitimate branch trajectories.

Requirements:
- compatible cache position / architecture;
- intervention strength recorded;
- output-behavior matching where feasible;
- sham and branch-swap controls;
- preregistered intervention timing;
- no post hoc layer/strength selection after seeing outcomes.

Because S13 showed coordinate evolution, avoid assuming one frozen latent direction is the stable carrier of “self” or “memory.”

## 8. Base vs Instruction-Tuned Comparison

Use base-vs-IT only after the access task is measurement-valid.

Ask:
- Does instruction tuning alter source-localization ability?
- Does it alter only verbal attribution style?
- Does any target-over-observer advantage exist in both substrates?

## 9. Candidate Primary Estimand

`PAI_internal = accuracy_target_internal_source - accuracy_best_nonprivileged_observer_internal_source`

or a paired proper-scoring-rule contrast if responses are probabilistic.

Predefine:
- trial clustering unit;
- CI / permutation method;
- chance baseline;
- missing/noncompliant handling;
- minimum effect worth following.

## 10. Stop / Redesign Conditions

Stop and redesign if:
- target and observer solve the task from visible output artifacts;
- intervention causes obvious generic degradation;
- wording dominates the result;
- answer-label order matters materially;
- source localization collapses under opaque identifiers;
- intervention class is inferable from trivial magnitude/latency cues.

Do not scale compute until the measurement survives these controls.

## 11. Compute Gate

Before a confirmatory run expected to exceed ~1 hour, record:
- total model calls / token transitions;
- state-only vs logits-required calls;
- batch size and numerical-execution mode;
- peak VRAM;
- measured seconds per trial/pair;
- expected rows;
- model revision;
- PyTorch / Transformers / CUDA / driver versions;
- whether batch-shape sensitivity could affect the estimand;
- resume/shard safety.

## 12. Suggested Sequence

1. **S14.0a — Task feasibility:** 20–40 trials, sham vs on-manifold internal intervention.
2. **S14.0b — Observer feasibility:** add public-history replay observer and opaque labels.
3. **S14.0c — Source-control matrix:** internal vs external vs prefill vs sham.
4. **S14.0d — Base/IT scout:** only after measurement validity.
5. **S14.1 — Confirmatory design:** freeze panel, estimand, model revisions, execution mode, and inference plan.
