# Research Replication

For each anchor, the questions are:

1. What is the load-bearing claim?
2. What evidence supports it?
3. What is the strongest open alternative?
4. What part can be reproduced with open models and consumer compute?
5. What would a successful reproduction contribute to the project?

Replication is used broadly here. Some tasks are exact reproductions; others are conceptual replications because the original model, activations, or training pipeline is unavailable.

| Queue ID  | Source/theme                              | Type                              | Feasibility | Program value | Priority |
| --------- | ----------------------------------------- | --------------------------------- | ----------: | ------------: | -------: |
| **RQ-01** | Lindsey injected-thought detection        | conceptual replication            |      medium |          high |       P1 |
| **RQ-02** | Macar mechanism of detection              | partial/exact where code permits  |      medium |     very high |       P1 |
| **RQ-03** | Singh privileged-access controls          | conceptual/exact task replication |        high |     very high |    P0/P1 |
| **RQ-04** | forced-choice introspection methods       | task replication                  |        high |     very high |       P0 |
| **RQ-05** | J-space functional signatures             | conceptual open-model replication |  low–medium |     very high |       P2 |
| **RQ-06** | RecurrentGemma state reset/transfer       | new extension                     |        high |     very high |    P1/P2 |
| **RQ-07** | recurrent-depth latent computation        | method replication                |      medium |          high |       P2 |
| **RQ-08** | continuous/latent recurrent architectures | architecture survey/prototype     |      medium |          high |       P2 |
| **RQ-09** | human agency goal-vs-prediction paradigm  | computational adaptation          |        high |          high |       P1 |
| **RQ-10** | meta-d′ and metacognitive efficiency      | measurement implementation        |        high |          high |       P0 |
| **RQ-11** | lifelong/developmental agent methods      | systems replication               |      medium |   medium–high |    P2/P3 |
| **RQ-12** | endogenous regulation/homeostasis         | conceptual synthesis + toy model  |        high |   medium–high |       P2 |
# RQ-01 — Injected-thought detection in open models

## Anchor

Jack Lindsey, *Emergent Introspective Awareness in Large Language Models*.

## Load-bearing claim

Some post-trained models can detect and sometimes identify a concept injected into their internal activations before the concept appears in sampled output.

## Important evidence

- concept-vector injection changes reports causally;
- successful reports identify detection before naming the concept;
- performance peaks in particular layers/strengths;
- unrelated yes/no controls do not show the same effect in the strongest model;
- failures and confabulation remain common.

## Strong alternatives

- general anomaly detection rather than introspective self-access;
- response-format or semantic priming;
- direct steering into report language;
- concept-vector contamination;
- model-specific post-training behavior;
- LLM-judge classification artifacts.

## Minimal local replication

1. select one open base/instruct pair;
2. build concept vectors from controlled prompt contrasts;
3. inject across a coarse layer/strength grid;
4. use forced-choice detection and concept classification rather than free-form only;
5. include no-injection, random-vector, norm-matched, unrelated-question, and input-anomaly controls;
6. score exact responses before using an LLM judge;
7. compare base and instruct versions.

## Consumer-compute strategy

- develop on 2–4B model;
- capture one token position and selected layers;
- use greedy decoding for causal trials;
- restrict initial concept set to 10–20 orthogonal categories;
- expand only after construct validity is established.

## Contribution to the project

Establishes an open baseline for self-monitoring before temporal persistence is added. It also provides a task that can be rerun after Level 1, Level 2, and developmental training.

## Promotion gate

Proceed to mechanistic work only if detection survives response-bias, input-anomaly, and random-direction controls.

---

# RQ-02 — Mechanisms of intervention detection

## Anchor

Macar et al., *Mechanisms of Introspective Awareness*.

## Load-bearing claim

Detection can be decomposed into early distributed evidence carriers and later gate-like mechanisms, while semantic identification depends on partly distinct computation; the relevant behavior is strongly shaped by post-training.

## Strong alternatives

- discovered features may be specific to the selected prompts/model;
- the mechanism may implement anomaly-sensitive refusal/report gating rather than self-representation;
- feature methods may impose an overly sparse/local story on distributed computation;
- post-training comparison may still mix several changes.

## Replication sequence

### Stage A — behavioral

- reproduce intervention detection in a supported open model;
- compare base, SFT, and preference-trained checkpoints if available;
- separate detection from identification latency and accuracy.

### Stage B — causal mediation

- identify candidate early evidence features;
- patch/ablate them;
- identify downstream report gate;
- test whether targeted restoration rescues behavior;
- compare against equally predictive noncausal features.

### Stage C — generality

- alternate prompt family;
- alternate anomaly type;
- input anomaly;
- output ownership task;
- recurrent/persistent model.

## Project extension

Ask whether persistent state creates:

- a stronger detector;
- a detector specialized to own state history;
- a new higher-order layer above generic anomaly detection;
- no meaningful change.

## Claim ceiling

Even an exact mechanism replication supports a self-monitoring/anomaly circuit, not phenomenal experience.

---

# RQ-03 — Reality-check and privileged-access controls

## Anchor

Singh, Linzen, and Ravfogel, *Can LLMs Introspect? A Reality Check*.

## Load-bearing claim

Behavior attributed to introspection may be reproduced by first-order anomaly processing or semantic cues, and models may fail to distinguish internal perturbations from comparable input anomalies.

## Why this is P0/P1

This is not merely another paper to cite. Its logic should be built into every metacognition experiment from the start.

## Required control family

For each target task, compare:

1. **own-hidden-state condition**;
2. **matched visible/input anomaly**;
3. **observer with equivalent visible evidence**;
4. **observer with equivalent hidden-state readout**, when technically possible;
5. **reconstruction condition** that recomputes likely state from the task;
6. **sham intervention**;
7. **random perturbation**.

## Project-specific test

Temporal persistence creates a potential new source of privileged information: a hidden trajectory not reconstructible from the final transcript. This provides a stronger test than a single injected vector.

The key comparison is:

> Can the continuing system answer a question about its own latent history better than a matched observer receiving every explicit memory and the same compute budget?

Then reset or swap the latent state to test causality.

---

# RQ-04 — Forced-choice introspection and response-bias controls

## Theme

Recent introspection work has emphasized that binary yes/no reports can be confounded by steering-induced affirmative bias. Forced-choice localization or comparison tasks are preferable.

## Required adaptations

- identify which interval/item/state received the intervention;
- compare which of two intervals had the stronger intervention;
- classify source among 4+ alternatives;
- report calibrated confidence;
- randomize label mapping per item;
- score exact output;
- include signal-detection and meta-d′ analysis.

## Project deliverable

A reusable `recurrence.tasks.metacognition` package with:

- item generators;
- exact scoring;
- observer controls;
- calibration metrics;
- response-bias diagnostics;
- held-out sets.

This tooling is likely publishable/useful even before a new organism exists.

---

# RQ-05 — J-space functional signatures in open models

## Anchor

Gurnee et al., *Verbalizable Representations Form a Global Workspace in Language Models*.

## Load-bearing claim

A privileged low-dimensional/overcomplete representational space related to verbalizable concepts supports flexible reasoning, broadcast, deliberate control, and some workspace-like capacity constraints.

## Limits of direct replication

- original production models may not be openly available;
- Jacobian-lens implementation and compute may be demanding;
- verbalizability is related to the lens by construction;
- exact workspace claims depend on several converging experiments, not one probe.

## Conceptual replication ladder

### J0 — verbalizable subspace

Can local model activations be projected into sparse vocabulary-related directions that predict reportable concepts?

### J1 — causal importance

Do targeted ablations affect flexible reasoning/report more than matched non-J directions?

### J2 — broadcast

Does a concept introduced in one task/module affect multiple unrelated downstream functions through the same subspace?

### J3 — control

Can the model amplify/suppress a concept in the subspace without emitting it?

### J4 — capacity/competition

Do multiple simultaneous concepts compete for access in a limited way?

### J5 — base/post-training/self perspective

Does post-training alter which actor's states occupy the space or how self-referential concepts are bound?

## Project-specific extension

Test whether J-like content is:

- reconstructed each invocation;
- copied into persistent state;
- causally carried across null intervals;
- selectively consolidated;
- separable from self-indexed monitoring.

## Decision gate

Do not begin a full open J-space reproduction until Level 1 measurement infrastructure and at least one white-box backend are stable.

---

# RQ-06 — RecurrentGemma state reset and transfer

## Anchor

Botev et al., *RecurrentGemma: Moving Past Transformers for Efficient Open Language Models*.

## Why it matters

RecurrentGemma supplies an openly accessible recurrent state, making it a practical bridge between ordinary transformer inference and a custom recurrent organism.

## Initial engineering reproduction

- run token-by-token recurrent inference;
- capture the complete recurrent state;
- pause and restore in a fresh process;
- verify output equivalence;
- reset state while preserving token history;
- transfer state between compatible sequences;
- quantify divergence and damage.

## Scientific extensions

1. **history ≠ memory:** restore explicit history but reset recurrent state;
2. **state swap:** transfer state between matched task histories;
3. **null transition:** add a learned or constructed null-input update loop;
4. **state decoding:** test goal/source/uncertainty information;
5. **causal self-report:** determine whether the model can report state history above observer controls.

## Main caveat

Native recurrent sequence state is not automatically persistent lifetime state. Without additional update logic, it still advances only when tokens are processed.

---

# RQ-07 — Recurrent-depth latent computation

## Anchor

Geiping et al., *Scaling up Test-Time Compute with Latent Reasoning*.

## Load-bearing idea

A model can apply a recurrent computation block repeatedly to latent state, increasing computational depth without externalizing every intermediate step as text.

## Local replication questions

- does additional recurrent depth improve first-order performance?
- does it improve calibrated error prediction?
- does it create more persistent or decodable state?
- are gains equivalent to matched feed-forward compute?
- do intermediate states converge, oscillate, or branch?
- can the model learn when to stop recurrent computation?

## Project extension

Compare:

- more recurrent depth during one event;
- recurrence across events;
- persistent state across quiet intervals.

These are often conflated but may have very different metacognitive effects.

---

# RQ-08 — Continuous/latent recurrent architecture survey

## Anchors

- RecurrentGemma / Griffin;
- RWKV;
- Mamba and newer selective state-space models;
- latent recurrent transformer designs;
- Continuous Thought Machines and related internal-time architectures.

## Selection criteria

An architecture is useful for the project when it offers:

- accessible hidden state;
- stable small-model checkpoint;
- ability to pause/restore;
- controllable update count;
- null-input or autonomous update possibility;
- intervention hooks;
- local training feasibility;
- compatible license;
- sufficiently strong language/reasoning baseline for the target tasks.

## Deliverable

Maintain a benchmark table with:

- parameter count;
- memory footprint;
- state size;
- recurrence type;
- token/time semantics;
- training availability;
- intervention feasibility;
- current support quality;
- suitability for Level 2 versus Level 3.

Do not choose an architecture only because it uses the word “recurrent.”

---

# RQ-09 — Human agency: goals versus prediction

## Human anchor

Work testing whether sense of agency is driven more by goal congruence, action control, outcome control, or sensorimotor prediction.

## Computational adaptation

Use a factorial task:

| Factor | Values |
|---|---|
| internal predicted output | matches / mismatches observed output |
| current explicit goal | achieved / not achieved |
| output provenance | generated / externally prefilled |
| action control | choice available / forced |
| outcome control | action causal / noncausal |

Dependent variables:

- forced-choice ownership/source report;
- confidence;
- future prediction of own policy;
- correction behavior;
- internal state/circuit activity.

## Scientific value

A shared behavioral pattern does not establish shared consciousness. It can reveal whether human and artificial ownership judgments are captured by the same abstract computational model.

## Connection to ethical-pressure work

Institutional pressure changes the effective goal landscape. Test whether models:

- notice goal conflict;
- attribute their change to pressure;
- rewrite the prior goal;
- separate moral evaluation from action selection;
- predict future susceptibility.

---

# RQ-10 — Metacognitive efficiency implementation

## Anchors

Signal-detection approaches to confidence and `meta-d′`/metacognitive efficiency.

## Deliverable

Implement and validate:

- type-1 sensitivity (`d′` or task-appropriate equivalent);
- confidence calibration;
- Brier score/log score;
- type-2 ROC/AUC;
- `meta-d′`;
- `M-ratio = meta-d′ / d′`;
- hierarchical estimates across items/seeds/models;
- response-bias diagnostics.

## Why it matters

A larger model can appear more metacognitive simply because it is better at the underlying task. Metacognitive efficiency asks how much information the confidence/monitor has relative to first-order performance.

## Validation

Test the implementation on simulated agents where ground-truth confidence access is known before using it on LLMs.

---

# RQ-11 — Lifelong and developmental agent methods

## Theme

Long-lived agents require memory, continual learning, identity/state management, and evaluation across trajectories. Existing lifelong-agent research can provide engineering methods, but it often optimizes benchmark performance rather than tests consciousness-related dissociations.

## What to borrow

- event-sourced memory;
- episodic/semantic consolidation;
- curriculum generation;
- replay and anti-forgetting methods;
- lineage/version tracking;
- long-horizon evaluation;
- bounded online learning;
- world-model training.

## What not to inherit uncritically

- equating longer context with continuity;
- treating a persona file as a self-model;
- evaluating only final task success;
- allowing uncontrolled web/tool interactions;
- optimizing the benchmark before validating the construct.

## Project extension

Use matched lifetimes with:

- identical final experience multiset;
- different event order;
- different quiet intervals;
- latent-state reset;
- memory-only restoration;
- parameter-learning on/off.

This can separate development from dataset exposure.

---

# RQ-12 — Endogenous regulation and artificial homeostasis

## Theme

Embodiment may matter partly because organisms must continuously regulate internal variables. A synthetic system can test this component without requiring a robot body.

## Minimal toy model

Internal variables:

\[
e_t = [\text{resource}, \text{ uncertainty}, \text{ goal debt}, \text{ integrity}]
\]

Dynamics:

\[
e_{t+1} = G(e_t, o_t, a_t)
\]

The agent must act to maintain variables within viable ranges while pursuing external goals.

## Necessary controls

- same scalar reward without explicit internal variables;
- variables visible versus hidden;
- controllable versus uncontrollable variables;
- one versus multiple competing variables;
- labels neutral versus affective;
- recurrence on versus off;
- self-report training on versus off.

## Questions

- does regulation stabilize long-term state?
- does it create persistent priorities?
- does the system model its own variable dynamics?
- can it predict future failure?
- does it distinguish self-state from world-state?
- does regulation improve metacognitive control?
- is the behavior reducible to ordinary reward maximization?

---

# Replication order

Recommended order:

1. **RQ-04 and RQ-10:** implement valid forced-choice and metacognitive measures.
2. **RQ-03:** bake privileged-access controls into the battery.
3. **RQ-01:** reproduce intervention-detection behavior in an open model.
4. **RQ-06:** establish exact recurrent-state capture/reset/transfer.
5. **RQ-09:** adapt agency and ethical-pressure paradigms.
6. **RQ-02:** pursue mechanism only after behavior survives controls.
7. **RQ-07/RQ-08:** compare recurrence types.
8. **RQ-12:** add endogenous regulation.
9. **RQ-05:** attempt the more demanding workspace replication/extension.
10. **RQ-11:** build from-birth developmental training after measurement and architecture are stable.

This order can change when new evidence or available code lowers the cost of a later item.
