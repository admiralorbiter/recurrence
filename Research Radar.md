# Research Radar

The governing rule is:

> New information earns a place in the program when it changes a prediction, exposes a confound, provides a feasible method, or alters the expected value of an experiment—not merely because it is interesting.

This document defines how to:

- monitor relevant research without chasing every new paper;
- turn literature into concrete changes in hypotheses or controls;
- score and prioritize experiments;
- preserve abandoned ideas and negative evidence;
- update the field map without rewriting history;
- decide when the long-term architecture should change.

Monitor the following domains separately. A paper may belong to more than one.

## R1 — Human consciousness science

Watch for:

- preregistered/adversarial tests of GNWT, IIT, HOT, recurrent processing, predictive processing, and attention-schema theories;
- no-report and report-confound methods;
- markers of level versus content of consciousness;
- perturbational complexity and dynamical-state work;
- anesthesia, sleep, coma, blindsight, neglect, and metacognitive dissociations;
- formal theories with implementable predictions.

Key question:

> Does this work identify a computational variable we can manipulate in an artificial system, or does it reveal that an assumed indicator is less specific than previously thought?

## R2 — Metacognition and introspection in AI

Watch for:

- privileged-access controls;
- causal intervention methods;
- self-report calibration;
- internal anomaly detection;
- higher-order/self-indexed representations;
- self-prediction and self-modeling;
- activation monitoring and control;
- prompt/post-training effects;
- adversarial critiques and replications.

Key question:

> Does the result distinguish information about a state from information that the state belongs to the current system?

## R3 — Workspace and flexible broadcast

Watch for:

- independent J-space replications;
- open-weight Jacobian-lens methods;
- workspace capacity and competition;
- broadcast across modules/tasks;
- verbalizable versus nonverbalizable computation;
- base-versus-post-trained workspace differences;
- functional versus architectural workspace criteria.

Key question:

> Is the proposed workspace a causal routing mechanism, a readable projection, or both?

## R4 — Recurrent and state-space language models

Watch for:

- RecurrentGemma/Griffin;
- RWKV;
- Mamba and successors;
- recurrent-depth transformers;
- latent recurrent transformers;
- continuous-time neural models;
- persistent memory/state architectures;
- test-time latent computation;
- state reset and state transfer work.

Key question:

> Does the architecture provide a genuine persistent state that can be isolated, restored, swapped, and updated under null observation?

## R5 — Continual, lifelong, and developmental learning

Watch for:

- catastrophic forgetting and stability-plasticity solutions;
- lifelong LLM agents;
- online learning with bounded drift;
- developmental curricula;
- curriculum-order effects;
- self-supervised world models;
- episodic-to-semantic consolidation;
- open-ended learning;
- developmental robotics and synthetic environments.

Key question:

> Is a capacity learned because of the final dataset, or because of the individual's causal developmental trajectory?

## R6 — Agency, source monitoring, and ownership

Watch for:

- comparator and goal-based agency models;
- source-memory errors;
- intentional binding;
- action/outcome control;
- authorship and ownership judgments;
- postdictive agency;
- confabulation and choice blindness;
- AI output/prefill ownership.

Key question:

> What information actually drives an ownership judgment: prediction, goal, action, outcome, provenance, or narrative reconstruction?

## R7 — Embodiment, interoception, and regulation

Watch for:

- active inference and homeostatic control;
- allostasis and interoceptive predictive processing;
- artificial homeostasis;
- valence and reinforcement-learning theory;
- intrinsic motivation;
- body/self models;
- minimal agency and enactivist accounts;
- evidence against embodiment as a necessary condition.

Key question:

> Does endogenous regulation add a distinct computational function, or merely another reward signal?

## R8 — Mechanistic interpretability methods

Watch for:

- activation patching and causal tracing;
- sparse autoencoders and feature dictionaries;
- distributed-representation methods;
- Jacobian and causal lenses;
- intervention specificity controls;
- circuit discovery automation;
- representation alignment across time/models;
- limitations of probes and steering vectors.

Key question:

> Can this method turn a behavioral dissociation into a causal mechanism without building the answer into the measurement?

## R9 — AI welfare and responsible research

Watch for:

- indicator frameworks;
- moral-status uncertainty;
- model welfare assessment proposals;
- responsible consciousness-research guidelines;
- public communication failures;
- policy developments affecting autonomous/developmental agents.

Key question:

> Does the program need stronger precautions, a different experimental design, or a different communication standard?


# Source triage rubric

Score each source from 0–3 on the following dimensions.

| Dimension | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **Construct relevance** | adjacent only | informs terminology | changes one task | directly targets central question |
| **Causal strength** | commentary | correlational | intervention/ablation | mechanism + specificity controls |
| **Human–AI bridge** | none | analogy | comparable construct | matched prediction/experiment |
| **Method portability** | unusable | frontier-only | adaptable | directly feasible locally |
| **Replication status** | contradicted/unclear | unreplicated | partial | multiple independent replications |
| **Program impact** | no change | note only | changes backlog | changes hypothesis/architecture |

Suggested interpretation:

- **0–5:** archive/reference;
- **6–10:** scan and tag;
- **11–14:** detailed note;
- **15–18:** deep read, reproduce, or revise program.

Do not let a high-profile venue substitute for methodological strength.

# Paper-note workflow

For every source that scores 11 or higher:

1. create a note from `templates/paper_note_quick.md`;
2. identify the exact claim relevant to the project;
3. separate observed result from authors' interpretation;
4. record the strongest alternative explanation;
5. identify what is needed to reproduce it;
6. state which map cell or hypothesis changes;
7. add one of:
   - **no action**;
   - **control to add**;
   - **experiment to add**;
   - **experiment to retire**;
   - **architecture to investigate**;
   - **ethics/safety update**.

A paper note is incomplete without a program consequence, even if that consequence is “none.”

The radar should flag:

- independent replication or failure of J-space-style workspace results in open models;
- mechanistic work separating anomaly detection, semantic identification, and self-indexing;
- validated forced-choice introspection tasks that eliminate response bias;
- open recurrent models with accessible, persistent state APIs;
- evidence that null-input recurrent updates learn useful consolidation rather than drift;
- lifelong-agent methods that preserve reproducibility and bounded online learning;
- formal consciousness theories that make architecture-level predictions for artificial systems;
- responsible research standards or policy changes for AI consciousness/welfare work.

These are not predictions that the results will support the program. They are the developments most likely to change its direction.