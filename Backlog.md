
# Backlog

## Retire or demote a direction when

- effects vanish under basic paraphrase/order controls;
- observer conditions match self conditions across well-powered tests;
- latent-state benefits vanish after memory and compute matching;
- intervention effects are fully explained by norm/damage;
- a new paper resolves the question more convincingly than we can;
- required access exceeds available compute with no smaller analogue;
- the experiment cannot discriminate competing explanations;
- the result would be interesting only under an unjustified consciousness interpretation.
## P0 — required foundations

| Item | Why now | Exit condition |
|---|---|---|
| deterministic replay and state restoration | all continuity claims depend on it | E00 passes on fresh process |
| prompt/measurement robustness | prevents research on prompt artifacts | one stable forced-choice battery |
| observer/reconstruction controls | defines claim ceiling | self versus observer comparison implemented |
| explicit-memory ladder | establishes the strongest nonrecurrent baseline | memory formats and costs quantified |
| research ledger + experiment registry | prevents retrospective narrative drift | every run and decision traceable |

## P1 — highest-value near-term experiments

| Item | Main gap | Why it could matter |
|---|---|---|
| incremental ticks versus transcript/compute-matched replay | continuity vs extra compute | cheap test of Level 1's scientific value |
| reset with all explicit memory restored | history vs memory | operationalizes “history is not autobiography” |
| ethical-pressure causal self-attribution | metacognition | uses existing user work in a stronger paradigm |
| forced-choice output ownership | source/self model | avoids yes-bias and free-form narrative |
| own-state versus matched-observer prediction | privileged access | separates self-modeling from ordinary inference |
| null-interval unresolved-goal task | endogenous temporal evolution | tests whether quiet processing changes later cognition |

## P2 — prepare while P1 runs

| Item | Main gap | Dependency |
|---|---|---|
| RecurrentGemma state interface | latent recurrence | E00 and backend tests |
| recurrent adapter prototype | persistent state | evidence that state question is worth training |
| latent reset/swap toolkit | causal continuity | stable Level 2 backend |
| dynamics analysis package | recurrent organization | recurrent state traces |
| micro-world v0 | developmental control | state/action schemas stable |
| endogenous variable module | regulation | micro-world and task battery |

## P3 — long-horizon candidates

| Item | Purpose | Reason to defer |
|---|---|---|
| from-birth recurrent micro-model | developmental self-model emergence | architecture and metrics not yet validated |
| online lifelong learning | path-dependent development | reproducibility and safety burden |
| persistent workspace adapter | workspace/self-state dissociation | requires a validated open workspace proxy |
| matched human experiment | cross-system comparison | artificial construct must stabilize first |
| larger 8–12B mechanistic replication | scale/generalization | small model should first establish method |
| real-time continuous deployment | wall-time continuity | logical-time experiments are cheaper and cleaner |
At each review, ask whether the strongest likely contribution is currently:

1. **measurement:** a better way to separate self-report, monitoring, and privileged access;
2. **control:** a transcript/memory/compute/state matching paradigm;
3. **method:** state restoration, swap, or developmental counterfactual tooling;
4. **negative evidence:** recurrence or persistence fails to add a proposed capacity;
5. **mechanism:** identified causal state/circuit for self-monitoring or continuity;
6. **architecture:** persistent workspace or recurrent self-state design;
7. **development:** a capacity depends on individual trajectory rather than final data;
8. **theory bridge:** an artificial dissociation sharpens a human consciousness theory;
9. **research infrastructure:** an open model organism and evaluation suite.

The answer may change over time. The program should follow the strongest evidence rather than remaining loyal to the most ambitious original story.

# Deferred Methodological & Architectural Refinements (Reviewer Capture Index)

The following high-value design refinements are indexed to ensure they are incorporated during sprint execution (S00–S15) without creating upfront specification debt:

1. **Construct Leakage & Ground-Truth Separation:** Expose ground-truth variables, noisy interoceptive sensors, and learned internal estimates separately. Distinguish Installation vs. Emergence protocols. (Indexed in [`9. The Continuity Garden.md`](9.%20The%20Continuity%20Garden.md)).
2. **Off-Manifold Intervention Safeguards:** Implement interpolation curves $z(\alpha)$, compatible-branch swaps, and post-swap recovery dynamics to prevent impossible neural state artifacts. (Indexed in [`5. Experiments.md`](5.%20Experiments.md)).
3. **Channel Capacity & Interface Controls:** Expand compute controls in E04/E06/E12 to include same-dimensional nonrecurrent channels and cleared read/write interfaces. (Indexed in [`5. Experiments.md`](5.%20Experiments.md)).
4. **Metric Refinements & Observer Ladder:** Preregister PAI paired contrasts (intersection criterion), CAS causal alignment, CDI difference-in-differences, and 4-rung observer ladder. (Indexed in [`7. Measurement and Analysis Plan.md`](7.%20Measurement%20and%20Analysis%20Plan.md)).
5. **Developmental Competence Equivalence:** Require matched first-order competence across developmental histories before attributing differences to developmental trajectory. (Indexed in [`1. Charter.md`](1.%20Charter.md)).
6. **RecurrentGemma Role Specification:** Treat RecurrentGemma as token-sequence recurrence control while developing event-time latent core $z_{t+1}=R(z_t, \varnothing)$ in parallel. (Targeted for S10–S12 execution).
7. **Bibliography Infrastructure & OSF Preregistration:** Set up `references.bib`, Zotero integration, and OSF time-stamped preregistrations during S00 foundation sprint.
8. **Forced-Choice vs. Free-Generation 2x2 Matrix (S02):** Implement a 2x2 factorial evaluation (`Meaningful vs. Opaque Identifiers` $\times$ `Forced-Choice vs. Free-Generation`) to separate associative availability from surface-realization / token-generation failure.
9. **Tokenizer Strata Diagnostic (S02):** Stratify benchmark items by tokenization complexity (1–2 $\to$ 3–4 $\to$ 5–6 $\to$ 7+ tokens) to measure whether opaque failure rates scale directly with BPE fragmentation.
10. **State Substitution Distribution (S02):** In multi-step tracking, record 50+ errors to classify whether failures systematically favor intermediate states (midpoint/previous) vs. initial states vs. random distractors.
11. **Three-Tier Uncertainty Taxonomy (S02–S03):** Formally separate (A) First-Order Uncertainty (logprobs/entropy), (B) Explicit Metacognitive Confidence (calibration on 50+ trials), and (C) Privileged Access (PAI private state vs. non-recurrent observer).

# Open Research Questions

## Q1 — What is the right system boundary?

Does the scientifically relevant system include:

- only weights;
- weights + current activations;
- KV/recurrent state;
- external memory;
- scheduler and tools;
- environment;
- online learning process?

Different experiments may legitimately use different boundaries, but every claim must state its boundary.

## Q2 — Is persistent state more than compressed memory?

A latent vector may simply store facts more efficiently. We need tasks where historical dynamics, not factual recall, matter.

## Q3 — Is null-time evolution meaningful?

The state may:

- converge;
- oscillate;
- become noisy;
- rehearse goals;
- consolidate memory;
- invent content;
- improve future performance;
- degrade future performance.

No one of these should be assumed desirable.

## Q4 — What makes a representation higher-order?

Possible criteria:

- encodes state content;
- encodes relation between system and content;
- distinguishes own from other's state;
- supports counterfactual queries about access;
- causally controls monitoring/report;
- is available for online control.

The map should track which criterion each paper actually tests.

## Q5 — What is privileged access in a distributed system?

Is the target advantage over:

- a transcript-only observer;
- a matched external model with activations;
- another copy of the same architecture;
- a probe with equivalent compute;
- any third-party process at equal cost?

This needs an explicit comparator in every experiment.

## Q6 — When does a stable self-model become overfitting?

A system can memorize an identity description without developing an integrated self-model. Stability, generalization, causal role, and self/other binding must be tested separately.

## Q7 — Does development matter beyond training data?

Two systems can receive the same experiences in different orders. A developmental claim requires order/path effects or individual history that survives matched final data.

## Q8 — Does regulation create intrinsic goals?

A resource variable may only be another externally specified reward. Test whether it reorganizes learning, attention, planning, and self-monitoring beyond generic reward magnitude.

## Q9 — What is the relation between workspace and persistence?

Possibilities:

- workspace is reconstructed each event;
- workspace content is copied to persistent state;
- persistent state biases workspace entry;
- workspace itself is recurrent;
- separate monitor tracks workspace history.

Each architecture predicts different reset and lesion effects.

## Q10 — When should language be introduced?

Language may scaffold self-modeling, but it can also make the system imitate human introspective narratives. Level 3 should test prelinguistic/structured representations before and after language grounding.

# “Do not chase” list

Avoid spending substantial time on:

- ungrounded chatbot declarations of consciousness;
- viral single transcripts;
- benchmarks where “introspection” means answering facts from training data;
- architecture scaling without a discriminating hypothesis;
- building a polished agent UI before a valid measurement exists;
- unrestricted autonomous tool use as a substitute for temporal continuity;
- synthetic emotion demos whose main result is evocative language;
- reproductions that cannot access the relevant internal variable;
- debates that turn entirely on definitions with no experimental consequence;
- attempting to solve phenomenality directly before the component map is sharper.

Keep a parking lot for such ideas rather than deleting them.