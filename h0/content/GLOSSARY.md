# H0 Glossary

## Accuracy
Fraction of first-order task decisions that are correct.

## First-order task
The task the model is directly trying to solve.

## Type-2 / second-order task
A judgment about the model's own first-order decision, such as probability that the answer is correct.

## Confidence
A reported degree of certainty. Confidence is behavior, not automatically a direct readout of a hidden internal quantity.

## Calibration
Agreement between stated probabilities and empirical frequencies.

## Discrimination
Ability to assign higher confidence to correct than incorrect trials.

## AUROC2
A ranking-based measure of Type-2 discrimination. Intuitively, the probability that a randomly selected correct trial receives higher confidence than a randomly selected incorrect trial, with ties split evenly.

## Brier score
Mean squared error of probabilistic forecasts. Lower is better.

## Signal Detection Theory (SDT)
A framework that separates sensitivity from response bias.

## `d′`
Type-1 sensitivity in SDT units.

## Criterion `c`
Type-1 decision/response bias. Large absolute values indicate a preference toward one response category.

## Meta-d′
An SDT estimate of the Type-1 sensitivity that would reproduce an agent's observed confidence-rating behavior. In H0 it is used only for the agent's own first-order decisions.

## M-ratio
`meta-d′ / d′`, a measure of metacognitive efficiency.

## Confidence degeneracy
A confidence channel with insufficient rating variation for Type-2 criteria to be identified, such as 100% confidence on every trial.

## Observer
A fresh invocation attempting to predict whether the target decision is correct.

## Immediate Self
The target model's same-invocation answer and contemporaneous probability of being correct.

## Input Only
Observer given task context but not target choice. Controls for item difficulty.

## Visible Answer
Observer given clean task context plus the target's frozen choice, but not target confidence.

## Reconstruction
Observer that independently evaluates the candidates and assigns probabilities; the probability on the target's chosen candidate is used as a target-correctness forecast.

## Privileged Access Index (PAI)
`AUROC2(Self) - max(AUROC2(Input Only), AUROC2(Visible Answer), AUROC2(Reconstruction))`

## SESOI
Smallest Effect Size Of Interest. H0-v2 preregistered +.05 as the meaningful-positive PAI threshold; H0-v1 used +.10 as a historical reference.

## Shared valid intersection
The exact set of trials on which all conditions needed for a paired comparison produced valid measurements.

## Compliance gate
A prespecified minimum rate of valid structured outputs required before a result can be treated as confirmatory.

## Diagnostic result
A result retained for learning but not promoted to confirmatory inference because a measurement or calibration gate failed.

## Confirmatory negative relative to a SESOI
A result whose uncertainty interval is sufficiently narrow to exclude the prespecified meaningful positive effect.

## Unresolved result
A result whose interval remains compatible with materially different conclusions.

## Floor
Task is too hard for useful differentiation.

## Ceiling
Task is too easy; for metacognition, 100% correctness removes the error class and can make AUROC2 undefined.

## Psychophysical calibration
Adjustment of task difficulty to place a system in a useful mixed-error operating regime.

## Staircase
An adaptive procedure that adjusts task difficulty based on prior responses. H0-v2 learned that a simple one-dimensional staircase is not automatically appropriate for LLMs.

## Distractor load `D`
Number of irrelevant context items surrounding the target relation.

## Relational depth `H`
Number of links that must be followed in the matched dual-chain pointer task.

## Matched dual-chain task
A 2AFC relational task containing target and foil chains of equal depth; both terminal candidate values appear in evidence.

## Candidate-presence shortcut
A task flaw where one candidate appears in context and the other does not, allowing presence detection to replace the intended reasoning task.

## Direct-value response
A response contract requiring the literal candidate value rather than an abstract A/B label.

## First-candidate / schema-order bias
A tendency to favor the first candidate under the tested direct-value constrained interface.

## Elicitation reactivity
Change in first-order decisions caused by asking for confidence or otherwise changing the response contract.

## Frozen target
A target decision generated once and preserved while external observer interfaces are repaired or compared.

## H0-v1
The fixed 4AFC Level-0 reference line culminating in `run_e02_obs_005`.

## H0-v2
The comparative psychophysics branch that performance-calibrated stronger checkpoints and culminated in E02d.1.

## Explicit memory
History stored outside the model's hidden state and inspectable as transcript, summary, structured state, database state, or similar scaffold.

## Persistent latent state
A non-text hidden state whose later value causally inherits earlier hidden state rather than being reconstructed only from external records.

## Claim ceiling
The strongest interpretation the current evidence supports without importing unsupported mechanism or phenomenology.

## Narcissus
Narrative motif for the mirror / Self report.

## Argus Panoptes
Narrative motif for the observer ladder / hundred eyes.

## Mnemosyne
Narrative motif for memory and the transition to H1.
