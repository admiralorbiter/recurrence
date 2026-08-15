# H0 Reader's Glossary

These definitions are written for a curious non-specialist and match the explanations used by the website.

## AUROC2

**Short version:** A score for how well confidence ranks correct answers above incorrect answers.

Imagine randomly choosing one correct trial and one incorrect trial. AUROC2 is closely related to how often the correct trial receives higher confidence. About 0.50 is chance-like ranking; 1.00 is perfect separation. It is undefined if there are no correct or no incorrect trials.

## bootstrap

**Short version:** A resampling method used to estimate how much an effect might vary across samples.

The observed trials are repeatedly resampled with replacement to create many synthetic datasets. H0 uses paired and stratified resampling so self and observer stay matched on the same items and both correctness classes remain represented.

## Brier

**Short version:** A probability-accuracy score; lower is better.

For each trial, the Brier contribution is (predicted probability − actual outcome)^2, where the outcome is 1 for correct and 0 for incorrect. Confidently wrong forecasts are penalized heavily.

## calibration

**Short version:** Whether stated probabilities match long-run success frequencies.

A calibrated system that says 70% on many trials should be correct about 70% of the time. Calibration is different from discrimination: a system can be well calibrated overall yet poor at ranking which individual answers are correct.

## ceiling

**Short version:** A task is too easy for a system, leaving too few or no errors to distinguish stronger performance.

If a model gets 100% correct, AUROC2 cannot be estimated because there are no incorrect trials. The same fixed test may therefore be unsuitable for cross-model metacognitive comparison.

## compliance

**Short version:** Whether the model produced a measurement that satisfies the experiment's required format and range.

If a probability is malformed, missing, or outside the allowed range, the measurement may be invalid. H0 uses a hard compliance gate so a run with unreliable measurements cannot become a confirmatory baseline.

## confidence

**Short version:** The system's reported probability that its own answer is correct.

Confidence is useful data, but it is not automatically introspection. It may reflect task difficulty, learned language patterns, public answer cues, or the prompting format itself.

## confidence interval

**Short version:** A range that communicates uncertainty in an estimated effect.

H0 uses bootstrap confidence intervals. The exact effect estimate would change with a different sample of items. The interval makes that uncertainty visible and helps prevent a point estimate from being treated as exact truth.

## confound

**Short version:** Another variable that changes along with the variable of interest, making the cause of a result ambiguous.

S01 compared tasks that differed in semanticity, response mode, output length, candidate space, and prompt structure. That meant the observed performance difference could not be attributed cleanly to any single mechanism.

## discrimination

**Short version:** Whether confidence tends to be higher on correct trials than on incorrect trials.

Metacognitive discrimination is the trial-by-trial separation between correct and incorrect answers. H0 measures this mainly with AUROC2.

## explicit memory

**Short version:** Externally stored, inspectable history such as a transcript, summary, or structured state object.

H1 studies explicit memory before hidden recurrence so the project can determine what ordinary access to history already solves.

## first-order

**Short version:** The task the system is directly trying to solve, such as choosing the correct answer.

First-order performance is ordinary task performance. In H0, that usually means whether the model selected the correct key-value option. It is distinct from a second-order judgment about whether that answer is correct.

## floor

**Short version:** A task is too hard for a system, leaving performance near the minimum useful level.

Near-floor performance can also make metacognitive comparison unstable or uninterpretable. Comparative psychophysics tries to place systems in a shared mixed-error regime.

## forced choice

**Short version:** The model selects from a fixed set of candidate answers instead of generating the answer freely.

Forced choice shifts the task toward recognition and away from exact surface reproduction. H0 used a counterbalanced four-option task as the main Level-0 substrate.

## latent state

**Short version:** A non-text internal representation carried by the system, such as a recurrent hidden state.

H2 is designed to manipulate genuine hidden state while controlling visible context. That is necessary for causal claims about recurrence beyond explicit memory.

## Level 0

**Short version:** The episodic reference condition: no experimental persistent memory or latent recurrent state is added.

Level 0 is the baseline architecture used before the project introduces explicit memory (H1) or persistent hidden state (H2). It tells us what the ordinary episodic system can already do.

## McNemar

**Short version:** A paired test focused on cases where two conditions disagree.

For the same items under two conditions, McNemar's test compares how often condition A alone succeeds versus condition B alone succeeds. It is useful when outcomes are paired correct/incorrect judgments.

## metacognition

**Short version:** A judgment about another cognitive process—here, whether the system can monitor the likely correctness of its own answer.

H0 operationalizes metacognition narrowly. It does not assume a philosophical theory of self-awareness. It asks whether confidence or another second-order report tracks first-order success.

## observer

**Short version:** A matched outside evaluator asked to predict whether the target model's answer is correct.

Observers receive different information vantages: prompt only, prompt plus answer, full transcript, independent reconstruction, or second-pass review. They are controls for simpler public-information explanations.

## PAI

**Short version:** Privileged Access Index: Self discrimination minus the strongest prespecified public/reconstructive comparator.

PAI asks whether the target's self-monitoring contains information that strong observers cannot reconstruct from public evidence. A positive PAI is necessary for a behavioral privileged-access claim, but would still not prove a causal introspection mechanism.

## persistent state

**Short version:** State whose later value causally depends on its earlier value rather than being reconstructed only from an external record.

The project reserves strong persistence claims for systems where intervening on prior state changes later behavior while current observations and explicit memory are held fixed.

## reconstruction

**Short version:** An independent solve that estimates a full probability distribution over the answer options.

The reconstruction observer does not simply accept the target's answer. It solves the task independently and assigns probability to A/B/C/D. The probability assigned to the target's chosen option becomes its estimate of target correctness.

## SESOI

**Short version:** Smallest Effect Size of Interest: the minimum effect the project decided would count as meaningfully positive.

H0 used +0.10 AUROC2 as a meaningful positive PAI margin in the joint statistic. This helps distinguish 'not statistically different from zero' from 'large enough to matter'.

## shortcut

**Short version:** A feature that lets the model score well without using the cognitive ability the researcher intended to measure.

For example, a state-tracking task may accidentally be solvable by repeating the most recent location rather than tracking all transitions. High accuracy would then overstate the intended ability.
