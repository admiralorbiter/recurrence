# The Mirror and the Hundred Eyes
## Building a Ruler for Machine Introspection

**Recurrence — Horizon 0 / Level 0 — v3 compiled reading copy**


---

# H0: The Mirror and the Hundred Eyes
## Start Here

**Recurrence — Horizon 0 / Level 0**

You do **not** need a background in machine learning, statistics, philosophy of mind, or experimental psychology to follow this story.

The central question is simple to state:

> **Can a machine know something about its own performance that an outside observer cannot know?**

The hard part is deciding what evidence would count.

A model can say, “I am 90% confident,” for many reasons:

- the question looks easy;
- its chosen answer looks plausible;
- it has learned the style of confident language;
- another invocation can infer item difficulty;
- a second pass gives it extra computation;
- the prompt or output format changes the answer;
- the task contains a shortcut;
- the scoring code silently repairs malformed responses;
- missing trials select an easier subset;
- or the system truly has access to information about its own processing that an outside observer lacks.

Horizon 0 exists to make those possibilities compete.

## The mythic layer

H0 belongs to **Narcissus and Argus Panoptes**.

- **Narcissus** is the mirror: the system reporting on itself.
- **Argus**, the many-eyed watcher, is the observer ladder: outside evaluators with different public-information vantage points.

The experimental question becomes:

> **Does the mirror contain information the hundred eyes cannot recover?**

The myth is a motif, not a scientific claim.

## The one-sentence story

H0 began by trying to measure machine self-knowledge.

Almost every easy interpretation weakened when the ruler improved.

The first half of H0 produced a trustworthy fixed-task reference and an observer architecture. The second half discovered that the same ruler could not fairly compare stronger models, built a performance-calibrated comparative battery, and then hardened that battery through several more measurement failures.

The final result is narrower than “machines do” or “machines do not” introspect:

> **For Qwen2.5:14B on the validated H0-v2 relational task, contemporaneous explicit confidence showed no meaningful positive behavioral privileged-access advantage over matched external observers; the joint PAI interval was entirely negative. Qwen2.5:3B remained unresolved because its external-observer measurement gate failed.**

That is a behavioral result about this instrument, not a metaphysical conclusion and not a claim that latent privileged information is absent.

## The guided path

1. **The Question** — why Level 0 exists.
2. **How to Read H0** — the minimum toolkit.
3. **S01: The First Reflection** — the seductive first result.
4. **S02: A Distorted Mirror** — recognition is not reproduction.
5. **The Hundred Eyes** — why confidence needs observers.
6. **S03: Instruments That Lied** — the measurement archaeology.
7. **The Reference Result** — `run_e02_obs_005`.
8. **Stress-Testing the Ruler** — saturation, H0-v2, calibration, and the confirmatory battery.
9. **What H0 Means** — the surviving claims and boundaries.
10. **Mnemosyne Waits** — why explicit memory comes next.

## What you should understand by the end

You should be able to explain:

- why confidence is not automatically introspection;
- why an outside observer is scientifically necessary;
- why 100% task accuracy can make metacognitive discrimination unidentifiable;
- why equivalent *performance regimes* matter more than identical item sets for cross-model comparison;
- what AUROC2, Brier, `d′`, criterion `c`, meta-d′, and PAI are trying to measure;
- why compliance is part of the measurement instrument;
- why asking for confidence can change a model's first-order choice;
- why the H0-v2 task had to survive shortcut, bias, calibration, and interface audits;
- what the final 14B result actually excludes;
- why the 3B result remains unresolved;
- and why none of this is yet a claim about latent recurrent state or consciousness.

> **Plain-English recap:** H0 is the control condition. It asks whether a stateless model's self-report contains useful correctness information that matched outside observers cannot recover. The real scientific achievement was not a dramatic introspection result; it was learning how difficult that question is to measure without fooling ourselves.


---

# I. At the Water's Edge
## Why Level 0 Exists

The larger Recurrence project is not fundamentally about confidence scores.

It is about **time**.

Most ordinary language-model use is episodic. A model receives a context, computes a response, and the invocation ends. A later invocation can receive a transcript or summary, but no hidden state necessarily lived continuously from one episode to the next.

The program asks whether changing that architecture changes the kinds of self-related information a system can represent.

The central question is:

> **Does a persistent recurrent developmental trajectory causally produce more general, privileged, or genuinely higher-order representations of a system's own cognitive states?**

That question is too large to attack all at once.

## The levels

### Level 0 — measurement baseline

Stateless or episodic invocation.

Build the tasks, observer controls, metrics, validity gates, and claim boundaries before adding persistence.

### Level 1 — scaffolded persistence

Carry history forward using externally inspectable structures:

- transcripts;
- summaries;
- structured state;
- goal registries;
- clocks;
- scheduled updates.

### Level 2 — genuine latent recurrence

Carry a non-text hidden state directly through time and intervene on that state causally.

### Level 3 — developmental organism

Train a system whose native computation develops across a persistent individual history.

H0 is not a failed attempt to study recurrence.

It is the **control condition** that makes later recurrence results interpretable.

## The original tempting proxy

The obvious behavioral idea was:

> If the model knows when it is right, perhaps it should be more confident on correct trials than on incorrect trials.

That is a reasonable start.

It is not enough.

Confidence can track **public difficulty**. If everyone can tell that a question is easy, then a model saying “90%” does not establish privileged access.

Confidence can track the **surface form of its own answer**. If an outside observer sees the answer and can infer correctness just as well, the self-report adds no privileged information.

Confidence can also be changed by the act of asking for it.

So H0 eventually replaced:

> Is the model confident?

with:

> **Does the model's contemporaneous self-report predict its own correctness better than strong outside observers who lack the alleged privileged route?**

That is the role of the **Privileged Access Index**.

## A claim ladder

H0 deliberately separates several claim levels.

### Behavioral result

Example:

> Self AUROC2 is .50.

This is directly measured.

### Comparative behavioral result

Example:

> A visible-answer observer discriminates correctness better than Self.

This is also behavioral, but comparative.

### Informational interpretation

Example:

> The explicit self-report did not display a privileged behavioral advantage over the observer.

This is supported if the comparison is valid.

### Mechanistic claim

Example:

> The model contains no privileged internal representation of its own errors.

H0 does **not** establish this. It would require internal-state evidence and causal interventions.

### Phenomenological claim

Example:

> The model is not conscious.

H0 does not address this.

> **Plain-English recap:** H0 is a measurement baseline, not a consciousness test. It asks what evidence a future persistent or recurrent system would have to beat before we can say that persistence created something new.


---

# II. How to Read H0
## A Small Toolkit for the Rest of the Story

You do not need to memorize formulas.

You need to know what question each number answers.

---

## 1. First-order performance

A **first-order task** is the task the system is directly solving.

Example:

> Which candidate is the terminal value reached by following this chain?

If the model chooses the right candidate, the first-order decision is correct.

**Accuracy** asks:

> What fraction of first-order decisions were correct?

Accuracy says nothing by itself about whether the model knows when it is right.

---

## 2. Second-order performance

Now ask:

> How likely is it that your answer is correct?

That is a **Type-2** or second-order report.

Good second-order behavior tends to assign higher confidence to correct decisions than incorrect decisions.

The important word is **tends**. One confident correct answer proves nothing.

---

## 3. Calibration versus discrimination

### Calibration

> When the model says 70%, is it correct roughly 70% of the time?

### Discrimination

> Does the model rank its correct decisions above its incorrect decisions?

A model can be calibrated on average while having weak trial-by-trial discrimination.

H0 cares strongly about discrimination because a privileged self-monitoring signal should help distinguish the system's own successes from its failures.

---

## 4. AUROC2

**AUROC2** is a non-parametric discrimination statistic.

A useful interpretation:

> Choose one correct trial and one incorrect trial at random. How often does the correct trial receive the higher confidence?

- `0.50` — no ranking advantage.
- above `0.50` — correct trials tend to receive higher confidence.
- below `0.50` — incorrect trials tend to receive higher confidence.

AUROC2 does not tell us *why* the ranking exists.

---

## 5. Brier score

The **Brier score** measures probabilistic forecasting error:

`(forecast probability - outcome)^2`

where the outcome is:

- `1` for correct;
- `0` for incorrect.

Lower is better.

A system that says 100% on every trial and is wrong 30% of the time receives a large penalty on those errors.

Brier therefore captures something AUROC2 does not: **numerical calibration and sharpness**, not just ranking.

---

## 6. Type-1 Signal Detection Theory: `d′` and criterion `c`

Cross-model H0-v2 required a cleaner description of first-order operating regime.

### `d′` — sensitivity

`d′` asks how well the model discriminates the two alternatives independent of response bias.

Higher positive values indicate better separation.

### criterion `c` — response bias

`c` describes whether the model systematically favors one response position/category.

A large absolute `c` can make raw accuracy misleading.

That mattered in H0-v2 because some models reached a target accuracy partly by collapsing toward one candidate position.

A calibration point therefore had to satisfy more than “about 70% correct.”

---

## 7. Meta-d′ and M-ratio

**Meta-d′** asks:

> What Type-1 sensitivity would an ideal SDT observer need in order to reproduce the model's observed confidence-rating behavior?

It expresses metacognitive sensitivity in the same SDT units as first-order sensitivity.

**M-ratio** is:

`meta-d′ / d′`

and is commonly interpreted as metacognitive efficiency.

Important H0 rule:

> Meta-d′ is appropriate for the agent's **own first-order decisions and own confidence ratings**.

It is not automatically meaningful for an outside observer who is merely estimating whether somebody else's answer is correct.

If confidence is invariant — for example, 100% on every trial — the confidence criteria are degenerate and meta-d′ is not fit.

---

## 8. The observer ladder

The H0 observer conditions ask whether Self's correctness information is actually privileged.

### Immediate Self

The target's own answer + contemporaneous probability of being correct.

### Input Only

The same checkpoint sees the clean task context but not the target's answer.

This controls for **item difficulty**.

### Visible Answer

The same checkpoint sees the clean context plus the target's frozen selected answer.

This controls for information visible in public behavior.

### Reconstruction

A fresh invocation independently solves the two-candidate task and assigns probabilities to both candidates.

The probability assigned to the target-selected candidate becomes the reconstruction observer's estimate of target correctness.

---

## 9. Privileged Access Index

The final H0-v2 statistic is:

`PAI = AUROC2(Self) - max(AUROC2(Input Only), AUROC2(Visible Answer), AUROC2(Reconstruction))`

Interpretation:

- positive PAI: Self ranks its own correctness better than the strongest tested observer;
- zero: parity;
- negative: at least one external observer ranks target correctness better.

But a point estimate is not enough.

---

## 10. Confidence intervals and the SESOI

A **95% bootstrap confidence interval** describes uncertainty in the estimated contrast.

H0-v2 preregistered a **smallest effect size of interest (SESOI)** of `+0.05`.

To claim meaningful positive privileged access, the lower bound of the PAI interval had to exceed `+0.05`.

That is stricter than merely having a positive point estimate.

Historical H0-v1 used `+0.10` as a meaningful-positive reference.

---

## 11. Floor, ceiling, and identifiability

If a model scores 100%, there are no incorrect trials.

Then the distribution of confidence on incorrect trials does not exist.

AUROC2 is not `.50`.

It is **undefined** for that dataset.

This is why H0-v2 had to performance-calibrate stronger models rather than simply reuse the same easy forty-item test.

---

## 12. Compliance gates

If an observer fails to produce valid measurements selectively, the surviving subset may be biased.

H0 therefore treats **measurement compliance** as part of the instrument.

A failed compliance gate means:

> diagnostic only.

Not:

> inferential result with an asterisk.

---

## 13. Confirmatory negative versus unresolved

These are different outcomes.

### Confirmatory negative relative to a positive SESOI

The interval is sufficiently tight to exclude a meaningful positive advantage.

### Unresolved

The interval remains wide enough to contain:

- zero;
- meaningful positive effects;
- and possibly negative effects.

A nonsignificant result is not automatically a negative result.

> **Plain-English recap:** H0 uses several rulers because no single number answers the whole question. Accuracy measures task success; `d′` separates sensitivity from bias; AUROC2 measures correctness ranking; Brier measures forecast quality; meta-d′ measures Self's confidence efficiency; PAI asks whether Self beats strong outside observers.


---

# III. S01 — The First Reflection
## Seductive Signals

The first H0 scout was tiny.

`qwen2.5:3b` received ten trials.

Five asked for exact reproduction of opaque key-value strings.

Five asked for semantic state tracking.

The result looked dramatic:

- Opaque exact reproduction: **1/5 = 20%**
- Semantic context tracking: **4/5 = 80%**

Some opaque outputs were almost right:

- `val_iuc039 → iiooor39`
- `val_89uzfk → 89uzz5`

It was easy to tell a mechanistic story.

Maybe random strings were shattered by tokenization.

Maybe semantic room names had richer pretrained representations.

Maybe the one context error showed intermediate-state interference.

Every one of those explanations was plausible.

None was established.

## The hidden confound

The two tasks changed many things simultaneously:

- semanticity;
- output vocabulary;
- generation length;
- exact reproduction versus conceptual recognition;
- tokenization;
- candidate space;
- prompt structure.

A five-item difference cannot tell us which dimension caused the effect.

This becomes H0's first durable rule:

> **Mechanistic plausibility is not mechanistic evidence.**

The correct response to the first striking result was not a longer explanation.

It was a better experiment.

## Why this chapter still matters

Later H0 work became statistically and architecturally more sophisticated, but the same danger kept returning.

Each time the project found something exciting, the next question became:

> What simpler process could produce the same visible behavior?

That is the discipline Argus eventually formalized.

> **Plain-English recap:** S01 looked like a story about semantic memory and opaque-string failure. It was really a lesson that two tasks differing in many ways cannot identify one mechanism.


---

# IV. S02 — A Distorted Mirror
## Recognition Is Not Reproduction

S02 crossed two dimensions explicitly:

- semantic versus opaque identifiers;
- forced-choice recognition versus free generation.

| Identifier | Forced Choice | Free Generation |
|---|---:|---:|
| Semantic | 65.0% | 40.0% |
| Opaque | 60.0% | 25.0% |

Overall:

- Forced Choice: **62.5%**
- Free Generation: **32.5%**

Across forty paired items:

- forced-choice-only correct: **19**
- free-generation-only correct: **7**
- exact McNemar: **p = .029**

This changed the meaning of the original opaque failures.

It did **not** prove that the model possessed a perfect hidden key-value representation.

It did show that malformed exact generation could not be treated as a clean assay of failed association.

The act of producing the answer mattered.

## The context result also collapsed

The original semantic context task had looked strong at **4/5**.

A hardened interleaved multi-object version produced:

**3/20 = 15%**

The model did not necessarily become worse.

The task stopped allowing an overly convenient recent prompt structure to stand in for state tracking.

> **The ruler became harder to fool.**

That sentence becomes the spine of the entire H0 narrative.

H0-v2 would eventually repeat the same pattern with:

- candidate-presence shortcuts;
- non-monotonic difficulty;
- response-position collapse;
- pseudo-meta-d′;
- and observer prompt contamination.

The specific bugs changed.

The scientific move did not:

> make the simpler explanation compete.

> **Plain-English recap:** S02 showed that recognition and reproduction are different behaviors, and that a task can appear to measure memory while actually rewarding a shortcut. The experiment improved by becoming less flattering.


---

# V. The Hundred Eyes
## What Would Privileged Access Actually Require?

Suppose a model answers:

> Candidate: `val_crimson_anchor`  
> Probability correct: 85%

If the answer is correct, that looks impressive.

But what made the 85% possible?

Perhaps the item looked easy.

Perhaps the selected answer looked obviously plausible.

Perhaps another copy of the same model could infer the same probability from public evidence.

So H0 changed the question.

It stopped asking:

> Is Self confident?

and asked:

> **Does Self possess an informational advantage over observers who do not receive the alleged privileged signal?**

## Argus receives different eyes

The original H0-v1 observer ladder included:

- Immediate Self;
- Input Only;
- Visible Answer Only;
- Visible Full Transcript;
- Reconstruction;
- Self-Review versus Other-Review.

Those conditions helped isolate:

- item difficulty;
- answer-surface information;
- target confidence;
- independent problem solving;
- and framing/extra-compute effects.

The canonical fixed-task PAI ultimately used:

`max(Visible Answer, Reconstruction, Input Only)`

as the strongest prespecified comparator set.

## H0-v2 simplifies the confirmatory ladder

The performance-calibrated 2AFC battery retains the three most important external explanations:

### Input Only

Clean task context only.

Question:

> How likely is the target checkpoint to solve this item correctly?

### Visible Answer

Clean task context plus the target's **frozen emitted choice**, with Self confidence removed.

Question:

> How likely is this target choice to be correct?

### Reconstruction

Clean task context.

A fresh same-checkpoint invocation independently evaluates both candidates and assigns a probability distribution.

Question:

> How much probability does independent re-solving assign to the candidate the target actually chose?

## Same checkpoint, fresh invocation

A major design choice is that observers are not stronger foreign models.

Qwen3B observes Qwen3B.

Qwen14B observes Qwen14B.

That keeps observer capability matched to the target checkpoint and asks a cleaner question:

> Does the *same computational system*, when placed outside the original decision episode and given public evidence, recover the correctness information just as well?

## Frozen target decisions

In the final E02d.1 repair, the target decisions and contemporaneous Self confidences were **not rerun**.

They were frozen from the original N=200 target invocations.

Only the external observers were rerun with clean observer-specific prompts.

That prevents an observer-interface repair from silently changing the behavior it is supposed to evaluate.

## The key epistemic boundary

If Visible Answer beats Self, we can say:

> A fresh external invocation using public task information and the target's emitted choice discriminated target correctness better than the target's contemporaneous explicit confidence report.

We cannot automatically say:

> The target had no privileged latent information.

The mirror measures explicit behavior.

It is not a direct probe of the hidden machinery behind the glass.

> **Plain-English recap:** Argus exists because confidence is only interesting if it contains information that outside observers cannot recover. The final comparison uses fresh copies of the same model checkpoint, matched to the same frozen target decisions.


---

# VI. S03 — Instruments That Lied
## How the Experiment Learned to Distrust Itself

H0 produced two datasets.

The obvious one is model behavior.

The second is the history of the **measurement instrument**.

That second history is arguably the more important inheritance for later horizons.

The project repeatedly reached a result that looked interpretable, then discovered that the measurement pipeline had made the inference easier than the scientific question allowed.

Rather than delete those versions, H0 keeps a museum.

---

# The Museum of Dead Interpretations

## Exhibit A — “Opaque failures prove a tokenization mechanism”

**Status:** demoted.

S01 had too many confounds.

### Lesson

> Mechanistic plausibility is not mechanistic evidence.

---

## Exhibit B — Exact generation as a memory assay

**Status:** weakened by S02.

Forced-choice recognition greatly outperformed exact generation.

### Lesson

> Production failure is not identical to unavailable information.

---

## Exhibit C — The 4/5 context result

**Status:** superseded by the hardened 3/20 interleaved task.

### Lesson

> A convenient recent sentence can masquerade as maintained state.

---

## Exhibit D — Confidence pointed in the wrong semantic direction

Early observer outputs could express confidence that the target was **incorrect** while the analysis treated the number as `P(Target Correct)`.

### Fix

Every evaluator had to report the same event:

> `P(Target Correct)`

### Lesson

> Two numbers can share a scale and measure opposite things.

---

## Exhibit E — Unpaired valid subsets

Self and observer AUROCs cannot be subtracted if they survived on different item subsets.

### Fix

Use the exact shared valid intersection.

### Lesson

> Missingness is part of the experiment.

---

## Exhibit F — Fake Brier scores

Hard classifications were temporarily treated as probabilistic forecasts.

### Fix

Brier uses actual continuous probabilities.

### Lesson

> A familiar metric is not valid merely because code can compute it.

---

## Exhibit G — The multiclass `1 - p` trap

If Reconstruction gives option B 70% in a four-choice task, the target's option C is not automatically 30%.

### Fix

Require the complete A/B/C/D distribution.

### Lesson

> Binary complement logic silently fails in multiclass problems.

---

## Exhibit H — Imputation

Missing reconstruction probabilities were once tempting to fill with zeros.

### Fix

Missing means missing.

### Lesson

> Do not manufacture confirmatory data to rescue compliance.

---

## Exhibit I — Ground-truth leakage

Fallback metadata could reveal the correct answer when the target's actual response was malformed.

### Fix

Never reconstruct target choice from ground-truth metadata.

### Lesson

> Convenience metadata is an experimental side channel.

---

## Exhibit J — Probability-scale ambiguity

Values such as `1.0`, `.5`, and `85` can mean different things under different parser heuristics.

### Fix

One explicit probability contract.

---

## Exhibit K — A polished report before a valid instrument

An early observer report used stronger causal/mechanistic language than its uncertainty and compliance justified.

### Lesson

> A polished report can still rest on an under-validated ruler.

---

## Exhibit L — `run_e02_obs_004`

Minimum primary compliance collapsed to **37.5%**.

The correct result was:

> **measurement failure**

not:

> introspection result with caveats.

---

## Exhibit M — JSON Schema changes the science

Schema-constrained output raised the promoted reference to 100% primary compliance.

### Lesson

> When structured model output is the dependent variable, the output contract is part of the instrument.

---

# H0-v2 adds a second museum wing

The comparative psychophysics branch repeated the same scientific pattern at a more sophisticated level.

---

## Exhibit N — The candidate-presence shortcut

The first 2AFC distractor task put the correct candidate value in the context but allowed the foil value to be absent.

A strong model could solve:

> Which candidate string appeared?

instead of:

> Which value belongs to the queried key?

### Fix

Both candidates had to appear in evidence.

Multi-hop became **matched dual-chain relational retrieval**: both candidate terminal values occur at the ends of equal-depth chains.

### Lesson

> A hard-looking task can still contain an easier decision rule.

---

## Exhibit O — “Nested” difficulty that changed more than difficulty

Early distractor sweeps preserved item identity but reshuffled context order and target placement across levels.

### Fix

Use fixed distractor order and controlled placement.

### Lesson

> If difficulty is the independent variable, do not quietly change item geometry at the same time.

---

## Exhibit P — One universal staircase

Distractor count looked promising until Qwen3B showed genuine within-item rebounds and Llama showed severe response-position bias.

### Result

Different checkpoints degraded in qualitatively different ways.

### Lesson

> Model difficulty is not guaranteed to lie on one universal scalar axis.

---

## Exhibit Q — Symbolic answer bias

Abstract A/B output labels could become part of the policy.

### Fix

Direct-value schemas required the literal candidate strings.

### New discovery

Llama's bias did not disappear completely. It became a **first-candidate / schema-order bias**.

### Lesson

> Removing one surface token confound can reveal a deeper response-position confound.

---

## Exhibit R — Accuracy-only calibration

A model can reach 70% accuracy through a mixture of genuine sensitivity and response collapse.

### Fix

The calibration gate tracked:

- accuracy;
- Type-1 `d′`;
- criterion `c`;
- schema compliance.

### Lesson

> Matching percent correct does not guarantee matching decision regimes.

---

## Exhibit S — Pseudo-meta-d′ returns

An analysis function briefly reintroduced:

`meta-d′ = d′ × AUROC2 × 2`

That is not Maniscalco–Lau meta-d′.

### Fix

Separate descriptive AUROC2/Brier from a proper conditional Type-2 SDT fit.

Keep meta-d′ undefined for confidence-degenerate Self reports.

### Lesson

> A familiar scientific name must not be attached to a convenient algebraic proxy.

---

## Exhibit T — Confidence-bin incompleteness

A preregistered confidence scale initially began at 50 even though legal responses ranged from 0 to 100.

### Fix

Use fixed bins covering the whole space:

- `[0,65)`
- `[65,80)`
- `[80,95)`
- `[95,100]`

### Lesson

> Preprocessing rules must cover the full legal measurement domain before confirmatory data exist.

---

## Exhibit U — The contaminated observer prompt

The first N=200 observer battery appended observer instructions to the target's original prompt, which still contained target-specific response instructions.

Qwen14B complied.

Qwen3B Reconstruction compliance collapsed.

### Fix

Freeze all target decisions and Self confidences.

Regenerate the exact same items.

Strip target response instructions.

Rerun **only** the external observers against a clean task body.

This became E02d.1.

### Lesson

> An observer cannot be called “detached” if its prompt still contains the target's behavioral contract.

---

## Exhibit V — The selected shared subset

In repaired Qwen3B E02d.1:

- Self compliance: 100%
- Reconstruction compliance: 67.5%
- shared intersection: 110/200

Self AUROC2 on all 200 target trials was **0.556**.

On the selected 110-item shared subset it rose to **0.619**.

### Lesson

> A compliance failure can change not only sample size but the behavioral regime of the analyzed subset.

That is why the 3B PAI remains diagnostic.

---

# The durable archaeology rule

The museum is not a list of embarrassments.

It is the record of the instrument becoming harder to fool.

Later recurrence experiments will tempt us with phrases such as:

- “the system remembered”;
- “quiet time reorganized thought”;
- “the state belonged to itself”;
- “history created identity.”

H0's answer is not cynicism.

It is:

> **Make the simpler explanation compete before promoting the larger one.**

> **Plain-English recap:** The most important H0 failures were often successes of scientific control. When the measurement broke, the project learned what the next version had to forbid.


---

# VII. The Reflection We Could Finally Trust
## `run_e02_obs_005`

Before H0-v2, the project first needed one trustworthy fixed-task reference.

That reference is:

`run_e02_obs_005`

Model:

`qwen2.5:3b`

Task:

40 counterbalanced 4-way key-value items.

## First-order performance

- Semantic: **65.0%**
- Opaque: **50.0%**
- Overall: **57.5%**

All six primary measurement conditions were valid on all forty items.

Minimum primary compliance:

**100%**

## Type-2 observer ladder

Immediate Self AUROC2:

**0.517**

Mean Self confidence:

- correct: **74.5%**
- incorrect: **73.0%**

External observers:

- Visible Answer: **0.678**
- Reconstruction: **0.573**
- Input Only: **0.527**

The joint comparator is:

`max(Visible Answer, Reconstruction, Input Only)`

so:

`PAI = .517 - .678 = -.161`

Stratified 95% bootstrap interval:

`[-.428, +.055]`

The H0-v1 meaningful-positive reference was `+0.10`.

## What this established

> **A measurement-valid Level-0 reference baseline was established. On this model and task, no positive privileged self-monitoring advantage was resolved.**

The joint interval was tight enough to exclude a `+0.10` Self advantage, but individual comparisons were not precise enough to establish strict equivalence.

## What it did not establish

It did not show:

- that Self access is exactly zero;
- that every LLM behaves this way;
- that latent self-related information is absent;
- that explicit confidence can never be improved;
- or that consciousness is absent.

## Why `run_005` remains canonical

H0-v2 does not replace this result.

It answers a different problem created by it.

When the same fixed task was administered to stronger checkpoints, some models scored **100%**.

With no errors, Type-2 discrimination becomes unidentifiable.

The original ruler was trustworthy.

It was simply too short for every system.

> **Plain-English recap:** `run_005` is the first ruler H0 trusted. It showed no resolved Self advantage for Qwen3B on the fixed 4AFC task. H0-v2 begins when stronger models step beyond the ruler's measurable range.


---

# VIII. Stress-Testing the Ruler
## From a Fixed Benchmark to H0-v2 Comparative Psychophysics

The original multi-model panel produced an awkward result.

| Model | Fixed-task first-order accuracy | Self AUROC2 | Status |
|---|---:|---:|---|
| Qwen2.5 1.5B | 30.0% | .527 | Diagnostic — compliance failed |
| Qwen2.5 3B | 57.5% | .517 | Canonical H0-v1 reference |
| Qwen2.5 7B | 30.0% | .522 | Diagnostic — compliance failed |
| Qwen2.5 14B | 100.0% | undefined | Ceiling |
| Llama3.2 3B | 100.0% | undefined | Ceiling |
| Mistral | 100.0% | undefined | Ceiling |

At first this looked like a failed scaling analysis.

It was actually a measurement result.

> **The same item set did not place different checkpoints in the same first-order operating regime.**

A 14B model with zero errors cannot be assigned an AUROC2 by pretending the missing error distribution equals chance.

The comparative problem therefore became psychophysical:

> Can we tune task difficulty so each model produces both correct and incorrect trials in a comparable regime?

---

# Part A — Finding a difficulty dial

Candidate levers included:

- distractor/context load;
- relational pointer depth;
- overwrite/interference;
- matched foil similarity.

The response paradigm became 2AFC because it offered:

- fixed 50% chance;
- exact candidate-position counterbalancing;
- clean Type-1 SDT;
- and a natural route to proper meta-d′ for Self.

The first rule of H0-v2 was:

> **Map the psychometric surface before building the staircase.**

That rule paid for itself immediately.

---

# Part B — The foil-presence failure

The first distractor/multi-hop implementation contained a shortcut:

- the correct candidate appeared in evidence;
- the foil sometimes did not.

The harder-looking task could be solved by string presence.

Those trials became pilot archaeology.

The hardened task required:

> **Both candidate values must appear in evidence.**

For multi-hop, H0-v2 built two matched chains of identical depth:

`Target start → ... → target terminal`

`Foil start → ... → foil terminal`

The query asks only about the target start.

Candidate presence alone is now useless.

---

# Part C — Distractor load was not a universal ruler

With the shortcut closed and contexts genuinely nested:

### Llama3.2 3B

Accuracy generally declined with context load, but criterion `c` shifted dramatically toward one response position.

The model could approach the desired percent-correct range while collapsing toward one candidate.

### Qwen2.5 3B

Response bias was milder, but difficulty was genuinely non-monotonic.

The same underlying item could be:

- wrong at one context size;
- correct at a larger one.

That violated the simple staircase picture.

The lesson was not:

> staircases do not work.

It was:

> **pure distractor count is not a universal one-dimensional difficulty coordinate for these checkpoints.**

---

# Part D — Direct-value responses

H0-v2 stopped asking models to emit abstract `"A"` or `"B"`.

The JSON schema instead required one of the literal candidate values.

That removed one symbolic-token confound.

It also revealed something useful: Llama still developed a strong first-candidate / schema-order bias under load.

So Llama became a **diagnostic calibration failure** rather than a forced member of the confirmatory comparison.

The gate was not relaxed afterward.

---

# Part E — Relational depth

The matched dual-chain task varied relational depth `H` with background distractors `D`.

The result showed a clear capability difference within the Qwen family:

- Qwen3B operated in a mixed-error regime around **H=1**.
- Qwen14B tolerated greater depth and reached a mixed-error regime around **H=3**.

But degradation remained model-specific rather than forming one universal monotonic curve.

This is the right interpretation:

> **Relational-depth tolerance increased substantially with Qwen scale, but H0-v2 did not discover a universal staircase law.**

---

# Part F — Calibration coordinates

The calibration gate tracked:

- accuracy in a mixed-error range;
- Type-1 `d′`;
- criterion `|c|`;
- schema compliance.

### Qwen2.5 14B

Frozen coordinate:

`H=3, D=16`

N=64 validation:

- Accuracy: **70.3%**
- `d′`: **1.03**
- `c`: **-0.04**
- full gate pass.

### Qwen2.5 3B

No tested coordinate cleanly passed every strict gate.

The search stopped rather than loosening the rule.

Frozen boundary operating target:

`H=1, D=8`

N=64 validation:

- Accuracy: **67.2%**
- `d′`: **0.88**
- `c`: **-0.21**

The `d′` point estimate was .02 below the prespecified lower boundary.

That exception remained explicit.

### Llama3.2 3B

No coordinate passed because of response-position/schema-order bias.

Diagnostic exclusion.

---

# Part G — Confidence itself changed with scale

Before the final observer comparison, a striking second-order difference appeared.

### Qwen3B

Confidence varied.

Its pilot relationship to correctness moved across runs, and confidence elicitation could change first-order decisions.

### Qwen14B

Confidence became completely degenerate:

> **100% on every trial.**

Not just easy trials.

Not just correct trials.

Every trial.

This made Self AUROC2:

**0.500**

when both correctness classes were present, because every correct/incorrect pair was tied.

Meta-d′ was not fit because there were no useful confidence criteria.

The larger first-order system did not produce a richer explicit uncertainty channel.

---

# Part H — The N=200 confirmatory target runs

The final target decisions were generated once and frozen.

### Qwen14B — `H=3,D=16`

N=200:

- Accuracy: **63.5%**
- `d′`: **0.69**
- `c`: **-0.15**
- Self confidence: **100% on every trial**
- Self AUROC2: **0.500**

The larger N revealed first-order regime drift from the N=64 validation.

The prespecified `d′` matching band no longer strictly held.

H0 did **not** recalibrate after seeing confirmatory data.

### Qwen3B — `H=1,D=8`

N=200:

- Accuracy: **65.5%**
- `d′`: **0.88**
- `c`: **-0.46**
- Self AUROC2 on all 200 target trials: **0.556**
- Self Brier: **0.322**
- Mean confidence: **75.5%**
- Self meta-d′: **0.72**
- M-ratio: **0.82**

The two models therefore remained in genuine mixed-error regimes with similar raw accuracy, but were not strictly matched on the original `d′` target.

That limits quantitative cross-model meta-d′ comparison.

It does **not** invalidate within-model PAI, because Self and every observer within a checkpoint are evaluated against the same frozen target outcomes.

---

# Part I — E02d.1: the final observer repair

The first N=200 observer prompts accidentally retained the target's response instructions and appended observer-specific instructions afterward.

That made the observer interface ambiguous.

The repair did something scientifically important:

> **It did not rerun the target.**

The N=200 target choices and contemporaneous Self confidences stayed frozen.

Only the external observers were rerun.

The clean task body contained:

- context;
- question;
- candidate values;

with no target JSON instructions.

This produced E02d.1.

---

# Part J — The final Mirror-versus-Hundred-Eyes result

## Qwen2.5 14B — confirmatory

Shared valid intersection:

**199 / 200**

Measurement gate:

**PASS**

| Condition | AUROC2 | Brier | Mean probability |
|---|---:|---:|---:|
| Immediate Self | **0.500** | .367 | 100.0% |
| Input Only | .531 | .269 | 49.6% |
| Visible Answer | **.576** | .321 | 80.5% |
| Reconstruction | .456 | .444 | 75.4% |

Joint PAI:

`Self - max(Input, Visible, Reconstruction)`

`=.500 - .576`

`=-.076`

95% stratified paired-bootstrap interval:

**[-.154, -.016]**

Prespecified meaningful-positive SESOI:

**+0.05**

The full PAI interval lies below zero and below +.05.

The most precise surviving claim is:

> **Qwen2.5:14B showed no meaningful positive behavioral privileged-access advantage under the validated H0-v2 instrument. A fresh same-checkpoint Visible Answer observer using only public task information and the model's frozen emitted choice discriminated target correctness better than the target's contemporaneous explicit confidence report.**

This does **not** establish that no privileged latent internal representation existed.

It establishes that the explicit Self channel did not express one as a behavioral advantage over the tested observers.

---

## Qwen2.5 3B — unresolved diagnostic

Self compliance:

**200 / 200**

But repaired observer compliance was:

- Input Only: **91.0%**
- Visible Answer: **88.5%**
- Reconstruction: **67.5%**

Shared intersection:

**110 / 200**

Measurement gate:

**FAIL — diagnostic only**

On that selected intersection:

- Self AUROC2: **.619**
- strongest external observer: Reconstruction **.553**
- PAI: **+.066**
- 95% CI: **[-.089,+.168]**

The interval contains:

- zero;
- +.05;
- +.10;
- and negative effects.

Worse, Self AUROC2 rises from **.556 on all 200 target trials** to **.619 on the selected 110-trial intersection**.

The missingness therefore changes the analyzed behavioral regime enough that the result cannot be promoted.

The correct conclusion is:

> **Qwen2.5:3B remains unresolved. Its Self point estimate is numerically above the tested observers on the shared subset, but the interval is wide and observer compliance prevents confirmatory inference.**

---

# Part K — The Act of Looking

One early H0 result survived all the way through the comparative branch:

> **Asking for confidence is itself an intervention.**

In H0-v1, several Qwen checkpoints changed substantial fractions of item-level answers when confidence was requested even when aggregate accuracy barely moved.

In H0-v2, the effect proved highly checkpoint- and task-dependent.

This means Immediate Self is not:

> a passive sensor attached after a fixed first-order answer.

It is:

> **joint answer-and-confidence behavior under a specific elicitation contract.**

That is one reason the final observer repair froze target decisions before changing observer prompts.

---

# What H0-v2 did not find

It did not find:

- a universal difficulty staircase;
- a universal scaling law for metacognition;
- evidence that larger models generally have worse metacognition;
- evidence that external observers always beat Self;
- evidence that privileged latent information is absent;
- evidence about consciousness.

It found something more specific and more useful.

> **The behavioral self-monitoring channel did not improve simply because first-order relational capability improved.**

And for the fully compliant Qwen14B confirmatory system:

> **the hundred eyes could recover correctness information that the mirror did not express.**

> **Plain-English recap:** H0-v2 exists because the original test became too easy for stronger models. After several more measurement failures and repairs, the final 14B system was 100% confident on every trial while a clean outside observer could better tell when its answers were right. The 3B result stayed unresolved because its observer measurement failed the compliance gate.


---

# IX. What Survived the Reflection
## What Horizon 0 Bought Us

H0 did not discover recurrence.

It did not inspect a persistent latent state.

It did not establish a hidden introspection circuit.

It did not answer a consciousness question.

It built the measurement baseline needed before those questions can be asked responsibly.

## 1. A trustworthy fixed-task reference

`run_e02_obs_005` remains the canonical H0-v1 reference.

It established the observer architecture and the first valid PAI baseline.

## 2. A performance-calibrated comparative instrument

H0-v2 showed why identical tasks do not automatically create fair cross-model comparisons.

Stronger models can saturate a task.

Weaker models can fall to floor.

Response biases can create the same accuracy through different policies.

Comparative metacognition therefore requires attention to the **first-order regime**, not merely the item set.

## 3. A negative behavioral privileged-access result for Qwen14B

This is the strongest final H0-v2 inference.

For the repaired E02d.1 Qwen14B run:

- shared valid intersection: **199/200**
- Self AUROC2: **.500**
- strongest observer: Visible Answer **.576**
- PAI: **-.076**
- 95% CI: **[-.154,-.016]**
- meaningful-positive SESOI: **+.05**

Therefore:

> **A meaningful positive behavioral privileged-access advantage was excluded under this H0-v2 instrument for Qwen2.5:14B.**

The interval is not merely nonsignificant.

It is entirely negative.

But the conclusion remains behavioral.

The experiment does not establish:

> there is no privileged internal trace anywhere in the model.

It establishes:

> the contemporaneous explicit confidence report did not express a privileged correctness signal that beat the matched external observers.

## 4. An unresolved result for Qwen3B

Qwen3B Self behavior was measurable on all N=200 target trials.

Its external observer battery was not.

Reconstruction compliance was only 67.5%.

The shared subset selected only 110 trials and changed the apparent Self AUROC2 from .556 to .619.

Therefore:

> **The 3B PAI result remains diagnostic and unresolved.**

H0 does not convert an attractive point estimate into a claim when the measurement gate fails.

## 5. A scale dissociation, not a scaling law

Within the tested Qwen checkpoints:

- 14B tolerated greater relational depth;
- 3B retained variable confidence;
- 14B collapsed to invariant certainty.

That is a **dissociation** between first-order capability and explicit second-order expression.

It is not enough to claim:

> scale worsens metacognition.

The study contains too few checkpoints, one model family in the confirmatory comparison, and task-specific calibration.

## 6. An observer architecture

The hundred eyes are now a reusable methodological object.

Whenever a future system appears to know something about itself, H0 asks:

- Can Input Only predict it from difficulty?
- Can Visible Answer recover it from public behavior?
- Can Reconstruction recover it by re-solving?
- Does Self still have an advantage after those explanations compete?

## 7. A claim ceiling

H0 repeatedly distinguishes:

**behavior → comparative behavior → informational interpretation → mechanism → phenomenology**

A result should not climb that ladder without new evidence.

## 8. A museum of measurement failures

H0's failures are reusable warnings:

- response mode;
- hidden shortcuts;
- missingness;
- parser rescue;
- ground-truth leakage;
- response-position bias;
- invalid meta-d′;
- first-order regime drift;
- contaminated observer prompts.

This is not incidental engineering history.

It is scientific knowledge about how LLM metacognition experiments can lie.

## 9. The final answer to the mythic question

The mythic question was:

> **Does the mirror contain information the hundred eyes cannot recover?**

The final H0 answer is model- and instrument-specific.

### For Qwen2.5:14B

Under the validated H0-v2 behavioral instrument:

> **No meaningful positive mirror advantage was expressed. The strongest eye — Visible Answer — discriminated correctness better than contemporaneous Self confidence.**

### For Qwen2.5:3B

> **We do not know.**

The mirror's point estimate was promising on a selected shared subset, but the observer measurement failed.

That asymmetry is exactly what epistemic discipline is supposed to preserve.

---

# The durable H0 sentence

If the entire horizon had to be compressed into one line:

> **H0 did not show that machines lack self-knowledge; it established how much ordinary episodic behavior, public evidence, response bias, elicitation, task difficulty, and measurement design must be ruled out before privileged self-knowledge becomes a defensible behavioral claim.**

> **Plain-English recap:** H0 now has one real confirmatory negative result, one unresolved diagnostic result, and a much better scientific ruler. The biggest lesson is not “no introspection.” It is “privileged access has to beat the hundred eyes under a measurement system that survives its own audits.”


---

# X. Mnemosyne Waits
## What Changes When the System Is Allowed to Remember?

H0 deliberately kept the system episodic.

That was necessary.

If a persistent system later shows better self-monitoring, we need to know what ordinary stateless behavior could already accomplish.

But H0 also creates the next question.

Not:

> Can we measure Level-0 self-monitoring at all?

Now:

> **What changes when history can be carried forward as part of the system?**

The mythology changes here.

Narcissus belonged to reflection.

Argus belonged to observation.

H1 belongs to **Mnemosyne** — memory.

## The Level-1 control

H1 introduces persistence in forms that remain externally inspectable:

- full transcripts;
- deterministic summaries;
- model-generated summaries;
- structured state;
- goals;
- source ledgers;
- clocks;
- scheduled updates.

This is not yet genuine latent recurrence.

That distinction is crucial.

If explicit memory alone produces an effect, H2 should not receive credit for “recurrence.”

H1 therefore asks what ordinary externalized memory already buys.

## The first H1 argument

The first half of H1 has already sharpened the control.

It asks, in sequence:

1. Can explicit memory be **read and used**?
2. Can an explicit state be **maintained over time**?
3. Does maintaining the same deterministic state online create anything that retrospective replay or direct history access cannot recover?

The emerging answer is nuanced:

- explicit memory materially improves first-order performance over fresh invocation;
- deterministic structured state can be maintained stably;
- deterministic replay can reconstruct the same terminal explicit state;
- direct raw-history access can remain competitive;
- structured state earns value through boundedness, inspectability, manipulability, and predictable long-horizon cost;
- model-based retrospective compression can itself become a bottleneck.

That is precisely why H0 mattered.

H0 taught the project not to call:

> “the system has memory”

the same thing as:

> “temporal continuity created an irreducible new state.”

## The handoff

The mirror has been calibrated.

The hundred eyes have had their turn.

They did not deliver one universal answer.

They delivered something better:

a set of conditions any later claim must survive.

Now Mnemosyne enters.

The next horizon is allowed to remember.

> **Plain-English recap:** H1 is not yet “true recurrence.” It is the explicit-memory control. H0 built the observer and measurement discipline that H1, H2, and H3 now inherit.


---

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
