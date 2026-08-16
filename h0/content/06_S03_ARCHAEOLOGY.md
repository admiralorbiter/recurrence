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
