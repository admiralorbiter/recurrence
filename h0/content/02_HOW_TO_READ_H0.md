# II. How to Read H0
## A Small Toolkit for the Rest of the Story

This page introduces the few concepts needed to understand the experiments.

You do not need to memorize the formulas.

The goal is to know what question each number answers.

---

# 1. First-order performance

A **first-order task** is the task the system is directly trying to solve.

Example:

> Which option contains the correct value for this key?

If the model chooses the correct option, the first-order decision is correct.

If it chooses the wrong option, the first-order decision is incorrect.

**First-order accuracy** simply asks:

> What fraction of the questions did the model answer correctly?

If the model gets 23 of 40 correct:

`23 / 40 = 57.5%`

That tells us how good the model is at the task.

It does **not** tell us whether the model knows when it is right.

---

# 2. Second-order or metacognitive performance

Now add a second question:

> How likely is it that your answer is correct?

That answer is about the model's **own first-order decision**.

Suppose two trials look like this:

| Trial | Answer correct? | Confidence |
|---|---|---:|
| A | Yes | 90% |
| B | No | 20% |

That is good metacognitive behavior.

Now imagine:

| Trial | Answer correct? | Confidence |
|---|---|---:|
| A | Yes | 90% |
| B | No | 95% |

The model is confident in both cases. Its confidence does not help us distinguish success from failure.

That distinction is why H0 needs more than ordinary accuracy.

---

# 3. Calibration and discrimination are different

These two ideas are easy to confuse.

## Calibration

Calibration asks:

> When the system says "70%," is it correct about 70% of the time?

A perfectly calibrated system can still be poor at separating individual correct and incorrect trials.

## Discrimination

Discrimination asks:

> Does the system tend to assign higher confidence to its correct answers than to its incorrect answers?

H0 cares especially about discrimination because privileged access should give the target useful trial-by-trial information about its own success.

---

# 4. AUROC2

**AUROC2** is one way to measure metacognitive discrimination.

A useful intuitive interpretation is:

> Randomly choose one correct trial and one incorrect trial. How often did the system give the correct trial the higher confidence?

If the system has no useful ranking ability, the score tends toward **0.50**.

If it perfectly ranks every correct trial above every incorrect trial, the score is **1.00**.

Very roughly:

- `0.50` — chance-like discrimination
- `0.60` — some useful separation
- `0.70` — stronger separation
- `1.00` — perfect separation

There is no universal magical cutoff for "introspection."

AUROC2 is simply a measurement tool.

## Why call it AUROC2?

"ROC" means receiver operating characteristic.

The "2" indicates a **Type-2** or second-order task: we are judging whether confidence predicts whether the first-order decision was correct.

---

# 5. Brier score

The **Brier score** asks a different question:

> How numerically accurate were the probability forecasts?

For each trial:

`Brier contribution = (predicted probability - actual outcome)^2`

The actual outcome is:

- `1` if the answer was correct
- `0` if the answer was wrong

Example:

The model says 90% and is correct:

`(0.90 - 1)^2 = 0.01`

Very small error.

The model says 90% and is wrong:

`(0.90 - 0)^2 = 0.81`

Very large error.

For Brier score:

> **Lower is better.**

AUROC2 and Brier tell us different things.

- AUROC2 asks about **ranking correct above incorrect**.
- Brier asks about **probability accuracy**.

---

# 6. Confidence intervals

Every experiment samples a limited number of trials.

If we repeated the experiment with different items, the exact result would move.

A **confidence interval** is a way to represent that uncertainty.

For the final H0 joint PAI:

`PAI = -0.161`

with a 95% bootstrap interval:

`[-0.428, +0.055]`

The point estimate is negative.

But the interval reaches slightly above zero.

So the careful interpretation is not:

> "The true effect is definitely negative."

It is:

> "The experiment did not resolve a positive self-advantage, and the range of plausible values under this analysis does not include a large positive effect."

That distinction becomes important later.

---

# 7. What is a bootstrap?

A **bootstrap** is a resampling technique.

Imagine the forty observed trials are slips of paper in a bag.

We repeatedly create new synthetic datasets by drawing from those observed trials with replacement.

Each synthetic dataset produces a new estimate.

The spread of those estimates gives us a sense of how unstable the result is at this sample size.

H0 uses a stratified paired bootstrap so that:

- self and observer remain compared on the same items;
- correct and incorrect classes remain represented.

You do not need the implementation details to read the main result.

The key idea is:

> The confidence interval acknowledges that forty trials do not reveal an effect with infinite precision.

---

# 8. Observer controls

Why not compare the model's confidence to chance and stop?

Because some information about correctness is public.

An outside observer may infer that:

- the question is easy;
- the selected answer fits the prompt;
- the response contains a suspicious pattern;
- or the target appears uncertain.

So H0 creates several observers with different information.

The observer ladder asks:

> How much of the apparent self-knowledge can be reconstructed without privileged access?

This is analogous to adding control groups in a biological or psychological experiment.

---

# 9. Privileged Access Index — PAI

The H0 Privileged Access Index is:

`PAI = Self score - strongest prespecified public/reconstructive comparator`

For the final reference:

- Self AUROC2 = `0.517`
- Visible Answer = `0.678`
- Reconstruction = `0.573`
- Input Only = `0.527`

The strongest comparator is Visible Answer.

So:

`PAI = 0.517 - 0.678 = -0.161`

A positive number would mean the target discriminated its own correctness better than those observers.

A negative number means at least one observer performed better.

## What PAI does not mean

PAI is **not**:

- a consciousness score;
- a measure of intelligence;
- a measure of self-awareness in everyday language;
- proof that an internal state exists or does not exist.

It is a narrow experimental comparison.

---

# 10. Compliance

A measurement only exists if the system actually produces a valid measurement.

Suppose we ask for a probability from 0 to 100.

Some outputs might be:

- `85`
- `-7`
- `"probably high"`
- malformed JSON
- missing entirely

If invalid outputs are silently repaired or dropped, the remaining dataset can become biased.

So H0 tracks **measurement compliance**.

The final promoted run required every primary condition to reach at least 90% valid measurement.

`run_e02_obs_005` reached **100%**.

Earlier runs failed this gate and were not treated as confirmatory evidence.

---

# 11. Floor and ceiling effects

Metacognitive discrimination requires both:

- some correct trials;
- some incorrect trials.

If a model gets **100% correct**, we cannot ask whether confidence separates correct from incorrect trials because there are no incorrect trials.

AUROC2 is then **undefined**, not 0.50.

Likewise, if a model gets almost everything wrong, the task is not a useful operating regime.

This is why the later cross-model panel teaches an important lesson:

> The same fixed test is not automatically a fair comparison across different model capabilities.

---

# 12. The reading rule for H0

Whenever you see a result, ask four questions:

1. **What behavior was actually measured?**
2. **What simpler explanation could produce the same behavior?**
3. **What control was introduced to rule that explanation out?**
4. **What conclusion still survives after the control?**

That is the logic of the entire horizon.
