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
