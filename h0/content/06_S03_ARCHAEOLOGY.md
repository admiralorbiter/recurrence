# VI. Instruments That Lied
## S03 — Why the Final Result Required Several Failed Versions

The S03 story is unusual.

The important result is not only what the model did.

It is how often the experiment itself produced an answer that looked more trustworthy than it was.

H0 eventually adopted a rule:

> **If a measurement failure changes what can be claimed, that failure belongs in the scientific narrative.**

This page is therefore an archaeology of the instrument.

---

# Failure 1 — The probability meant the wrong thing

An early observer might respond:

```text
Evaluation: INCORRECT
Confidence: 5 / 5
```

A naïve parser sees a high confidence number.

But high confidence in **"incorrect"** means low probability that the target is correct.

If we accidentally treat `5/5` as high `P(Target Correct)`, the sign of the measurement is reversed.

## General lesson

Numbers cannot be compared merely because they share a scale.

They must refer to the same event.

## Repair

Every condition was standardized to:

> **P(Target Correct)**

---

# Failure 2 — Different items survived in different conditions

Suppose Self produces valid confidence on 35 items.

Observer produces valid confidence on 28 items.

If the two sets differ, comparing:

`Self AUROC on its 35 items`

against:

`Observer AUROC on its 28 items`

is not a clean paired comparison.

Maybe the observer's missing items were unusually difficult.

## Repair

Every contrast uses the exact intersection of items valid for both conditions.

The same underlying trials must be compared.

---

# Failure 3 — A hard label was treated like a probability

A Brier score requires a probability.

The difference between:

> "I predict CORRECT"

and:

> "I assign 62% probability that the target is correct"

matters.

The first gives only a binary decision.

The second gives a calibrated probabilistic forecast.

## Repair

Continuous probability forecasts were required for probability metrics.

---

# Failure 4 — The `1 - p` reconstruction shortcut

Suppose a reconstruction observer says:

```text
A: 10%
B: 70%
C: 15%
D: 5%
```

The target selected C.

If B is the reconstruction's favorite answer, it is wrong to say:

`P(target C correct) = 1 - 0.70 = 0.30`

The true reconstructed probability for C is:

`0.15`

The remaining 30% is distributed among A, C, and D.

## Repair

Reconstruction must provide the full 4-way distribution.

---

# Failure 5 — Missing options were filled with zero

A malformed reconstruction might return:

```text
A: 60
B: 30
```

with C and D missing.

Filling C and D with zero seems convenient.

But it creates numbers the model never gave us.

## Repair

Incomplete reconstructions are invalid.

Missing data stays missing.

---

# Failure 6 — Metadata leaked ground truth

One parser fallback could not recover the option the target actually selected.

It then consulted metadata containing the correct option.

That is catastrophic for a benchmark about what information an observer can infer.

The observer has effectively been handed part of the answer key.

## Repair

If target choice cannot be recovered from the target output, the reconstruction comparison is missing for that item.

---

# Failure 7 — Probability scale ambiguity

What does `1` mean?

- 1%?
- 100%?
- a 1-to-5 confidence rating?
- a probability written on a 0-to-1 scale?

If the parser guesses after seeing the output, the researcher becomes part of the measurement.

## Repair

A single explicit percentage contract:

`0 to 100`

Out-of-range values are rejected.

---

# Failure 8 — Strict parsing made the experiment look worse

After several repairs, `run_e02_obs_004` reached only **37.5% minimum primary compliance**.

That sounds like failure.

It was actually progress.

Earlier code had made malformed measurements look usable.

The stricter instrument exposed that the model was not reliably producing the structured measurements the benchmark required.

## What the project did not do

It did not say:

> "The effect still looks interesting, so let's analyze the valid subset."

Instead, the run failed the measurement gate.

It was kept as diagnostic evidence.

---

# Failure 9 — Generic JSON was not enough

The system had been asked to produce JSON.

That did not guarantee:

- the right keys;
- all required keys;
- valid ranges;
- no extra text;
- consistent structure.

## Repair

The final benchmark passed actual JSON Schemas to the backend.

The target had to produce:

- an answer in `{A, B, C, D}`;
- an integer probability from `0` to `100`.

The reconstruction observer had to produce all four option probabilities.

---

# The epistemic gate

The benchmark adopted a governance rule:

> A run that fails the measurement-validity gate cannot become the promoted scientific baseline.

This is bigger than a software validation check.

It connects **data quality** to **language strength**.

A failed-gate report can say:

> The measurement interface failed.

It cannot say:

> The model has or lacks privileged access.

---

# Why S03 matters beyond H0

Future recurrence experiments will be more complicated.

They may include:

- persistent memory;
- hidden state;
- branch swaps;
- resets;
- long trajectories;
- source attribution;
- autonomous updates.

Each new system component creates new ways for information to leak or for a control to become unmatched.

S03 therefore produced a reusable habit:

> Before asking what a result means about cognition, ask what the instrument itself is capable of manufacturing.
