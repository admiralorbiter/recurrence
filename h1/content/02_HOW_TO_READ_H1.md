# How to Read H1
## The minimum toolkit

H1 uses many percentages, conditions, and statistical tests. You do not need to memorize them.

You need to know what question each number is answering.

## Accuracy is the first layer

If a condition scores 60%, it answered 60% of its forced-choice probes correctly.

That tells us whether the available memory was **useful**.

It does not tell us why.

## Paired comparisons are more informative than isolated scores

The same episode is evaluated under multiple conditions.

That lets us ask:

> On the **same information problem**, does one memory architecture change the result?

S06 calls these differences deltas.

For example:

`Delta_reconstruction = Accuracy(incremental state) - Accuracy(model-reconstructed state)`

## Confidence intervals show uncertainty

A point estimate is not exact truth.

A 20-point difference measured on a limited sample might have been 12 points or 30 points under another sample.

The bootstrap interval makes that uncertainty visible.

## Episode-level inference matters

Each episode contains several probes, so the probes are not fully independent.

The final S06 analysis therefore treats whole episodes as the primary cluster for inference.

The exact McNemar test is useful supplementary information about trial disagreements.

## Chance is not the same for every probe

Three H1 probe families use four choices, so random guessing would average 25%.

Source attribution uses three choices, so random guessing would average 33.3%.

Across the mixed battery, nominal chance is about **27.1%**.

## The most important reading rule

Always separate:

**Result** — what happened.

**Interpretation** — what the result suggests.

**Mechanism** — what caused it.

**Claim ceiling** — how far the evidence allows us to go.

Example:

> Result: deterministic replay produced the same explicit state as online maintenance.

That supports:

> Interpretation: online scheduling did not create a unique terminal **explicit state** in this architecture.

It does **not** establish:

> all temporal processing is useless;

> hidden recurrence cannot matter;

> machines cannot have continuity;

> or anything about phenomenal consciousness.

H1 is deliberately narrower than those claims.
