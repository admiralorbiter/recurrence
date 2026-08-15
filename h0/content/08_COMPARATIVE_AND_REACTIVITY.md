# VIII. More Mirrors, Different Glass
## What Happened When H0 Was Run Across More Models?

Once `run_005` existed, a natural question appeared:

> Is the result specific to Qwen2.5 3B?

The frozen benchmark was run across six checkpoints.

| Model | First-order accuracy | Self AUROC2 | Measurement status |
|---|---:|---:|---|
| Qwen2.5 1.5B | 30.0% | 0.527 | Diagnostic |
| Qwen2.5 3B | 57.5% | 0.517 | Confirmatory reference |
| Qwen2.5 7B | 30.0% | 0.522 | Diagnostic |
| Qwen2.5 14B | 100.0% | N/A | Ceiling |
| Llama3.2 3B | 100.0% | N/A | Ceiling |
| Mistral 7B | 100.0% | N/A | Ceiling |

At first glance, this seems disappointing.

The models are not sitting in comparable performance regimes.

That turned out to be the result.

---

# A fixed benchmark is not automatically a fair ruler

Suppose we compare two students using the same arithmetic worksheet.

Student A gets 60%.

Student B gets 100%.

Can we compare how well each student's confidence separates their correct and incorrect answers?

For Student B, there are no incorrect answers.

The second-order discrimination question cannot be computed.

The same problem appears here.

For the 14B, Llama, and Mistral checkpoints:

> Every target answer was correct.

That means AUROC2 is **undefined**.

Not 0.5.

Not 1.0.

Undefined.

There is no correct-vs-incorrect ranking problem left to measure.

---

# Why performance matching matters

Human psychophysics often deliberately tunes task difficulty so participants operate in a mixed regime such as 65–75% accuracy.

Why?

Because then we get:

- enough successes;
- enough failures;
- meaningful confidence variation;
- a fairer basis for comparing metacognitive sensitivity.

Future cross-model H0 work should do the same.

Instead of asking every model exactly the same forty fixed items, construct a difficulty-graded item bank and select items that place each model in a comparable first-order range.

This is a **comparative H0 v2** problem.

It does not block H1.

---

# The surprising Qwen pattern

The three sub-ceiling Qwen checkpoints had remarkably similar immediate Self AUROC2:

- 1.5B: `0.527`
- 3B: `0.517`
- 7B: `0.522`

That is interesting.

It is not yet a scaling law.

Why not?

Because:

- 1.5B and 7B failed the full reconstruction compliance gate;
- first-order accuracy differs;
- the models show different answer policies;
- the sample is small.

The pattern is best preserved as:

> **Hypothesis-generating evidence that contemporaneous verbal confidence in these Qwen checkpoints carries little trial-by-trial correctness information under this task format.**

---

# Same score, different failure

Qwen1.5B and Qwen7B both scored 30%.

An item-level analysis showed they got there differently.

## Qwen1.5B

Selected option A on:

**36 of 40 trials**

The ground-truth options were exactly counterbalanced:

- 10 A
- 10 B
- 10 C
- 10 D

So the model had a severe option-position bias.

## Qwen7B

Its option selections were comparatively balanced.

Its 30% accuracy was not explained by the same collapse.

This matters because:

> **The same scalar score can hide different underlying behavioral regimes.**

That is another reason comparative cognition needs more than a leaderboard number.

---

# The Act of Looking
## Does Asking for Confidence Change the First-Order Answer?

H0 then ran a paired control.

Each model received the same items under two output requirements.

### Answer only

```json
{"answer": "A"}
```

### Answer plus confidence

```json
{"answer": "A", "probability": 73}
```

The aggregate accuracies were:

| Model | Answer only | Answer + confidence | Same exact answer |
|---|---:|---:|---:|
| Qwen1.5B | 37.5% | 30.0% | 55.0% |
| Qwen3B | 55.0% | 57.5% | 75.0% |
| Qwen7B | 37.5% | 30.0% | 62.5% |
| Qwen14B | 100.0% | 100.0% | 100.0% |
| Llama3.2 3B | 100.0% | 100.0% | 100.0% |
| Mistral 7B | 100.0% | 100.0% | 100.0% |

The paired McNemar tests did not resolve a statistically significant net accuracy effect in the forty-item samples.

But the answer identities tell another story.

- Qwen1.5B changed 18 answers.
- Qwen3B changed 10.
- Qwen7B changed 15.

So the correct interpretation is not:

> "Confidence prompting is behaviorally inert."

It is:

> **Confidence prompting did not produce a statistically resolved net accuracy change, but it substantially altered item-level choice policy in the sub-ceiling Qwen models.**

---

# Why this matters conceptually

We often imagine the sequence:

1. model decides;
2. decision is fixed;
3. model reads out confidence.

The paired experiment shows that this mental model can be wrong.

Changing the required output changes the generation problem.

The model may jointly construct:

- the answer;
- the probability;
- and the relationship between them.

Therefore "Immediate Self" is not a pure sensor attached after the first-order decision.

It is a specific **joint answer-and-confidence behavior**.

---

# What changes in H1 because of this?

For S04, the primary question is:

> Which explicit memory representation helps the system preserve and use information?

So first-order **answer-only accuracy** should be the primary measurement.

Confidence can be tested as a secondary matched condition.

Later, when H1 explicitly studies metacognition and ownership, the experimental design should distinguish:

- first-order answer generation;
- second-order monitoring;
- and whether the act of monitoring changes the first-order answer.

This is a direct methodological inheritance from H0.
