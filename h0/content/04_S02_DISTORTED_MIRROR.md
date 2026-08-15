# IV. A Distorted Mirror
## S02 — Recognition, Reproduction, and the Danger of Easy Tasks

S02 was the first point where H0 began to behave like a real measurement program rather than a collection of interesting prompts.

The problem from S01 was clear.

Opaque key-value retrieval might fail because:

1. the association was never retrieved;
2. the correct value was partially available but difficult to reproduce exactly;
3. arbitrary strings were unusually hard to tokenize or copy;
4. the output format itself was causing errors.

The new design separated some of those possibilities.

---

# The 2 × 2 design

Two dimensions were varied.

## Dimension 1 — Identifier type

- **Semantic** — meaningful words and familiar combinations.
- **Opaque** — arbitrary strings.

## Dimension 2 — Response mode

- **Forced choice** — choose the correct answer from four options.
- **Free generation** — reproduce the exact value without options.

This creates four conditions.

| | Forced Choice | Free Generation |
|---|---:|---:|
| Semantic | 65% | 40% |
| Opaque | 60% | 25% |

The central comparison was not simply semantic vs. opaque.

It was:

> Does the same underlying item become easier when the model only has to **recognize** the correct answer rather than **produce** it exactly?

---

# Why forced choice matters

Imagine you know a person's face but cannot remember their name.

If someone asks:

> "What is their name?"

you may fail.

If someone asks:

> "Is it Alex, Brian, Chris, or Daniel?"

you may immediately recognize Alex.

That does not prove the name was perfectly represented in memory before seeing the options.

But it shows that **recognition and free recall are different demands**.

The same logic applies here.

---

# The result

Across the paired items:

- Forced choice: **62.5%**
- Free generation: **32.5%**

The difference was **30 percentage points**.

More importantly, because the items were paired, we could look at cases where one response mode succeeded and the other failed.

- Forced-choice only correct: **19**
- Free-generation only correct: **7**
- Both correct: **6**
- Both wrong: **8**

An exact McNemar test gave:

`p = .029`

---

# What is a McNemar test?

This is a paired comparison.

It ignores cases where both conditions gave the same outcome and focuses on the disagreements.

In simplified form:

> If response mode truly had no effect, we would expect "forced-choice only" and "free-generation only" successes to be roughly balanced.

They were not:

`19 vs. 7`

That is why the test became useful.

The result supports the claim that **response mode materially changed task success**.

It does not identify the hidden mechanism.

---

# What died here

One of the earliest stories was:

> "Opaque-string errors show the model failed to retrieve the value."

S02 made that too strong.

The better statement became:

> **Opaque exact-reproduction failures combine associative retrieval with surface-generation demands.**

That sounds less dramatic.

It is also more defensible.

---

# The second S02 lesson: context tracking collapses

The original semantic tracking result was 4/5.

That sounded good.

The task was redesigned to make the model track several objects through interleaved transitions rather than letting the relevant final state appear in an easy recent position.

The harder result:

**3 / 20 = 15%**

That is not a small decline.

It changes what the earlier 80% meant.

The original task had not established robust state tracking.

It had established success on an easy task that contained a strong shortcut.

---

# What is a shortcut?

A shortcut is a feature that predicts the answer without requiring the cognitive process the researcher intended to measure.

In machine-learning evaluations, shortcuts are dangerous because a model can score well while solving the wrong problem.

For example:

> Intended construct: "maintain object state across several updates"

Possible shortcut:

> "repeat the location mentioned near the end of the prompt"

If the shortcut works, high accuracy does not establish the intended ability.

---

# Why the 4-way task survived

The new forced-choice KV task was not chosen because it looked more intelligent.

It survived because it had good measurement properties.

It had:

- exact ground truth;
- randomized/counterbalanced answer positions;
- nontrivial but imperfect accuracy;
- a compact answer format;
- a confidence report;
- the possibility of constructing observers;
- a way to compare the same item across conditions.

This is an important research principle:

> **A scientifically useful task is not always the task that looks most impressive.**

The best task is often the one whose failure modes are easiest to understand.

---

# What S02 established

S02 established:

- response mode strongly affected performance;
- exact generation is not equivalent to recognition;
- the original context task overestimated robust tracking;
- the 4-way KV task was useful enough to become the Level-0 measurement substrate.

It still did not establish self-knowledge.

For that, H0 needed the hundred eyes.
