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
