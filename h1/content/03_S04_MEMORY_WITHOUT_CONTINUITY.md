# S04 — Memory Without Continuity
## What can explicit memory already solve?

The first H1 experiment does something intentionally unromantic.

It gives the model records.

No hidden state is required.

The question is:

> **If we simply write the past down, how much does that already help?**

## Six memory conditions

S04 compared:

- fresh invocation;
- full transcript;
- deterministic summary;
- model-written summary;
- structured self-state;
- structured state plus narrative.

### Canonical S04 result

| Memory format | Accuracy | Prompt tokens |
|---|---:|---:|
| Fresh | 35.7% | 109 |
| Full transcript | **81.0%** | 499 |
| Deterministic summary | 61.9% | 274 |
| Model summary | 69.0% | 469 |
| Structured state | 64.3% | 371 |
| Combined | 66.7% | 730 |

The first lesson is simple:

> **Explicit history helps a lot.**

## The surprising part

Structured state did **not** win.

The full transcript had the highest static accuracy.

So why did the project choose StructuredSelfState for S05?

Because S05 was not asking which format is easiest to read.

It needed something that could be:

- bounded;
- inspected;
- versioned;
- updated;
- swapped;
- reset;
- compared field by field.

Structured state was selected as an **experimental control surface**, not as a champion memory format.

## Memory fidelity is not the same as memory usefulness

Model-written summaries preserved only a small fraction of exact key-value strings, yet still performed reasonably well on forced-choice retrieval.

That means a memory can lose exact wording while retaining enough structure for recognition.

This becomes an important rule:

> **Do not confuse “can reproduce the memory exactly” with “can still use information from the memory.”**
