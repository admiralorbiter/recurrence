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
