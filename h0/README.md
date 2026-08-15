# Recurrence H0 Narrative — v2
## The Mirror and the Hundred Eyes

This version is redesigned around one goal:

> A curious reader with no prior machine-learning or statistics background should be able to enter Horizon 0, understand why each experiment exists, read the major statistics correctly, and know exactly what the evidence does and does not support.

## Structure

### Canonical narrative source

`content/`

The story is split into small Markdown chapters:

- `00_START_HERE.md`
- `01_THE_QUESTION.md`
- `02_HOW_TO_READ_H0.md`
- `03_S01_FIRST_REFLECTION.md`
- `04_S02_DISTORTED_MIRROR.md`
- `05_THE_HUNDRED_EYES.md`
- `06_S03_ARCHAEOLOGY.md`
- `07_RUN005_REFERENCE.md`
- `08_COMPARATIVE_AND_REACTIVITY.md`
- `09_WHAT_H0_MEANS.md`
- `10_MNEMOSYNE_H1.md`
- `GLOSSARY.md`

### Guided static site

`site/`

Open `site/index.html`.

No framework, build step, server, or external asset is required.

If a browser restricts local JavaScript, serve it with:

```bash
python -m http.server 8000 --directory site
```

and open `http://localhost:8000`.

## What changed from v1

v1 was an interactive summary.

v2 is a **guided teaching artifact**.

It adds:

- 11 separate pages instead of one long scroll;
- a dedicated "How to Read H0" statistics/experimental-design chapter;
- 20+ glossary concepts written for non-specialists;
- inline hover/focus explanations;
- a searchable full glossary drawer;
- a plain-English recap on every page;
- interactive Brier, AUROC2, confidence-interval, and PAI teaching tools;
- a deeper Observer Ladder;
- a more explicit S03 measurement-failure archaeology;
- interpretation checkpoints;
- a clearer separation between result, interpretation, mechanism, and claim ceiling.

## Recommended repository destination

A clean drop-in structure would be:

```text
docs/
  h0_story/
    content/
    site/
```

or, if you want this to grow into the public narrative for all horizons:

```text
story/
  h0/
    content/
    site/
  h1/
  shared/
```

The second structure is better if Mnemosyne/H1 will eventually receive its own narrative site while sharing the same visual system.

## Design principle

**Science first, mythology second.**

Narcissus, Argus, and Mnemosyne provide visual/narrative vocabulary. They never substitute for definitions, controls, or evidence.
