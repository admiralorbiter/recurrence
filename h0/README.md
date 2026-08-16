# Recurrence H0 Narrative — v3
## The Mirror and the Hundred Eyes

This is the updated, drop-in narrative package for **Recurrence — Horizon 0 / Level 0**.

Its design goal remains the same as v2:

> A curious reader with no prior machine-learning, statistics, or experimental-psychology background should be able to enter Horizon 0, understand why each experiment exists, read the major statistics correctly, and know exactly what the evidence does and does not support.

The scientific story has now changed in one important way.

The original H0 narrative ended after the first trustworthy fixed-task reference (`run_e02_obs_005`) and the discovery that larger checkpoints saturated that ruler. H0-v2 followed that problem to its conclusion: it built a harder comparative psychophysical battery, repeatedly hardened that battery against shortcuts and interface artifacts, calibrated model-specific mixed-error operating regimes, and ran a frozen-target observer comparison.

The final result is deliberately asymmetric:

- **Qwen2.5:14B:** confirmatory negative result for a meaningful positive **behavioral privileged-access advantage** under this instrument.
- **Qwen2.5:3B:** unresolved / diagnostic because the external observer measurement gate failed.
- **Llama3.2:3B:** diagnostic calibration failure under the tested direct-value 2AFC interface.

This is **not** a claim that latent privileged internal information is absent, that LLMs have no introspection, or that scale generally worsens metacognition.

## Structure

### Canonical narrative source

`content/`

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

Additional audit material:

- `H0_ARCHAEOLOGY_PACKET.md` — the compact museum of measurement failures, now extended through H0-v2.
- `H0_COMPILED_READING_COPY.md` — all chapters in one file.
- `DROP_IN_NOTES.md` — suggested repository destination and replacement guidance.

### Guided static site

`site/`

Open `site/index.html`.

No framework, build step, server, or external asset is required. The site is generated from the Markdown content in this package and includes:

- 11-page guided navigation;
- plain-English recaps;
- glossary drawer;
- experiment checkpoints;
- H0-v1 and H0-v2 result tables;
- the final Mirror-versus-Hundred-Eyes comparison.

## Recommended repository destination

If this remains a narrative artifact:

```text
docs/
  h0_story/
    content/
    site/
```

If the mythology becomes the public narrative system for all horizons:

```text
story/
  h0/
    content/
    site/
  h1/
  shared/
```

## Design principle

**Science first, mythology second.**

Narcissus, Argus, and Mnemosyne are narrative vocabulary. They never substitute for definitions, controls, run IDs, uncertainty intervals, compliance gates, or claim boundaries.

## Canonical H0 result hierarchy

When historical text conflicts, prefer this order:

1. E02d.1 frozen-target repaired-observer result.
2. H0-v2 final synthesis / frozen N=200 target trials.
3. `run_e02_obs_005` as the canonical fixed-task H0-v1 reference.
4. Earlier hardened H0-v2 mapping/calibration runs.
5. Earlier H0-v1 exploratory runs.
6. Superseded interpretations preserved in archaeology.

The old results remain valuable because H0 is partly the story of the instrument learning how to distrust itself.
