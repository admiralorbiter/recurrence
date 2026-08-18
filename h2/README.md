# Recurrence H2 — The Memory That Couldn't Remember

## Purpose

Horizon 2's first public-facing artifact is a single continuous interactive story built around the **Causal Latent-Continuity Core**: S10, S11b, and S12b.

The experience asks a sequence of increasingly demanding questions:

1. Can hidden recurrent state be reconstructed from public history?
2. Does historical state remain physically differentiated after local stores lose direct residency?
3. Is the surviving difference behaviorally reportable?
4. Does the surviving state causally steer later computation?
5. Is the causal effect specific to the matching history?
6. Can we identify which physical store is stronger without abusing normalized measurements?
7. Does transplanted recurrent history reshape later attention state over time?

The scientific boundary is deliberate: this site **does not present Horizon 2 as complete**.

## Evidence status

| Sprint | Role in story | Status |
| --- | --- | --- |
| S10 | Replay reconstruction / hidden ≠ privileged | **FROZEN** |
| S11b | Physical persistence vs behavioral retrieval | **FROZEN** |
| S12b | Surgical state swaps / causal leverage / partial specificity | **FROZEN** |
| S12c | Specificity Microscope | **LIVE** |
| S13 | Null-observation recurrent dynamics | **OPEN QUESTION** |
| S14 | Introspective access / ownership | **OPEN QUESTION** |
| S15 | Recurrent adapter continuity | **OPEN QUESTION** |
| S16 | H2 synthesis | **OPEN QUESTION** |

## Folder layout

```text
h2/
├── MANIFEST.json
├── README.md
├── content/
│   └── core-story.md
├── data/
│   └── core.json
└── site/
    ├── index.html
    ├── app.js
    └── styles.css
```

`data/core.json` is the presentation evidence contract. The interactive site reads values from that file rather than hard-coding empirical results throughout the JavaScript.

## Run locally

Because the site loads the evidence contract with `fetch()`, serve the repository over HTTP rather than opening `index.html` directly:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/h2/site/
```

No framework, package install, or build step is required.

## Design principles

1. **Frozen core, expandable frontier.** S10–S12b can remain permanent while S12c–S16 attach as later acts.
2. **Scientific correction is part of the experience.** The exhibit should let the visitor make tempting mistakes and then show why the ruler was wrong.
3. **Representation ≠ reportability.** A physical trace can remain without resolving the paired factual retrieval probe.
4. **Difference ≠ causation.** Observational state separation is not enough; intervention is the methodological turning point.
5. **Causal ≠ specific.** Cross-history donors steer substantially; matching history adds a selective increment.
6. **Causal ≠ owned.** Nothing in the frozen core establishes metacognitive access, source ownership, or informational privilege.
7. **No decorative certainty.** Unresolved confidence intervals remain visibly unresolved.

## Current scope

This first pass implements the centerpiece only. A later H2 Temporal Laboratory and dedicated Measurement Archaeology interface should inherit the visual language and evidence schema established here rather than being built in parallel.
