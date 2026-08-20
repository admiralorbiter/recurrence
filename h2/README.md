# Recurrence H2 — The Memory That Couldn't Remember

## Purpose

Horizon 2's public-facing exhibit is a continuous interactive story built around the **Causal Latent-Continuity Core**: Sprints S10, S11b, S12b, S12c, and S13.

The experience asks a sequence of increasingly demanding questions:

1. Can hidden recurrent state be reconstructed from public history?
2. Does historical state remain physically differentiated after local stores lose direct residency?
3. Is the surviving difference behaviorally reportable?
4. Does the surviving state causally steer later computation?
5. Is the causal effect specific to the matching history?
6. Can we identify which physical store is stronger without abusing normalized measurements?
7. Does transplanted recurrent history reshape later attention state over time?
8. How does surviving value-specific recurrent memory evolve across 2,048 tokens of subsequent task-irrelevant drive?

The scientific boundary is deliberate: this site **does not present Horizon 2 as complete**. Horizon 2 Core (S10–S13) is frozen; the introspective access and ownership frontier (S14+) remains active.

## Evidence status

| Sprint | Role in story | Status |
| --- | --- | --- |
| S10 | Replay reconstruction / hidden ≠ privileged | **FROZEN** |
| S11b | Physical persistence vs behavioral retrieval | **FROZEN** |
| S12b | Surgical state swaps / causal leverage | **FROZEN** |
| S12c | Specificity Microscope / value-specific historical binding | **FROZEN** |
| S13 | Controlled dynamics / coordinate loss / state reorientation | **FROZEN** |
| S14 | Introspective access / reality monitoring / state ownership | **ACTIVE FRONTIER** |
| S15 | Recurrent adapter continuity | **OPEN QUESTION** |
| S16 | H2 synthesis & Level 2 decision | **OPEN QUESTION** |

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
    ├── data.js
    ├── app.js
    └── styles.css
```

`data/core.json` is the canonical evidence contract, mirrored in `site/data.js` for standalone zero-server execution. Use `python scripts/sync_h2_data.py` to synchronize them.

## Run locally

Open `site/index.html`.

No framework, package install, build step, or server is required. If local JavaScript execution is restricted by your environment:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/h2/site/
```

## Design principles

1. **Frozen core, expandable frontier.** S10–S13 remain permanent while S14–S16 attach as later acts.
2. **Scientific correction is part of the experience.** The exhibit lets the visitor make tempting mistakes and then shows why the ruler was wrong.
3. **Representation ≠ reportability.** A physical trace can remain without resolving the paired factual retrieval probe.
4. **Difference ≠ causation.** Observational state separation is not enough; intervention is the methodological turning point.
5. **Causal ≠ specific.** Cross-history donors steer substantially; matching history adds a selective increment.
6. **Specific ≠ coordinate-stable.** Value-specific historical structure can remain causally active while rapidly losing alignment with its original baseline output axis.
7. **Causal ≠ owned.** Nothing in the frozen core establishes metacognitive access, source ownership, or informational privilege.
8. **No decorative certainty.** Unresolved confidence intervals remain visibly unresolved.
