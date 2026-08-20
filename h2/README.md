# Recurrence H2 — The Memory That Couldn't Remember

## Purpose

Horizon 2's public-facing exhibit is a continuous interactive story built around the **Causal Latent-Continuity Core**: Sprints S10, S11b, S12b, S12c, S13, and S14.

The experience asks a sequence of increasingly demanding questions:

1. Can hidden recurrent state be reconstructed from public history?
2. Does historical state remain physically differentiated after local stores lose direct residency?
3. Is the surviving difference behaviorally reportable?
4. Does the surviving state causally steer later computation?
5. Is the causal effect specific to the matching history?
6. Can we identify which physical store is stronger without abusing normalized measurements?
7. Does transplanted recurrent history reshape later attention state over time?
8. How does surviving value-specific recurrent memory evolve across 2,048 tokens of subsequent task-irrelevant drive?
9. Does the model have introspective access to its private state, and does it know whether that state caused an earlier decision or was installed post hoc?

---

## Evidence status

| Sprint | Role in story | Status |
| --- | --- | --- |
| S10 | Replay reconstruction / hidden ≠ privileged | **FROZEN** |
| S11b | Physical persistence vs behavioral retrieval | **FROZEN** |
| S12b | Surgical state swaps / causal leverage | **FROZEN** |
| S12c | Specificity Microscope / value-specific historical binding | **FROZEN** |
| S13 | Controlled dynamics / coordinate loss / state reorientation | **FROZEN** |
| S14 | Latent metacognition / state-conditioned reporting ≠ historical provenance | **FROZEN** |
| S15 | Recurrent adapter continuity | **STRATEGIC FRONTIER** |
| S16 | Monitor/content dissociation & H3 transition | **STRATEGIC FRONTIER** |

---

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

`data/core.json` is the canonical evidence contract, mirrored into `site/data.js` via `python scripts/sync_h2_data.py` for standalone zero-server execution.

---

## Run locally

Open `site/index.html`.

No framework, package install, build step, or server is required. If local JavaScript execution is restricted by your environment:

```bash
python -m http.server 8000
```

Then open: `http://localhost:8000/h2/site/`

---

## The Seven Design Principles (The Seven Dissociations)

1. **Hidden ≠ Privileged.** An internal state variable not exposed in prompt text is still 100% determined by public tokens under deterministic execution (S10).
2. **Persistent ≠ Reportable.** A physical trace remains resolved at 2W without resolving the paired factual retrieval probe (S11b).
3. **Different ≠ Causal.** Observational state separation is not enough; surgical transplantation establishes causal leverage (S12b).
4. **Causal ≠ Specific.** Structured cross-history donors steer substantially; matching history adds a selective increment (S12c).
5. **Specific ≠ Coordinate-Stable.** Value-specific memory remains causally active while dynamically reorienting away from its original output axis (S13).
6. **State-Sensitive Report ≠ Generic Read Head.** Private state modulates self-report in strongly counterfactual settings, but report shifts are not globally coupled across arbitrary perturbations (S14).
7. **State-Conditioned Reporting ≠ Historical Provenance.** Installing the identical RG-LRU state after a decision reproduces practically the same intention report as having that state participate during the decision (S14).
