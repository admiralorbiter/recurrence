# Repository Hardening Checklist — Post S13 Core Freeze

**Goal:** Put the repository in a stable handoff state before starting S14.

## P0 — Fix Before New Experimental Work

### 1. Synchronize the H2 static exhibit evidence mirror
`h2/data/core.json` is current, but `h2/site/data.js` must be regenerated from it.

Add a test/script asserting semantic equality between the two.

### 2. Update `h2/README.md`
It should state:
- S10–S13 core frozen;
- S12c frozen, not live;
- S13 frozen, not open;
- S14 active frontier;
- `core.json` includes S13.

### 3. Update `h2/content/core-story.md`
It currently ends by introducing S13 as the next question.

Add the completed S13 act:
- historical-axis coordinate loss;
- recurrent-state reorientation;
- contemporaneous steerability;
- numerical execution sensitivity;
- final frontier becomes S14 ownership / privileged access.

### 4. Update master `6. Roadmap.md`
The dedicated H2 roadmap is current, but the master roadmap still contains the planned pre-execution S13 design.

Replace S10–S13 with short actual-result summaries or mark old details as historical/superseded.

## P1 — Governance / Orientation

### 5. Add “Current Program State” near the top of root `README.md`
Include:
- H0 complete;
- H1 frozen;
- H2 Core S10–S13 frozen;
- current frontier S14;
- links to `walkthrough.md`, `docs/H2_Core_Retrospective_Memo.md`, `docs/H2_Recurrent_Architecture_Roadmap.md`, and the freeze manifest.

### 6. Refresh `Backlog.md`
Current P0/P1/P2 priorities mostly describe completed H0/H1/H2 work.

Recommended structure:
- Completed / frozen foundations;
- P0 current: S14 measurement validation;
- P1 after S14 scout;
- Methods sidecar;
- Deferred;
- Do not chase.

### 7. Refresh `Decision Log.md`
Suggested durable H2 decisions:
- RecurrentGemma-2B is the frozen H2 Core substrate at pinned revision.
- Use the reconstructible/persistent/causal/specific/coordinate-stable/access taxonomy.
- Treat batch shape and precision as methodological variables for long recurrent trajectories.
- S10–S13 core frozen; new work belongs to S14+ unless a freeze-reopening criterion is met.
- Privileged-access claims require a matched nonprivileged observer.

Resolve / retire PD-04 (first native recurrent substrate).

## P1 — Reproducibility

### 8. Add an environment manifest / lock
Freeze:
- Python;
- PyTorch;
- Transformers;
- CUDA runtime;
- NVIDIA driver;
- GPU model;
- attention backend;
- dtype / TF32 settings;
- model revision.

### 9. Pin model revisions in runners
Pass expected revision explicitly and fail closed in confirmatory mode.

### 10. Stop timestamp-only derived-artifact churn
The latest literal commit changes only a regenerated E04 timestamp.

Options:
- omit regeneration timestamps from deterministic derived artifacts;
- use the source-run timestamp;
- exclude such timestamps from versioned content;
- add a pre-commit check.

## P2 — Quality-of-Life

### 11. Add a canonical artifact index
One short manifest should link each frozen sprint report, exact confirmatory run, sensitivity appendix, model/panel hashes, guardrails, and active frontier.

### 12. Add explicit document status labels
Use:
- `FROZEN`
- `ACTIVE`
- `HISTORICAL / SUPERSEDED`
- `DRAFT`

### 13. Add a repo-state test
Assert:
- current H2 frozen core includes S13;
- next major sprint is S14;
- `core.json` and `data.js` agree;
- no canonical H2 README calls S12c live or S13 open;
- frozen model revision appears in the H2 manifest.

## Suggested Immediate Order

1. Fix `h2/site/data.js`.
2. Update `h2/README.md` and `h2/content/core-story.md`.
3. Update master `6. Roadmap.md`.
4. Add root README current-state section.
5. Add Horizon 2 Core Freeze Manifest.
6. Refresh Decision Log.
7. Refresh Backlog.
8. Add environment/version lock and model revision pinning.
9. Start S14.0 measurement-validation planning.
