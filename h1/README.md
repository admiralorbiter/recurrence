# Recurrence H1 Narrative — Mid-Horizon v1
## Mnemosyne and the River

This package mirrors the teaching-oriented architecture of `h0/`.

Its goal is:

> A curious reader with no prior machine-learning or statistics background should be able to enter Horizon 1, understand why explicit memory is a necessary control for recurrence, follow S04–S06, read the core results correctly, understand the measurement hardening, and know exactly what remains unresolved before the H1 gate.

## Structure

### `content/`

Canonical narrative Markdown:

- `00_START_HERE.md`
- `01_THE_QUESTION.md`
- `02_HOW_TO_READ_H1.md`
- `03_S04_MEMORY_WITHOUT_CONTINUITY.md`
- `04_S05_STATE_THAT_PERSISTS.md`
- `05_S06_THE_REPLAY_TEST.md`
- `06_THE_RULER_GETS_HARDER.md`
- `07_WHAT_SURVIVED.md`
- `08_WHAT_H1_MEANS_NOW.md`
- `09_THE_UNFINISHED_HALF.md`
- `GLOSSARY.md`
- `MIDPOINT_SYNTHESIS.md` — deeper conceptual integration
- `SOURCE_MAP.md` — page-to-evidence provenance map

### `site/`

Open `site/index.html`.

No framework, build step, server, or external asset is required.

If local JavaScript is restricted:

```bash
python -m http.server 8000 --directory site
```

## Design principle

**Science first, mythology second.**

Mnemosyne and the river are memory/time motifs. They never substitute for definitions, controls, or evidence.

## Roadmap status

This is a **mid-H1** artifact.

H1 is S04–S09. This version tells the story through final S06/E05d and deliberately ends with S07–S09 unresolved.

## Canonical scientific sources

- `6. Roadmap.md`
- `docs/E03_Explicit_Memory_Report.md`
- `docs/E04_Update_Loop_Report.md`
- `docs/E05_Scheduled_vs_Replay_Report.md`
- Protocol anchor for final E05d: `db7273c`
- Canonical E05d result anchor: `e75a963`

## Suggested future update

After S09, preserve this midpoint version and create an H1 final narrative revision rather than silently rewriting the historical midpoint.
