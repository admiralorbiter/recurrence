# H1: Mnemosyne and the River
## Start Here

**Recurrence — Horizon 1 / Scaffolded Persistence**

You do **not** need a background in machine learning, statistics, cognitive science, or philosophy of mind to follow this site.

H0 asked whether a model could report something about its own performance that outside observers could not recover.

H1 changes the system.

For the first time, the model is allowed to **carry a history forward**.

That sounds simple. It is not.

If a system remembers yesterday, is that already a form of continuity? If it keeps a structured state from one moment to the next, does the timing of those updates matter? Or is the final state merely a convenient rewrite of information that could have been reconstructed later from a transcript?

The first half of H1 asks three questions:

1. **S04 — Can the model read explicit memory?**
2. **S05 — Can a persistent state be maintained without drifting apart?**
3. **S06 — Does maintaining that state online create anything that retrospective replay cannot recover?**

The answer that emerges is more constrained than the project began with:

> **Level-1 persistence is a useful explicit control substrate, but deterministic continuity is reconstructible from history.**

That does not make H1 unimportant.

It makes H1 the strong control that future hidden-state claims must beat.

## The one-sentence story

Explicit memory helps. A deterministic structured state can be maintained reliably. But scheduling those deterministic updates through time does **not** create a unique final state: the same event history can rebuild it later. Structured state mainly buys us a bounded, inspectable control surface and lower long-horizon context cost. A separate bottleneck remains when Qwen2.5-3B itself is asked to reconstruct that compact state from history in one pass.

## The mythic layer

H1 belongs to **Mnemosyne**, memory.

The river is time.

The scientific question is not whether a mythological figure makes the machine conscious. The imagery simply gives us a way to remember the tension:

> **Does memory merely archive what passed through the river, or does carrying state through the river change what the system becomes?**

## What you should be able to explain when you finish

- Why explicit memory is a control for recurrence.
- Why the full transcript can outperform structured state on some tasks.
- Why StructuredSelfState was selected anyway.
- What failed in S05: the scaffold, the model updater, or both.
- Why error inheritance matters.
- What deterministic replay proves.
- Why model reconstruction is different from direct history access.
- What S06 does **not** show.
- Why S07–S09 are still necessary before the H1 gate.
