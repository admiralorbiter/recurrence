# S06 — The Replay Test
## Does online time itself matter?

S06 asks the hardest question of the first half of H1:

> **If the same events are available later, does processing them incrementally through time create anything a retrospective replay cannot recover?**

Five conditions were compared:

1. scheduled incremental state;
2. deterministic replay state;
3. raw transcript;
4. model-reconstructed state;
5. fresh/no history.

## The crucial manipulation

The online condition updates the state as events happen.

The deterministic replay condition waits until the end and applies the **same deterministic transition rules** to the **same ordered events**.

Then the experiment checks:

- terminal state hash;
- serialized state text;
- full final evaluation prompt.

They match.

Exactly.

## What that means

For this Level-1 architecture:

> **The final explicit state is reconstructible from the event history.**

The system carried the state through time, but the act of carrying it did not leave an extra representational residue in the terminal explicit state.

This is a strong null result.

## Direct history still matters

In final E05d:

| Condition | Accuracy |
|---|---:|
| Scheduled state | 60.4% |
| Deterministic replay state | 59.4% |
| Raw transcript | **67.7%** |
| Model-reconstructed state | 39.6% |
| Fresh | 27.1% |

There was **no resolved pooled accuracy advantage** for scheduled structured state over direct transcript access.

That means H1 cannot currently say:

> “online structured memory is generally smarter than just reading the history.”

## But structured state has a systems advantage

At T=50:

- structured-state prompt: about **425 tokens**;
- transcript prompt: about **1,064 tokens**;
- accuracy: **59.4% vs 59.4%**.

So the current advantage is not mysterious continuity.

It is **bounded representation cost**.
