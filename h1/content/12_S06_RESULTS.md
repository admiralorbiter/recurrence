# S06 · The Final Evidence
## The canonical E05d table

<div class="interactive-lab" data-widget="s06-condition-explorer">
<div class="kicker">Interactive evidence</div>
<h2>Compare the five conditions</h2>
<div id="s06-condition-explorer"></div>
</div>

| Condition | Micro | KV | Source | Goal | Multi-hop | Query tokens | Amortized tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Incremental state | 60.4% | 66.7% | 58.3% | 79.2% | 37.5% | 420.9 | 420.9 |
| Deterministic replay | 59.4% | 62.5% | 58.3% | 79.2% | 37.5% | 420.9 | 420.9 |
| **Raw transcript** | **67.7%** | **83.3%** | **66.7%** | 58.3% | **62.5%** | 807.4 | 807.4 |
| Model reconstruction | 39.6% | 45.8% | 41.7% | 41.7% | 29.2% | 378.7 | 558.4 |
| Fresh | 27.1% | 33.3% | 20.8% | 29.2% | 25.0% | 113.8 | 113.8 |

## Result 1 · Scheduling does not create a unique explicit state

Incremental minus deterministic replay:

- point estimate: **+1.0pp**;
- bootstrap interval: `[0.0,+3.1]`;
- episode permutation: `p = 1.0`;
- terminal state hash: identical;
- final prompt hash: identical.

The behavioral one-point difference cannot be attributed to a different explicit state because the prompts are literally the same.

The strongest result is architectural:

> **The deterministic Level-1 terminal state is algorithmically reconstructible from the ordered event history.**

## Result 2 · Model reconstruction is a real bottleneck

Incremental minus model-reconstructed state:

- **+20.8pp**;
- 95% bootstrap CI `[+9.4,+32.3]`;
- McNemar `p = .0055`;
- primary episode permutation `p = .0025`.

When the task requires a compact structured state, maintaining that state deterministically avoids a substantial loss introduced by asking Qwen2.5-3B to reconstruct the entire multi-slot object in one pass.

This does **not** mean retrospective history access fails.

The raw transcript scores 67.7%.

The bottleneck is specifically **retrospective compression into the prescribed structured representation**.

## Result 3 · Raw history remains a formidable reasoning surface

Incremental state is 7.3 points below transcript overall.

The primary episode test does not resolve the pooled difference (`p = .1469`).

The safe statement is:

> No resolved overall accuracy advantage for incremental structured state over direct raw-history access at the tested horizons.

Not:

> The conditions are equivalent.

## Result 4 · Bounded cost appears at longer horizons

<div class="interactive-lab" data-widget="s06-horizon-explorer">
<div class="kicker">Horizon explorer</div>
<h2>Accuracy and prompt growth at T = 10, 25, 50</h2>
<div id="s06-horizon-explorer"></div>
</div>

At `T = 50`:

- incremental state: 59.4%, 424.7 prompt tokens;
- raw transcript: 59.4%, 1,063.6 prompt tokens.

The state uses roughly 60% fewer query tokens in that slice while matching observed accuracy.

This is a systems result, not a selfhood result.

## What S06 changes conceptually

Before S06, explicit persistence can feel inherently temporal.

After S06, the better description is:

> **A continuously maintained materialized view of history.**

That view is useful because it is compact and immediately available. But when deterministic replay can recreate it exactly, the final explicit state does not by itself establish an irreducible continuity through time.

## Why S07 becomes necessary

If online scheduling does not add much, perhaps meaningful computation happens during intervals when no new event arrives.

That possibility cannot be tested with S05's identity no-ops.

The next sprint must give the system actual update cycles during informational silence—and distinguish useful consolidation from arbitrary drift.

<div class="handoff-card">
<span>S06 → S07</span>
<strong>From carrying state to changing state while nothing happens</strong>
<p>The new question is no longer whether the state exists between events. It is whether those intervals produce selective, useful transformation.</p>
</div>
