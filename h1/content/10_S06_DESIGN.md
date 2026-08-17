# S06 · The Replay Challenge
## Does processing the same information incrementally through time matter?

S05 gives H1 a stable deterministic state machine.

That creates a dangerous intuition:

> The state existed continuously, so perhaps its continuity mattered.

S06 attacks that intuition directly.

## The pure temporal comparison

### Scheduled incremental state

Events arrive across logical ticks. The state is updated after each relevant event.

```text
S0 + E1 → S1
S1 + E2 → S2
...
S(T−1) + ET → ST
```

### Deterministic replay state

No state is maintained online. At the terminal query, the same ordered events are passed through the same deterministic transition function.

```text
[E1, E2, ... ET] → Replay → ST(replay)
```

If the two terminal states are identical, then the explicit state contains no irreducible trace of having existed at intermediate moments.

## The five conditions

1. **Scheduled Incremental State** — query the maintained `ST(online)`.
2. **Deterministic Replay State** — query the state reconstructed by deterministic replay.
3. **Raw Transcript** — query directly from the complete event log.
4. **Model-Reconstructed State** — ask Qwen to compress the history into state in one pass, then query that state.
5. **Fresh Floor** — no history and no state.

The conditions separate three very different processes that are easy to blur together:

- deterministic replay;
- direct history access;
- model-based reconstruction.

## The four probe families

### Delayed key–value retrieval

Recover the current value bound to a key.

### Source attribution

Identify which source established a binding.

### Goal state

Recover the status of an explicitly represented goal.

### Multi-hop inference

Follow a relational chain assembled from multiple events.

## The horizon manipulation

Episodes span:

- `T = 10` ticks;
- `T = 25` ticks;
- `T = 50` ticks.

This lets the project observe whether raw-history cost or accuracy changes with longer event sequences.

## In-context foils

All delayed-KV and multi-hop distractors are values that genuinely appeared elsewhere in the same episode.

Why?

Without that control, the correct answer can be the only candidate the model remembers seeing. The task then measures candidate familiarity rather than the intended binding or path traversal.

## Pending goals must actually exist

An earlier version treated “goal absent” as “goal pending.”

E05d explicitly asserts pending goals in both state and transcript.

This sounds minor. It is not.

A scientific instrument cannot score a state the system never represented.

## The strongest invariant

For online state and deterministic replay, S06 verifies:

- canonical state equality;
- serialized state-string equality;
- full probe-prompt hash equality.

The final evaluation prompts are literally the same.

That means any tiny answer difference between the two conditions is backend repeatability noise—not evidence that scheduling created a different explicit state.

## Scope

Final E05d includes:

- 12 exploratory episodes;
- 24 confirmatory episodes;
- 36 total episodes;
- 720 paired trials;
- 480 confirmatory trials.

<div class="expedition-log">
<strong>Expedition log · the pre-registered fork</strong>
<p><strong>If incremental beats deterministic replay:</strong> order-sensitive online state construction may matter.</p>
<p><strong>If replay matches incremental:</strong> deprioritize scaffolded “existence” claims and treat the state as a materialized view of history.</p>
</div>
