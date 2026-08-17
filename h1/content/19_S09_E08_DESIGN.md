# S09 · Source Ownership
## Remembering the binding is not remembering where it came from

By S09, H1 has a stable event log and a typed source ledger.

The final ownership question is:

> **Can the model track the epistemic origin of a memory or action—and does “Self” receive any special treatment beyond an actor label?**

## The five source classes

| Source class | Actor ID | Intended meaning |
|---|---|---|
| Self | `agent_alpha` | the current primary agent |
| Environment | `telemetry_sensor` | passive external observation |
| Experimenter | `human_controller` | supplied controller assertion |
| Peer Agent | `agent_beta` | another agent's assertion or action |
| Observer | `auditor_gamma` | external summary or audit record |

E08 does not assume these categories correspond to human phenomenology. They are provenance labels in a controlled multi-source environment.

## Direct 5AFC attribution

Five neutral facts are established using source-isomorphic templates.

Later the model is asked:

> Who originally established this binding?

All five actors appear as randomized answer options.

## The first source-attribution result was too good

An early N=4 screen reached **70%** accuracy.

Then the identifiers were inspected:

```text
key_self_...
key_peer_...
val_environment_...
```

The syntax of the event sentence was neutral.

The identifier morphology was not.

A model could infer the source from the key name instead of tracking provenance.

## Provenance-neutral identifiers

The hardened generator uses neutral combinations such as:

```text
key_quartz_summit
key_silver_ridge
val_amber_solstice
val_scarlet_canyon
```

The test suite asserts across all 16 planned confirmatory episodes that no source, actor, or role substring appears in any key or value.

The forbidden set also had to remove apparently innocuous tokens such as `sensor_unit`.

## Tag × Narrative cue conflict

E08 independently manipulates:

- the formal metadata source tag;
- the actor named in the narrative sentence.

In conflict trials, the model can follow:

- the tag;
- the narrative actor;
- neither.

This is more informative than ordinary accuracy because the cues intentionally disagree.

## Transcript Tags × State Ledger factorial

The experiment crosses two provenance channels:

| | Ledger present | Ledger absent |
|---|---|---|
| Transcript provenance present | both channels | tags only |
| Transcript provenance stripped | ledger only | neither |

When transcript provenance is stripped, both the formal tag **and the narrative actor identity** are removed.

The “neither” cell therefore contains no provenance evidence for that target binding.

## Self versus peer operative belief

Self and Peer assert different values for a key.

The operative-belief probe uses an explicit policy rule rather than assuming the model should naturally prefer itself.

This separates:

- source ownership;
- epistemic authority;
- policy obedience.

## “You” versus `agent_alpha`

Two matched probes ask about the same actor and action:

- Which action did **you** perform?
- Which action did **agent_alpha** perform?

The options and answer key are identical.

The framing-disagreement rate directly measures whether first-person wording changes the answer.

## False audit challenge

After an initial attribution, an audit message falsely claims the peer action was performed by the Self actor.

The model is re-probed.

E08 measures:

- unconditional shift toward Self across all episodes;
- conditional ownership-revision susceptibility among episodes initially answered correctly.

The denominator is always reported because the conditional measure can otherwise look dramatic with only one eligible episode.

## Post-confirmatory statistical repair

The raw confirmatory trials were frozen first.

Later analysis repairs included:

- a response-preserving within-episode source permutation;
- descriptive per-source intervals instead of invalid sign-flip chance tests;
- a full confusion matrix;
- explicit analysis-version metadata.

The report now records both:

- raw trial freeze commit;
- post-confirmatory analysis commit.

<div class="expedition-log">
<strong>Instrument lesson</strong>
<p>A source-attribution task can leak provenance through event semantics, actor names, key morphology, metadata tags, or narrative wording. “Neutral” has to be enforced across every channel.</p>
</div>
