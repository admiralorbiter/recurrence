# S04 · The Notebook
## What can explicit memory already solve?

The first H1 experiment is intentionally unromantic.

It does not build a recurrent neural architecture. It does not claim a hidden self. It gives the model records.

The question is:

> **If the past is explicitly available at the moment of the query, how much of delayed cognition can already be recreated?**

## The three task families

### 1. Delayed key–value retrieval

Earlier in the episode, an opaque binding is established:

```text
key_quartz_summit → val_amber_solstice
```

After intervening events, the model must select the matching value from four options.

This task probes whether a representation preserves a specific associative binding.

### 2. Source attribution

A fact is introduced by one of several sources. Later, the model must identify the source from three options.

This separates content memory from provenance memory.

### 3. Goal resumption

A goal is interrupted, suspended, or queued. Later, the model must identify which goal should be resumed.

This tests whether the representation preserves task state, not just isolated facts.

## The six memory configurations

<div class="condition-cards">
<div><strong>Fresh</strong><p>No history. The uninformed floor.</p></div>
<div><strong>Full transcript</strong><p>The complete chronological record.</p></div>
<div><strong>Deterministic summary</strong><p>A rule-based extractive map of selected facts and sources.</p></div>
<div><strong>Model-written summary</strong><p>An LLM-produced narrative autobiography.</p></div>
<div><strong>StructuredSelfState</strong><p>A typed oracle state containing working memory, goals, sources, and unresolved items.</p></div>
<div><strong>Combined</strong><p>Structured state plus model-written narrative.</p></div>
</div>

These are not six cosmetic encodings of exactly the same information.

They implement different **selection policies**.

The deterministic summary is excellent at key–value extraction because that is what it is designed to keep. It underperforms on goal resumption because suspended goals are excluded by construction.

The full transcript keeps everything but asks the model to find what matters.

The structured state makes selected variables explicit and easy to manipulate but discards narrative chronology.

The model summary may preserve gist while mutating exact symbolic associations.

## Scope and scoring

The hardened S04.1 run used:

- 6 synthetic episodes;
- 42 probes per memory format;
- 252 total forced-choice trials;
- temperature 0.0;
- native JSON schema constraints;
- 100% schema compliance.

Each condition received:

- 18 delayed-KV probes;
- 18 source-attribution probes;
- 6 goal-resumption probes.

That imbalance is why the report includes both micro and macro accuracy.

## Why forced choice?

Earlier H0 work showed that exact string generation and recognition can diverge. Opaque identifiers are difficult to reproduce character by character even when the underlying association may be partially available.

S04 therefore uses forced-choice evaluation to reduce surface-generation burden and focus more directly on information use.

## The first expectation to abandon

A reader might expect the clean, typed `StructuredSelfState` to dominate.

It does not.

The full transcript becomes the strongest static memory condition.

That result will echo through the entire horizon.

<div class="expedition-log">
<strong>Expedition log · before the run</strong>
<p><strong>Possible story:</strong> structured state will organize the past better than an unwieldy transcript.</p>
<p><strong>Danger:</strong> the structured representation may simply omit information that the transcript preserves.</p>
<p><strong>What the experiment must decide:</strong> whether compact organization beats complete chronology on the tested tasks.</p>
</div>
