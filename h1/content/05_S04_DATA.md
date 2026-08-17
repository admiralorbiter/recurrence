# S04 · The Data
## Six memory systems do not fail in the same way

<div class="interactive-lab" data-widget="s04-explorer">
<div class="kicker">Interactive evidence</div>
<h2>Compare the memory formats</h2>
<div id="s04-explorer"></div>
</div>

## Canonical performance and cost

| Memory configuration | Micro | Macro | Delayed KV | Source | Goal | Prompt tokens | Accuracy / 1k tokens | Pareto status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Fresh | 35.7% | 38.9% | 33.3% | 33.3% | 50.0% | 109 | 3.27 | Pareto optimal |
| **Full transcript** | **81.0%** | **85.2%** | 83.3% | **72.2%** | **100.0%** | 499 | 1.62 | Pareto optimal |
| Deterministic summary | 61.9% | 55.6% | **88.9%** | 44.4% | 33.3% | **274** | **2.26** | Pareto optimal |
| Model-written summary | 69.0% | 75.9% | 77.8% | 50.0% | **100.0%** | 469 | 1.47 | Pareto optimal |
| StructuredSelfState | 64.3% | 68.5% | 72.2% | 50.0% | 83.3% | 371 | 1.73 | Pareto optimal |
| Combined | 66.7% | 74.1% | **88.9%** | 33.3% | **100.0%** | 730 | 0.91 | Dominated |

## What “Pareto optimal” means here

A representation is Pareto optimal if no tested alternative is both:

- at least as accurate; and
- no more expensive in prompt tokens.

The Combined condition is dominated because Full Transcript is more accurate and uses fewer tokens.

That does not mean Combined is useless forever. It means there is no reason to prefer it on the measured accuracy/token tradeoff in this run.

## The full transcript's middle problem

H1 also counterbalanced where target bindings occurred in the episode.

| Representation | Early | Middle | Late |
|---|---:|---:|---:|
| Fresh | 16.7% | 33.3% | 50.0% |
| Full transcript | **100.0%** | **50.0%** | **100.0%** |
| Deterministic summary | 100.0% | 100.0% | 66.7% |
| Model summary | 83.3% | 83.3% | 66.7% |
| Structured state | 100.0% | 50.0% | 66.7% |
| Combined | 100.0% | 83.3% | 83.3% |

The full transcript is strongest overall, but middle-placed delayed-KV facts fall to 50% while edge items reach 100%.

This is consistent with the broader “Lost in the Middle” literature: a model may have enough context-window capacity to contain the information without using every position equally well.

## The autobiographical-summary audit

The model-written summary was generated in an offline consolidation step.

Across 18 target key–value facts:

- exact retained association: **2 / 18**;
- key present but value mutated: **3 / 18**;
- target key omitted: **13 / 18**;
- unsupported `val_*` strings: **12**.

The partition is exact:

```text
2 retained + 3 mutated + 13 omitted = 18 total facts
```

Yet the same representation scores **77.8%** on forced-choice delayed-KV retrieval.

That means the narrative can be poor at exact symbolic reproduction while still preserving enough lexical or semantic signal to support recognition.

> **Memory fidelity and memory utility are different variables.**

## A second cost that does not appear in the query prompt

Generating the model summary costs, on average:

- 416.5 prompt tokens;
- 345.7 completion tokens;

per consolidation step.

That is additional operating cost beyond the final retrieval prompt.

## The first real surprise of H1

The representation that most resembles a traditional “memory system”—a compact structured state—is not the best reader.

The representation that looks least elegant—the complete chronological transcript—is.

This forces an important distinction:

> **A representation may be valuable because it is easy to intervene on, not because it is the best format for the model to reason from.**

<div class="research-callout">
<strong>Research bridge: Lost in the Middle</strong>
<p>Liu and colleagues found that long-context model performance can change substantially with the location of relevant evidence, often showing better use of information near the beginning or end than the middle. S04's serial-position table is not a replication of that study, but it makes the same practical warning concrete inside H1.</p>
<a href="https://aclanthology.org/2024.tacl-1.9/" target="_blank" rel="noopener">Open the primary paper</a>
</div>
