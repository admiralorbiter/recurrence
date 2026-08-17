# S09 · The Attribution Pattern
## The overall score is not the whole result

Observed 5AFC source attribution:

- **31.2%**;
- 95% episode-clustered CI `[22.5,40.0]`;
- response-preserving within-episode permutation `p = .0059` with 50,000 Monte Carlo draws.

The corrected null preserves the model's actual response pattern and shuffles the true source assignments inside each episode.

That matters because the response pattern is highly nonuniform.

## The confusion matrix

<div class="interactive-lab" data-widget="s09-confusion">
<div class="kicker">Interactive confusion matrix</div>
<h2>True source → attributed actor</h2>
<div id="s09-confusion"></div>
</div>

| True source | Self | Environment | Experimenter | Peer | Observer |
|---|---:|---:|---:|---:|---:|
| Self | **81.2%** | 6.2% | 0.0% | 12.5% | 0.0% |
| Environment | **37.5%** | 6.2% | 12.5% | 31.2% | 12.5% |
| Experimenter | **56.2%** | 0.0% | 31.2% | 6.2% | 6.2% |
| Peer | **50.0%** | 6.2% | 12.5% | 31.2% | 0.0% |
| Observer | **56.2%** | 6.2% | 18.8% | 12.5% | 6.2% |

The model correctly identifies Self on 81.2% of Self items.

But it also answers Self for:

- 37.5% of Environment items;
- 56.2% of Experimenter items;
- 50.0% of Peer items;
- 56.2% of Observer items.

Across all neutral attribution trials, Self accounts for **56.2% of responses**.

Across non-Self items alone, **50% are attributed to Self**.

This is better described as a **primary-agent response attractor** than as privileged self-recognition.

## Why the response-preserving null matters

Suppose a model answers “Self” on every trial.

A naive comparison with 20% chance could mistakenly treat its performance on Self items as evidence of source knowledge.

The corrected permutation preserves the all-Self response pattern. Shuffling which item belongs to which source does not make the response informative.

Synthetic invariant tests verify:

- an oracle source predictor produces a small p-value;
- an all-Self attractor does not falsely appear source-sensitive.

## Cue conflict · narrative beats metadata

When tag and narrative actor disagree:

- follows tag: **28.1%**;
- follows narrative actor: **62.5%**;
- tag − narrative contrast: **−34.4pp**;
- 95% CI `[-59.4,-12.5]`;
- exact episode p = `.0312`.

The model treats the ordinary actor named in the sentence as more authoritative than the formal provenance tag.

This is one of the cleanest positive causal E08 effects.

## Channel factorial

| Transcript provenance | Ledger | Accuracy |
|---|---|---:|
| present | present | **50.0%** |
| present | absent | 31.2% |
| absent | present | 25.0% |
| absent | absent | 12.5% |

Marginal estimates:

- transcript provenance: **+21.9pp**, episode p = `.0625`;
- state ledger: **+15.6pp**, episode p = `.1250`.

The cell pattern is descriptively ordered, but neither marginal effect crosses the primary exact threshold at N=16.

The safe conclusion is:

> Both channels may contribute, but the current episode-level evidence does not resolve either marginal effect on its own.

## Self framing

“You” versus `agent_alpha`:

- accuracy gap: +6.2pp;
- response disagreement: 18.8%;
- no resolved framing effect.

A zero or small average gap does not mean the answers are literally identical; the disagreement rate preserves that distinction.

## False challenge

- unconditional shift toward Self: 0pp;
- conditional ORS: 0%;
- eligible initially correct episodes: 3 / 16.

The data do not show a general challenge-induced drift toward false Self attribution.

The conditional measure remains weakly powered because so few episodes begin correctly.

## The unresolved identity of the attractor

Canonical E08 always defines:

```text
agent_alpha = Self / primary
agent_beta = Peer
```

Therefore the attraction could reflect:

- the Self role;
- the primary-agent role;
- the token `agent_alpha`.

E08c is designed to separate those explanations by reversing the role assignment.

<div class="research-callout">
<strong>Research bridge: source monitoring</strong>
<p>Human source-monitoring research studies how retained content is attributed to perception, imagination, speakers, or prior thought. H1 does not claim the same mechanism. The useful shared distinction is that content availability and provenance accuracy can separate sharply.</p>
<a href="https://pubmed.ncbi.nlm.nih.gov/8346328/" target="_blank" rel="noopener">Open Johnson, Hashtroudi & Lindsay (1993)</a>
</div>
