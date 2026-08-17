# The H2 Boundary
## Hidden state is a new channel, not a guaranteed self

H2 changes the causal architecture.

Instead of carrying only public prompt-level state, a recurrent model maintains a hidden state:

```text
h(t) = fθ(h(t−1), x(t))
```

That creates an information channel unavailable to a prompt-only observer.

It does not automatically create:

- a self-model;
- introspection;
- agency;
- consciousness;
- reliable continuity.

Those remain empirical questions.

## H2.1 · Latent continuity

Can two systems with:

- identical visible history;
- identical current input;
- identical explicit memory;

behave differently because their hidden recurrent trajectories differ?

A strong design needs:

- exact state serialization and restore;
- branch cloning;
- visible-history matching;
- reset and swap interventions;
- reconvergence controls;
- backend determinism or documented statistical tolerance.

## H2.2 · Causal recurrent leverage

Does a targeted hidden-state intervention produce a selective, graded behavioral change?

The intervention should avoid off-manifold nonsense.

Controls may include:

- locally matched state swaps;
- interpolation between compatible states;
- norm-matched random perturbations;
- same-dimensional nonrecurrent channels;
- explicit-memory controls;
- unrelated-task damage checks;
- recovery dynamics.

The strong result is not merely “we can break the model.”

It is:

> A plausible state intervention selectively transfers or removes a history-dependent behavior.

## H2.3 · Privileged metacognition

Now the target may have access to recurrent information a public observer lacks.

The target and observer should predict the correctness of the same future decision.

Observer ladder:

- visible input and history;
- output-only;
- reconstructed state summary;
- matched-compute observer;
- another recurrent model with its own state;
- strongest feasible public-information predictor.

A privileged-access result requires the target to outperform the strongest relevant observer because of information in its recurrent state—not because of first-person phrasing.

## Candidate H2 benchmarks inherited from H1

### Replay and path dependence

Can visible state be matched while latent history differs?

### Null intervals

Can recurrent silent evolution selectively improve a derivable task without explicit chain-of-thought writes?

### State × Memory conflict

When visible history is fixed, does hidden state transplantation redirect behavior?

### Source ownership

Can latent trajectory preserve who generated an event after surface actor cues are stripped?

### Output ownership

Can the model distinguish its prior intended output from an artificial prefill based on earlier hidden state inaccessible to a fresh observer?

### Metacognitive monitoring

Can the recurrent target predict its own future failure better than a strong observer with matched visible evidence?

## What would count against H2?

A negative H2 is allowed.

The program should be willing to stop or revise if:

- latent state is accurately reconstructible from public history;
- state swaps cause only generic damage;
- history effects disappear when explicit memory is matched;
- null updates add no selective benefit;
- source ownership remains role-cue driven;
- observers match all target metacognition;
- apparent introspection collapses under input-level anomaly controls.

The goal is not to force the word “self” onto the system.

The goal is to discover which causal properties are actually present.

## What the live H1 closure studies decide

E08c asks whether the canonical source-attribution attractor follows the designated Self role.

E09c asks whether the metacognitive format reversal survives when the exact target decision is fixed.

Once those confirmation runs finish, they should update the H1 claim ledger—not change the H2 architecture from scratch.

<div class="research-callout">
<strong>Research bridge: activation-level introspection</strong>
<p>Recent work injects known concept representations into hidden activations and tests whether models can report or distinguish them. Those studies are relevant because they create a target-private variable unavailable to a text-only observer. H2 must retain H1's hardening discipline: input-level anomalies, reconstruction, relabeling, matched observers, and selective interventions remain mandatory.</p>
<a href="https://arxiv.org/abs/2601.01828" target="_blank" rel="noopener">Open the 2026 activation-intervention paper</a>
</div>
