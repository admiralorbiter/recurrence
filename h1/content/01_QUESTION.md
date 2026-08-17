# The Question
## What would persistence have to add beyond a very good notebook?

A normal independent language-model invocation can be treated as an episode: text goes in, hidden activations mediate generation, text comes out, and those activations are not ordinarily carried into the next independent call.

That means several very different systems can all look “persistent” from the outside:

- a model rereading the full transcript every time;
- a model given a compact deterministic summary;
- a model given a model-written autobiography;
- a model given a typed JSON state;
- a genuinely recurrent architecture carrying hidden state across inference steps.

If all five answer a later question correctly, the behavior alone does not tell us which form of continuity mattered.

## Three ideas that must stay separate

<div class="concept-triad">
<div><strong>Memory</strong><p>Information about the past is available now.</p></div>
<div><strong>Persistence</strong><p>A later state depends on an earlier state.</p></div>
<div><strong>Recurrence</strong><p>Earlier internal state is fed back into later processing.</p></div>
</div>

H1 studies **scaffolded persistence** in explicit prompt space. H2 is reserved for native or latent recurrence.

## The strong boring explanation

The project deliberately builds the strongest ordinary memory explanation before crediting a more mysterious one.

If a transcript already solves a task, recurrence does not get credit.

If a deterministic replay reconstructs the same terminal state, online scheduling does not get credit for an irreducible temporal property.

If a public observer predicts target errors as well as the Self-framed evaluator, first-person wording does not get credit for privileged access.

> **Build the strongest boring explanation first. Then ask what remains.**

## A causal hierarchy

H1 moves from weak evidence to stronger evidence:

1. **Readability:** Can the model answer from the state when it is shown?
2. **Maintenance:** Can the state remain valid over time?
3. **Replay resistance:** Does online processing produce anything later reconstruction cannot reproduce?
4. **Selective quiet dynamics:** Does computation without new evidence improve the right variables without corrupting others?
5. **Causal authority:** Does changing state redirect behavior while history is held fixed?
6. **Ownership:** Does the system know who originated a memory or action?
7. **Observer-adjusted metacognition:** Does Self-framing know more about the target decision than a strong external evaluator?

The farther down this list a result survives, the harder it is to explain with ordinary public prompt cues.

## What would count against the stronger H1 hopes?

A scientifically healthy program must state its disappointments in advance.

H1 would become less “self-like” if:

- raw history matched or beat structured state;
- autonomous state-writing drifted;
- deterministic state could be replayed later;
- quiet reflection generated clutter rather than knowledge;
- state edits lost to conflicting history;
- source ownership collapsed when lexical cues were removed;
- Self and Observer performed similarly when evaluating the same decision.

H1 eventually observed every one of those patterns in some form.

That is why the horizon matters.

<div class="research-callout">
<strong>Research bridge: extended cognition</strong>
<p>Clark and Chalmers' “Extended Mind” asks when an external notebook might participate in cognition rather than merely assist it. H1 does not settle that philosophical question. It gives us experimental distinctions the thought experiment leaves open: availability, reliability, causal leverage, replayability, authority, and privileged access.</p>
<a href="https://onlinelibrary.wiley.com/doi/10.1111/1467-8284.00096" target="_blank" rel="noopener">Open the primary source</a>
</div>
