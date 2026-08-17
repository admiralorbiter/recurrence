# What H1 Means
## Level 1 is an externalized state machine

The first horizon can now be described precisely.

It consists of:

- an immutable event history;
- a typed explicit state;
- deterministic transition rules;
- a clock and event queue;
- capacity constraints;
- goal-transition constraints;
- model readout from public prompt context;
- optional model-written updates that are themselves externally represented.

Conceptually:

```text
S(t+1) = Update(S(t), E(t))
```

When `Update` is deterministic and the history remains available:

```text
S(T) = Replay(E0 ... ET)
```

S06 verifies that relationship directly for the explicit terminal state.

## What Level 1 is good at

### External continuity

The state remains available across episodes and can be serialized.

### Experimenter control

The state can be reset, cloned, swapped, and surgically edited.

### Bounded representation

The query representation can stay roughly constant while the transcript grows.

### Auditability

Every event, update, and state version can be logged and inspected.

### Strong null modeling

Many apparent continuity effects can be recreated without persistent hidden recurrence.

## What Level 1 does not provide

### Reliable autonomous maintenance

The model updater fails.

### Irreducible temporal state

Deterministic state is replayable.

### Useful quiet explicit consolidation

The tested write mechanism produces clutter rather than correct derivations.

### Epistemic authority

Balanced history usually defeats conflicting structured state.

### Clean source ownership

Source attribution is weak, role-sensitive, and cue-driven.

### Privileged public-information metacognition

A matched observer is not beaten by Self-framing in the positive direction.

## A self-state can be useful without becoming a self

This is the central conceptual result.

`StructuredSelfState` can be:

- useful;
- causally readable;
- compact;
- persistent;
- inspectable;
- behaviorally relevant;

without being:

- private;
- authoritative;
- self-maintained;
- irreducibly historical;
- metacognitively privileged.

The word “self” in the schema is therefore a **functional index** and experimental convenience, not a settled ontological claim.

## The strongest H1 boundary

A useful current statement is:

> **Persistence becomes scientifically stronger when later behavior depends on prior internal state in a way that cannot be reduced to rereading, replaying, or reconstructing an externally available record.**

H1 maps much of the opposite side of that boundary.

That is valuable because it makes H2 harder to fake.

## The surprising role of negative results

Each null or failure sharpens the next mechanism claim:

- replay null → require hidden path dependence;
- quiet-reflection failure → require selective endogenous processing;
- weak state leverage → require causal recurrent-state interventions;
- source confusion → require ownership beyond actor labels;
- no Self advantage → require target-private information unavailable to observers.

The result of H1 is therefore partly a **specification**.

It tells us what “continuity,” “ownership,” and “introspection” would have to mean experimentally.

## Science first, mythology second

Mnemosyne and the river remain useful images.

Memory is the archive.

The river is temporal passage.

H1 finds that a well-kept archive can reproduce a great deal of what looks like continuity from the shore.

H2 must ask whether anything important is carried in the current itself.
