# What H1 Means Now
## An explicit state machine, not yet a hidden continuing self

The first half of H1 changes how the project should talk about persistence.

Level 1 is best understood as an **externalized state machine**.

It has:

- an event history;
- an explicit state;
- deterministic transition rules;
- a logical clock;
- capacity constraints;
- an audit trail.

Its state follows something like:

`S(t+1) = Update(S(t), Event(t))`

But if `Update` is deterministic and the entire event history is available, then:

`S(T) = Replay(Event(0)...Event(T))`

That is exactly what S06 demonstrated.

## Why this matters for the larger program

H1 is not a failed version of H2.

It is the **strong explicit-memory null model**.

A future hidden recurrent system will have to show something Level 1 cannot explain.

A useful current boundary is:

> **Persistence becomes scientifically interesting for this program when later behavior depends causally on prior state in a way that cannot be reduced to rereading or deterministically rebuilding an externally available record.**

S04–S06 mostly mapped the other side of that boundary.

They tell us what explicit context, explicit state, deterministic maintenance, and replay can already do.

## Memory, persistence, recurrence

### Memory
The past is available.

### Scaffolded persistence
An explicit state is carried through time.

### Genuine latent recurrence
A hidden internal state itself carries causal history.

The project has reached the second category.

It has **not** yet demonstrated the third.

## Claim ceiling

The first half of H1 supports claims about:

- memory architecture;
- state maintenance;
- reconstruction;
- information compression;
- temporal scheduling of explicit state;
- experimental controls.

It does not support claims about:

- phenomenal consciousness;
- irreducible subjective continuity;
- a persistent self;
- privileged internal access.

Those remain far above the current evidence.
