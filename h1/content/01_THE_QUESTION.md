# The Question
## What does it mean for a machine to persist?

Imagine two systems.

**System A** starts fresh every time. It sees the current question and nothing else.

**System B** receives a notebook containing what happened before.

System B may behave more intelligently simply because it has more information.

Now imagine **System C** keeps a structured state and updates it after every event.

It feels more continuous. But is it scientifically different from a notebook?

That is the problem H1 is designed to separate.

## Three ideas that sound similar but are not

### Memory

Information about the past is available now.

A transcript is memory. A summary is memory. A JSON state object is memory.

### Persistence

A state is carried or maintained across time.

In H1, persistence is scaffolded explicitly.

### Recurrence

An internal state repeatedly feeds into future processing.

The project reserves the stronger hidden-state question for H2.

## Why the distinction matters

Suppose a recurrent model beats a fresh model.

Without H1, we could not tell whether the important ingredient was hidden recurrence or simply **access to history**.

H1 therefore tries the simpler explanations first.

> **Before claiming that a hidden state matters, build the strongest explicit-memory system you can understand and control.**

## The burden of proof gets harder on purpose

Every successful H1 control raises the bar for H2.

If transcript access solves a task, H2 does not get credit for solving it.

If deterministic replay recreates the same final state, H2 must demonstrate something that cannot be reduced to that replay.

This is not a detour from the recurrence question.

It is how the project makes the recurrence question testable.
