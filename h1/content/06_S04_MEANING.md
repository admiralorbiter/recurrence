# S04 · What Changed
## The benchmark answer and the architectural decision are not the same thing

The benchmark answer is straightforward:

> **For the tested static read tasks, the full transcript is the strongest overall memory condition.**

The architectural decision is different:

> **Use StructuredSelfState for S05 because it is a better experimental object.**

## Why choose the lower-scoring state?

`StructuredSelfState` exposes named, typed fields:

- working-memory bindings;
- goal registry and statuses;
- source ledger;
- unresolved queue;
- logical time;
- later, derived inferences.

That makes it possible to:

- bound capacity;
- enforce legal goal transitions;
- hash and compare state;
- clone it;
- reset it;
- swap it across branches;
- surgically alter one slot;
- preserve some fields while allowing others to change.

A full transcript is richer, but much harder to intervene on cleanly.

S04 therefore distinguishes two meanings of “better”:

### Better for the model to read

The full transcript.

### Better for the experimenter to manipulate

The structured state.

That distinction becomes the foundation of S08.

## Three claims that survive S04

### 1. Explicit memory explains a great deal

Fresh performance is 35.7%. Every serious explicit-memory condition rises far above it.

Future recurrent systems must therefore be compared with strong memory controls, not only fresh invocation.

### 2. Memory formats are policies, not containers

A representation decides what gets retained, omitted, foregrounded, and made easy to query.

The deterministic summary does not “fail goals” randomly; it excludes suspended goals.

The full transcript does not “fail the middle” because information is absent; the model has difficulty accessing it.

### 3. Exact fidelity is not sufficient or necessary for utility

The narrative summary can lose exact associations while preserving enough information for recognition.

That means later memory research needs both:

- object-level state fidelity metrics; and
- downstream behavioral utility metrics.

## Claims S04 does not support

S04 does **not** show:

- that any state persists autonomously;
- that the model writes its own state accurately;
- that structured state is authoritative;
- that explicit memory is equivalent to recurrence;
- that the model experiences autobiographical memory.

It only shows what the model can read from externally constructed memory representations.

## Why S05 becomes unavoidable

Reading a supplied state is easy compared with maintaining one.

The next question is no longer:

> Can the model retrieve from a state?

It is:

> Can a system keep that state valid as new events arrive, goals change, capacity fills, and time passes?

<div class="handoff-card">
<span>S04 → S05</span>
<strong>From reader to maintainer</strong>
<p>The project now has a state representation worth manipulating. It does not yet have a reliable process for keeping that state alive.</p>
</div>
