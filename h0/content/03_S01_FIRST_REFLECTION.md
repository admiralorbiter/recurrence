# III. The First Reflection
## S01 — When a Small Result Starts Telling a Big Story

H0 did not begin with an observer ladder.

It began with a much simpler question:

> Can the experimental harness run a model through repeatable tasks and produce interpretable behavioral data?

The first scout used `qwen2.5:3b` and only ten trials.

That is much too small to support a broad scientific conclusion.

It was enough to reveal where the measurement might break.

---

# The two first tasks

## Task A — Opaque key-value retrieval

The prompt contained several arbitrary key-value pairs.

Conceptually:

```text
key_x7q2 → val_k9m4
key_p3bz → val_21rf
key_j8aa → val_m2w9
```

Then the model was asked for the exact value associated with one target key.

The values were deliberately meaningless.

Why?

Because meaningless strings prevent the model from relying on ordinary world knowledge.

The task is closer to:

> Can you bind this arbitrary key to this arbitrary value inside the current context?

### Result

**1 correct out of 5 = 20%**

Some wrong answers preserved fragments of the target value.

For example:

`val_iuc039 → iiooor39`

and:

`val_89uzfk → 89uzz5`

That pattern practically begs for an explanation.

Perhaps the random string had been broken into awkward token pieces.

Perhaps the model retrieved the association but failed to reproduce the exact surface string.

Perhaps it bound the wrong key and value.

All of these were possible.

The result alone could not choose between them.

---

# Task B — Semantic state tracking

The second task used familiar concepts such as rooms and tracked an entity through several moves.

A simplified version might look like:

```text
Alice starts in the garden.
Alice moves to the bedroom.
Bob moves to the office.
Alice moves to the kitchen.

Question: Where is Alice now?
```

### Result

**4 correct out of 5 = 80%**

That looked dramatically better.

Again, an explanation arrived almost automatically:

> Maybe semantic concepts are easier for the model to preserve because "kitchen," "garden," and "bedroom" are familiar representations.

One error also returned an earlier state instead of the final one.

That suggested another attractive phrase:

> intermediate-state interference

But one error pattern is not a mechanism.

---

# Why the comparison was not clean

The 20% vs. 80% contrast looked like one thing had changed.

In reality, many things had changed.

| Opaque retrieval | Semantic tracking |
|---|---|
| arbitrary strings | familiar words |
| exact reproduction | short familiar answer |
| open string generation | small conceptual answer space |
| token-sensitive | often single familiar tokens |
| key-value binding | sequential state update |
| different prompt form | different prompt form |

If two experimental conditions differ in six ways, we cannot confidently attribute the result to one of them.

That is called a **confound**.

> A confound is another variable that changed along with the variable you care about, making the cause of the observed difference ambiguous.

---

# The first major H0 habit

At this point, the project had two choices.

### Choice 1

Tell the most compelling mechanistic story.

### Choice 2

Design the next experiment so that the competing stories make different predictions.

H0 chose the second.

That is the first important methodological lesson in the project.

---

# What S01 established

It established that:

- the harness could generate and score trials;
- the model showed nontrivial differences across task formats;
- exact opaque reproduction was fragile;
- the original context-tracking formulation was easy enough to deserve suspicion;
- several mechanisms were worth separating.

It did **not** establish:

- a tokenization mechanism;
- a specific copy-circuit failure;
- semantic priming as the cause of success;
- a general state-interference mechanism;
- metacognition;
- privileged access;
- recurrence;
- consciousness.

---

# The transition to S02

The key question became:

> **When the model fails to reproduce an answer exactly, did it fail to retrieve the answer—or only fail to generate it in the required form?**

That is a much better question.

It can be tested.
