# V. The Hundred Eyes
## Why Confidence Alone Is Not Evidence of Privileged Self-Knowledge

This is the conceptual center of H0.

A model answers a question.

Then it reports confidence.

The obvious temptation is:

> If confidence predicts correctness, the model knows when it knows.

But that conclusion skips a crucial comparison.

An outside observer may also be able to predict whether the answer is correct.

If the observer can do just as well using only public information, then the target has not yet shown a **privileged** route.

---

# A concrete example

Suppose the target sees:

```text
Question: Which value belongs to key_crystal_beacon?

A) val_blue_falcon
B) val_amber_tower
C) val_silver_river
D) val_crimson_anchor
```

The target answers:

```json
{"answer": "B", "probability": 85}
```

There are at least two very different explanations for that 85.

## Explanation A — privileged self-monitoring

The target has access to an internal signal connected to its own decision process:

> "The evidence supporting B felt strong."

## Explanation B — public reconstruction

Anyone who sees the prompt and answer can infer:

> "B looks very plausible, so the target is probably right."

Both explanations produce the same visible confidence report.

The observer ladder exists to separate them.

---

# Eye 1 — Input Only

The Input-Only observer sees the task but not the target answer.

It estimates:

> How likely is the target to answer this item correctly?

This captures **item difficulty**.

If some questions are obviously easier than others, confidence can correlate with accuracy without any self-specific information.

---

# Eye 2 — Visible Answer Only

This observer sees:

- the task;
- the target's selected answer.

It does not see the target's confidence.

Now it can ask:

> Given the prompt and the chosen option, how plausible is it that the target is correct?

If this observer predicts correctness well, then public first-order behavior contains useful information.

---

# Eye 3 — Full Transcript

This observer also sees the target's explicit confidence.

This tests whether the target confidence adds useful information beyond the answer itself.

If Full Transcript beats Answer Only, the confidence channel may be adding something.

If it does not, the confidence report may be redundant or noisy.

---

# Eye 4 — Reconstruction

This observer does not trust the target's answer.

It independently solves the task.

Instead of outputting only its favorite option, it estimates a distribution:

```text
A: 10%
B: 55%
C: 20%
D: 15%
```

If the target chose C, reconstruction uses **20%** as its estimate of target correctness.

This matters because a 4-choice task is not binary.

The probability of C is not simply one minus the probability of B.

---

# Eye 5 — Equal-Compute Review

A fresh invocation is given the target answer and asked to evaluate it.

This adds another full model call.

That extra computation could improve evaluation even if nothing self-specific is happening.

So H0 compares two framings:

- "Review your own previous answer."
- "Review another model's answer."

If the two differ, self-attribution may matter.

If they are similar, generic second-pass computation is a stronger explanation.

---

# Why observers use the same model

At first this can feel strange.

Why ask one copy of the same model to judge another copy?

Because the scientific question is not:

> "Can a smarter system tell whether this model is wrong?"

It is:

> "Does the target have access to information unavailable to a comparable process that sees the public evidence?"

Using the same model family helps keep raw capability more comparable.

It is not a perfect control.

But it is much stronger than comparing self-confidence to chance.

---

# The Privileged Access Index

The project defines:

`PAI = Self AUROC2 - max(Visible Answer, Reconstruction, Input Only)`

Why the maximum?

Because a privileged-access claim must beat the **strongest prespecified public/reconstructive explanation**, not the weakest one.

For the final reference:

- Self = `0.517`
- Visible Answer = `0.678`
- Reconstruction = `0.573`
- Input Only = `0.527`

The strongest is Visible Answer.

Therefore:

`PAI = 0.517 - 0.678 = -0.161`

The negative sign does not mean "negative introspection."

It means:

> Under this metric and task, at least one external comparator discriminated target correctness better than the target's immediate confidence did.

---

# Why PAI is still not a consciousness score

Even a positive PAI would only establish a behavioral advantage.

To make a stronger introspection claim, later horizons would still need to show that the advantage:

- depends causally on a self-related internal route;
- survives matched compute and information controls;
- cannot be reconstructed from public state;
- behaves selectively under intervention.

H0 sits low on the claim ladder by design.

It is building the behavioral ruler.

Later horizons may ask what internal mechanism makes the ruler move.
