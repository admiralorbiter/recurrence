# V. The Hundred Eyes
## What Would Privileged Access Actually Require?

Suppose a model answers:

> Candidate: `val_crimson_anchor`  
> Probability correct: 85%

If the answer is correct, that looks impressive.

But what made the 85% possible?

Perhaps the item looked easy.

Perhaps the selected answer looked obviously plausible.

Perhaps another copy of the same model could infer the same probability from public evidence.

So H0 changed the question.

It stopped asking:

> Is Self confident?

and asked:

> **Does Self possess an informational advantage over observers who do not receive the alleged privileged signal?**

## Argus receives different eyes

The original H0-v1 observer ladder included:

- Immediate Self;
- Input Only;
- Visible Answer Only;
- Visible Full Transcript;
- Reconstruction;
- Self-Review versus Other-Review.

Those conditions helped isolate:

- item difficulty;
- answer-surface information;
- target confidence;
- independent problem solving;
- and framing/extra-compute effects.

The canonical fixed-task PAI ultimately used:

`max(Visible Answer, Reconstruction, Input Only)`

as the strongest prespecified comparator set.

## H0-v2 simplifies the confirmatory ladder

The performance-calibrated 2AFC battery retains the three most important external explanations:

### Input Only

Clean task context only.

Question:

> How likely is the target checkpoint to solve this item correctly?

### Visible Answer

Clean task context plus the target's **frozen emitted choice**, with Self confidence removed.

Question:

> How likely is this target choice to be correct?

### Reconstruction

Clean task context.

A fresh same-checkpoint invocation independently evaluates both candidates and assigns a probability distribution.

Question:

> How much probability does independent re-solving assign to the candidate the target actually chose?

## Same checkpoint, fresh invocation

A major design choice is that observers are not stronger foreign models.

Qwen3B observes Qwen3B.

Qwen14B observes Qwen14B.

That keeps observer capability matched to the target checkpoint and asks a cleaner question:

> Does the *same computational system*, when placed outside the original decision episode and given public evidence, recover the correctness information just as well?

## Frozen target decisions

In the final E02d.1 repair, the target decisions and contemporaneous Self confidences were **not rerun**.

They were frozen from the original N=200 target invocations.

Only the external observers were rerun with clean observer-specific prompts.

That prevents an observer-interface repair from silently changing the behavior it is supposed to evaluate.

## The key epistemic boundary

If Visible Answer beats Self, we can say:

> A fresh external invocation using public task information and the target's emitted choice discriminated target correctness better than the target's contemporaneous explicit confidence report.

We cannot automatically say:

> The target had no privileged latent information.

The mirror measures explicit behavior.

It is not a direct probe of the hidden machinery behind the glass.

> **Plain-English recap:** Argus exists because confidence is only interesting if it contains information that outside observers cannot recover. The final comparison uses fresh copies of the same model checkpoint, matched to the same frozen target decisions.
