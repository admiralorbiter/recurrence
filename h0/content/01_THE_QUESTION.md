# I. At the Water's Edge
## What Are We Actually Trying to Learn?

The Recurrence project begins with a question about **continuity through time**.

Most ordinary language-model interactions are episodic. A model receives some text, performs a bounded computation, produces more text, and stops. A later invocation can be given the earlier conversation, but the model does not necessarily carry forward the same hidden internal state that existed during the previous invocation.

That creates a scientific question:

> If we give an artificial cognitive system a more persistent history, does it change what the system can know about itself?

The long-term program separates several possibilities.

## Level 0 — Episodic baseline

The model receives the current prompt and answers.

There is no special persistent state added by the experiment.

This is the control condition.

## Level 1 — Scaffolded persistence

We explicitly carry information forward:

- full transcripts;
- summaries;
- structured state;
- goals;
- clocks;
- event histories.

This is real persistence at the **system** level, but the persistent information is externally stored and inspectable.

## Level 2 — Genuine latent recurrence

A non-text internal state is carried directly from one moment to the next.

Now an intervention can change that state while keeping visible text the same.

This is where the project can begin asking whether causal history is doing something that explicit memory cannot reproduce.

## Level 3 — Developmental organism

The system is trained from the beginning as an individual with a continuing state, rather than adding continuity after the fact.

---

# Why not start with recurrence immediately?

Imagine we add recurrent state and later observe that a model says:

> "I think I am probably wrong."

That sounds interesting.

But what caused the statement?

Maybe recurrent state gave the model a new internal signal.

Or maybe the task was simply hard and the model learned that hard-looking tasks deserve low confidence.

Maybe the model's own output contained obvious signs of error.

Maybe another model could predict the same failure from public information.

Maybe the confidence prompt itself changed the answer.

If we do not know what the **non-recurrent** system already does, we cannot interpret what changed.

H0 therefore asks a deliberately smaller question first.

> **Can we construct a trustworthy baseline for self-monitoring before adding persistence?**

That is why H0 is not "the boring part before the real experiment."

It is what makes the later experiment capable of meaning anything.

---

# What is "self-monitoring" here?

We are not trying to read subjective experience.

We are asking a narrower behavioral question.

Suppose a system makes many decisions. Some are correct and some are wrong.

After each decision, the system reports how likely it thinks it is to be correct.

If those reports are useful, then confidence should tend to be higher on correct trials and lower on incorrect trials.

That is **metacognitive discrimination**.

But even good metacognitive discrimination is not yet **privileged access**.

Why?

Because an outside observer may be able to make the same prediction.

If an observer sees:

- the question;
- the answer;
- perhaps the response time;
- perhaps a reconstruction of the task;

and can predict correctness just as well, then the target has not yet demonstrated an informational advantage.

That distinction is the heart of H0.

---

# The mirror and the eyes

Narcissus asks:

> "What does the system say about itself?"

Argus asks:

> "What can an outside observer infer from everything publicly visible?"

The strongest H0 claim would require the mirror to outperform the eyes **after the observers have been given every fair public cue**.

That is a harder standard than merely asking whether the model can emit a confidence number.

It is also a more useful standard for later recurrence experiments.
