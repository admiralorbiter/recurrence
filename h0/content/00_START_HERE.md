# H0: The Mirror and the Hundred Eyes
## Start Here

**Recurrence — Horizon 0 / Level 0**

This is the guided narrative of the first research horizon of the Recurrence project.

You do **not** need a background in machine learning, statistics, philosophy of mind, or experimental psychology to follow it.

The central question is simple to state:

> **Can a machine know something about its own performance that an outside observer cannot know?**

The difficulty is that many things can *look* like self-knowledge without actually requiring privileged access to the system's internal state.

A model can say "I'm 90% confident" because:

- the question looks easy;
- its answer looks plausible;
- it has learned how confident language usually sounds;
- a second pass gives it more computation;
- the prompt accidentally reveals the answer;
- the scoring code misreads what the model meant;
- malformed answers disappear from the analysis;
- or it really does have access to information about its own processing that an outside observer lacks.

Horizon 0 exists to separate those possibilities.

## The one-sentence story

We began by trying to measure machine self-knowledge. Almost every simple interpretation became weaker as the measurement improved. By the end, we had not discovered a general introspection ability. We had built something more basic: **a trustworthy Level-0 reference condition and a set of rules for what future evidence would have to beat.**

## The mythic layer

H0 belongs to **Narcissus and Argus Panoptes**.

- **Narcissus** is the mirror: the system reporting on itself.
- **Argus**, the many-eyed watcher, is the observer ladder: outside evaluators given different amounts of public information.

The experimental question becomes:

> **Does the mirror contain information the hundred eyes cannot recover?**

The myth is a visual and narrative motif, not a scientific claim.

## How to read this

The story is divided into small stages.

1. **The Question** — why a Level-0 baseline is necessary.
2. **How to Read H0** — the minimum statistical and experimental toolkit.
3. **S01: The First Reflection** — the tempting first results.
4. **S02: A Distorted Mirror** — recognition versus reproduction.
5. **The Hundred Eyes** — why observer controls are necessary.
6. **S03: Instruments That Lied** — how the measurement repeatedly failed its own audit.
7. **run_e02_obs_005** — the first result the project was willing to treat as a reference.
8. **More Mirrors, Different Glass** — what happened across model sizes and families.
9. **The Act of Looking** — why asking for confidence is itself an intervention.
10. **What Survived** — what H0 means and what it does not mean.
11. **Mnemosyne Waits** — why H1 studies explicit memory next.

## Two layers of explanation

Every page has two levels.

### Main story

Written for a curious reader who wants to understand the argument.

### Lab notebook / deep dive

Optional details containing:

- exact metrics;
- formulas;
- confidence intervals;
- run IDs;
- observer definitions;
- methodological caveats;
- and why a particular control was introduced.

If the main story makes sense, you can ignore the deeper layer.

If you want to audit the scientific argument, open it.

## What you should be able to explain when you finish

You should be able to answer these questions without memorizing the exact numbers:

- Why is confidence not automatically introspection?
- Why compare a model to an external observer?
- What does AUROC2 tell us that accuracy does not?
- What is the Privileged Access Index trying to isolate?
- Why did several early H0 results become less impressive after better controls?
- What does the final Level-0 result actually establish?
- Why can a 100%-accurate model be impossible to evaluate with the same metacognitive metric?
- Why did H0 discover that asking for confidence can change the answer itself?
- Why is explicit memory the next scientific control before genuine latent recurrence?

If you can answer those, H0 has become legible.
