# H0: The Mirror and the Hundred Eyes
## Start Here

**Recurrence — Horizon 0 / Level 0**

You do **not** need a background in machine learning, statistics, philosophy of mind, or experimental psychology to follow this story.

The central question is simple to state:

> **Can a machine know something about its own performance that an outside observer cannot know?**

The hard part is deciding what evidence would count.

A model can say, “I am 90% confident,” for many reasons:

- the question looks easy;
- its chosen answer looks plausible;
- it has learned the style of confident language;
- another invocation can infer item difficulty;
- a second pass gives it extra computation;
- the prompt or output format changes the answer;
- the task contains a shortcut;
- the scoring code silently repairs malformed responses;
- missing trials select an easier subset;
- or the system truly has access to information about its own processing that an outside observer lacks.

Horizon 0 exists to make those possibilities compete.

## The mythic layer

H0 belongs to **Narcissus and Argus Panoptes**.

- **Narcissus** is the mirror: the system reporting on itself.
- **Argus**, the many-eyed watcher, is the observer ladder: outside evaluators with different public-information vantage points.

The experimental question becomes:

> **Does the mirror contain information the hundred eyes cannot recover?**

The myth is a motif, not a scientific claim.

## The one-sentence story

H0 began by trying to measure machine self-knowledge.

Almost every easy interpretation weakened when the ruler improved.

The first half of H0 produced a trustworthy fixed-task reference and an observer architecture. The second half discovered that the same ruler could not fairly compare stronger models, built a performance-calibrated comparative battery, and then hardened that battery through several more measurement failures.

The final result is narrower than “machines do” or “machines do not” introspect:

> **For Qwen2.5:14B on the validated H0-v2 relational task, contemporaneous explicit confidence showed no meaningful positive behavioral privileged-access advantage over matched external observers; the joint PAI interval was entirely negative. Qwen2.5:3B remained unresolved because its external-observer measurement gate failed.**

That is a behavioral result about this instrument, not a metaphysical conclusion and not a claim that latent privileged information is absent.

## The guided path

1. **The Question** — why Level 0 exists.
2. **How to Read H0** — the minimum toolkit.
3. **S01: The First Reflection** — the seductive first result.
4. **S02: A Distorted Mirror** — recognition is not reproduction.
5. **The Hundred Eyes** — why confidence needs observers.
6. **S03: Instruments That Lied** — the measurement archaeology.
7. **The Reference Result** — `run_e02_obs_005`.
8. **Stress-Testing the Ruler** — saturation, H0-v2, calibration, and the confirmatory battery.
9. **What H0 Means** — the surviving claims and boundaries.
10. **Mnemosyne Waits** — why explicit memory comes next.

## What you should understand by the end

You should be able to explain:

- why confidence is not automatically introspection;
- why an outside observer is scientifically necessary;
- why 100% task accuracy can make metacognitive discrimination unidentifiable;
- why equivalent *performance regimes* matter more than identical item sets for cross-model comparison;
- what AUROC2, Brier, `d′`, criterion `c`, meta-d′, and PAI are trying to measure;
- why compliance is part of the measurement instrument;
- why asking for confidence can change a model's first-order choice;
- why the H0-v2 task had to survive shortcut, bias, calibration, and interface audits;
- what the final 14B result actually excludes;
- why the 3B result remains unresolved;
- and why none of this is yet a claim about latent recurrent state or consciousness.

> **Plain-English recap:** H0 is the control condition. It asks whether a stateless model's self-report contains useful correctness information that matched outside observers cannot recover. The real scientific achievement was not a dramatic introspection result; it was learning how difficult that question is to measure without fooling ourselves.
