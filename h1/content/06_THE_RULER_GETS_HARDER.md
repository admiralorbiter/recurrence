# The Ruler Gets Harder
## Why S06 needed several versions

S06 is useful for another reason: it shows how attractive findings disappear when the measurement gets stricter.

This is not embarrassment to hide.

It is part of the result.

## E05a — the first version

The early benchmark appeared to show a structured-state accuracy advantage.

But several probes contained shortcuts.

For example, a correct value might be the only answer candidate that had appeared in the episode.

A model could score well by recognizing **familiarity** rather than retrieving the intended binding.

## E05b — remove obvious surface shortcuts

Numeric suffixes and fixed goal-action cues were removed.

The result became less flattering.

That was good.

## E05c — all candidate answers must be in context

KV and multi-hop foils were changed so **every answer option had really appeared in the episode**.

Now the model had to retrieve the correct association rather than select the only familiar candidate.

The large structured-state advantage disappeared.

Again: good.

The ruler got harder.

## E05c also exposed a reconstruction bug

The model was asked to emit a state schema that did not match the Pydantic object used to validate it.

Valid-looking reconstructions could fail validation and fall back to an empty state.

That made retrospective reconstruction look catastrophically bad for the wrong reason.

## E05d — repair the reconstruction interface

A dedicated ReconstructedSelfState interface was added.

Validation succeeded.

The empty-state confound disappeared.

And a smaller but real reconstruction deficit remained.

### The methodological lesson

> **A result that survives an attack is more valuable than a larger effect measured with a weaker ruler.**

The final H1 story uses E05d, not the more dramatic earlier numbers.
