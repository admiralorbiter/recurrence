# Measurement Archaeology
## The ruler kept changing the result

A research program can fail because the hypothesis is wrong.

It can also fail because the measurement is answering an easier question.

H1 repeatedly encountered the second problem.

<div class="interactive-lab" data-widget="archaeology-timeline">
<div class="kicker">Timeline</div>
<h2>What the first ruler said—and what the repaired ruler found</h2>
<div id="archaeology-timeline"></div>
</div>

## H0 · Recognition was confused with reproduction

Opaque strings were difficult to generate exactly.

Forced-choice recognition revealed information availability that free generation obscured.

**Lesson:** Do not define a capability by one output format.

## S04 · Narrative fidelity was confused with utility

The model summary retained only 2/18 exact bindings but supported 77.8% delayed-KV forced choice.

**Lesson:** Exact object fidelity and behavioral recognition utility need separate metrics.

## S05 · Schema validity was confused with state correctness

Every model update satisfied JSON schema.

The state still lost most facts, invented phantoms, and mishandled goals.

**Lesson:** Syntactic compliance is not semantic state quality.

## S05 · Cheap prompts were confused with efficiency

Full-state rewriting appeared token-efficient because catastrophic forgetting shortened future prompts.

**Lesson:** A broken memory system can look cheap because it no longer carries information.

## S06 · Candidate familiarity was confused with binding retrieval

Early answer sets contained one familiar target and several novel foils.

In-context foils forced the model to retrieve the relation rather than recognize the only previously seen value.

**Lesson:** Every answer candidate must be equally plausible under superficial familiarity.

## S06 · Goal absence was confused with pending state

An absent goal was scored as pending.

The repair explicitly represented the pending goal.

**Lesson:** A probe cannot test a state variable that the environment never instantiated.

## S06 · A schema bug was confused with cognitive reconstruction failure

The model's reconstruction output schema did not match the downstream state validator.

Valid-looking reconstructions fell into an empty fallback state.

After the interface was repaired, a smaller reconstruction deficit survived.

**Lesson:** The most dangerous result is the result that agrees with the hypothesis for the wrong technical reason.

## S07 · Premature reflection was confused with consolidation

The first multi-hop design withheld the second premise until after the null interval.

The repaired E06b placed both premises before the interval.

**Lesson:** A mechanism needs a legitimate opportunity to succeed before a negative result can constrain it.

## S07 · Evidence preservation was confused with epistemic quality

Protected facts remained unchanged.

Bad derived state still interfered with final reasoning.

**Lesson:** A state can be factually intact and epistemically degraded.

## S08 · State-only steering was confused with independent state authority

A model can answer from state when state is the only channel.

The full State × Memory factorial shows that history has far more leverage under direct balanced conflict.

**Lesson:** Causal claims require holding competing information channels fixed.

## S09 · Source-neutral syntax was confused with source-neutral identifiers

The first E08 task used strings such as `key_self_*` and `val_peer_*`.

Removing the lexical leak collapsed the attractive 70% source score.

**Lesson:** A task can look semantically neutral while leaking the label through morphology.

## S09 · Two independent decisions were confused with an observer control

Self and Observer originally predicted the correctness of their own different choices.

The repaired design makes both forecast the same `agent_alpha` decision.

**Lesson:** A privileged-access comparison must predict the same outcome.

## S09 · A naive chance test was confused with a bias-preserving null

The model disproportionately answered Self.

A valid null had to preserve that response distribution while shuffling true source ownership.

**Lesson:** “Chance” must match the actual exchangeability structure of the task.

## Post-S09 · The synthesis itself became a provenance failure

An early H1 master synthesis accidentally replaced canonical S04–S06 results with incorrect stories.

The raw reports remained correct.

The error lived in the epistemic layer above the data.

**Final lesson:**

> **Synthesis needs provenance, versioning, and regression tests too.**

## Statistical invariants added at closeout

The final cleanup adds synthetic tests for:

- AUROC ties and rank mass;
- degenerate classes;
- oracle source prediction;
- all-Self response bias under the corrected null;
- null and planted format interactions;
- canonical offline replay from frozen trials to derived summaries.

That is an important endpoint for a research program about memory:

The code now tests not only the model, but also whether the analysis remembers its own rules correctly.
