# H1 Glossary

## explicit memory

History stored outside the model in an inspectable form such as a transcript, summary, or structured state.

H1 begins with explicit memory because it is a strong, simple control. If ordinary written history already solves a problem, hidden recurrence should not receive credit for solving it.

## StructuredSelfState

The typed Level-1 state object used to store working memory, goals, source information, and unresolved items.

StructuredSelfState is not assumed to be a mind or self. It is an experimentally convenient control surface: bounded, inspectable, versioned, and easy to intervene on.

## persistence

The property that a later state depends on an earlier state rather than every episode starting from scratch.

H1 studies scaffolded persistence: state is explicitly carried across ticks. S06 shows that deterministic Level-1 state can also be reconstructed later from the event history.

## recurrence

A system whose internal state is repeatedly fed forward to influence later processing.

The project reserves the stronger recurrence question for H2, where hidden state itself is carried and causally manipulated. H1 is an explicit-memory control, not yet genuine latent recurrence.

## latent state

A hidden internal representation carried by the model rather than a visible text or JSON record.

H2 will ask whether native hidden state carries causal history that cannot be reduced to rereading or deterministically reconstructing an external record.

## full transcript

The raw chronological history of events given directly to the model at evaluation time.

The transcript is an important control because it preserves the whole observable history. In S06, raw transcript access was not resolved as less accurate than scheduled structured state.

## deterministic replay

Rebuilding the same explicit state later by applying the same deterministic update rules to the same ordered event history.

If online processing and deterministic replay end in the identical state and prompt, then scheduling those explicit transitions through time did not create a unique final representation.

## model reconstruction

Asking the language model itself to compress the full history into the structured state in one retrospective pass.

This differs from deterministic replay. In final E05d, Qwen2.5-3B validated a real reconstructed state but still performed substantially worse than the deterministically maintained state.

## null tick

A processing interval with no new task-relevant external information.

S07 asks whether meaningful state changes can happen during such intervals. A null tick can include computation; it simply may not introduce new semantic evidence about the task.

## quiet tick

In S05, a tick with no incoming event that was implemented as a deterministic identity no-op.

Quiet-tick stability showed that the scaffold could remain stable. It did not show autonomous hidden cognition, because the model was not called during these identity ticks.

## deterministic / oracle updater

A rule-based state updater used as the reliable Level-1 control substrate.

The updater deterministically maps events into StructuredSelfState. It is intentionally stronger and more reliable than the tested Qwen2.5-3B autonomous update procedures.

## delta update

Updating only what changed instead of rewriting the entire state.

In S05, delta updating was less destructive than full-state rewriting, but Qwen2.5-3B still omitted most state and was not reliable enough to become the canonical maintainer.

## error inheritance

Persistence can preserve a false state just as effectively as a true one.

S05 showed why memory is not automatically intelligence. Once a bad update is written into a persistent state, the architecture can protect that error over time.

## forced choice

A task where the model selects from a fixed set of candidate answers.

Forced-choice probes reduce exact-generation burden. H1 later hardened the candidates so all options appeared in context, preventing simple familiarity from solving the task.

## in-context foils

Incorrect answer options that really appeared elsewhere in the same episode.

This prevents a model from answering merely by selecting the only candidate it remembers seeing. It must retrieve the correct binding or follow the actual relation.

## clustered bootstrap

A resampling method that treats whole episodes as the unit of resampling.

Multiple probes from one episode are correlated. Resampling episodes rather than individual trials better reflects that dependency when estimating uncertainty.

## McNemar test

A paired test that focuses on items where two conditions disagree.

H1 uses exact two-sided binomial McNemar tests as supplementary trial-level inference.

## sign-flip permutation test

A cluster-level paired test asking whether episode-level differences could plausibly arise if their signs were exchangeable.

The final S06 reporting treats the episode-level sign-flip test as the primary inferential decision criterion, with bootstrap intervals showing effect-size uncertainty.

## chance baseline

Expected accuracy when choosing randomly among the offered answers.

The H1 battery mixes three 4-choice probes and one 3-choice probe, giving a nominal average chance level of about 27.1%.

## claim ceiling

The strongest conclusion the current evidence is allowed to support.

A useful result can still have a low claim ceiling. H1 can support engineering and behavioral claims about explicit state without supporting claims about phenomenal consciousness or irreducible selfhood.