# Research Bridges · AI Systems
## H1 sits between memory engineering and mechanistic introspection

Several neighboring AI research programs ask related but different questions.

## Lost in the Middle

Liu and colleagues show that long-context models can use information differently depending on where it appears, often performing better at the edges than the middle.

This helps interpret S04:

- the transcript contains all facts;
- middle-position delayed-KV accuracy still falls sharply;
- “inside the context window” is not the same as “uniformly usable.”

It also explains why compact state remains interesting even after raw transcript wins many accuracy comparisons.

A state can reduce search and prompt cost even if it is not the strongest static reasoning representation.

<div class="research-callout"><strong>Lost in the Middle</strong><p>Liu et al., TACL 2024. DOI: 10.1162/tacl_a_00638.</p><a href="https://aclanthology.org/2024.tacl-1.9/" target="_blank" rel="noopener">Primary source</a></div>

## MemGPT and memory as an operating-system problem

MemGPT uses an operating-system / virtual-memory analogy to manage information across memory tiers beyond a model's immediate context.

That is close to H1's engineering layer:

- limited active context;
- external stores;
- controlled retrieval;
- memory-management policies;
- persistent multi-session interaction.

The scientific goals differ.

MemGPT primarily asks how to build a more capable long-running system.

Recurrence asks:

> What does that external memory architecture explain before hidden recurrence is allowed to claim a special role?

<div class="research-callout"><strong>MemGPT</strong><p>Packer et al., 2023. arXiv:2310.08560.</p><a href="https://arxiv.org/abs/2310.08560" target="_blank" rel="noopener">Primary source</a></div>

## Prompt injection as role confusion

Ye, Cui, and Hadfield-Menell frame prompt injection as role confusion: models infer authority from how text is written, not only where it comes from.

That is unusually relevant to E08.

Canonical E08 finds:

- narrative actor leverage: 62.5%;
- formal tag leverage: 28.1%.

E08c then asks whether the attribution attractor follows the actor designated as Self rather than the lexical token `agent_alpha`.

The two projects are not the same:

- E08 measures source attribution and ownership in a synthetic memory setting;
- role-confusion research targets security and authority assignment.

But both warn that formal interface provenance may lose to role cues encoded in natural language.

<div class="research-callout"><strong>Prompt Injection as Role Confusion</strong><p>Ye, Cui & Hadfield-Menell, 2026. arXiv:2603.12277.</p><a href="https://arxiv.org/abs/2603.12277" target="_blank" rel="noopener">Primary source</a></div>

## Activation-level introspection

Recent activation-intervention work injects known concept representations into hidden activations and asks whether the model can report or distinguish those internal states.

That lies beyond H1's claim boundary.

H1 state is public prompt text.

An activation intervention creates a private channel relative to an observer who sees only the input and output.

This is why the work is relevant to H2:

- hidden state can be reset, injected, or swapped;
- the observer can be denied direct access;
- the target's self-report can be compared with strong input-level controls;
- output ownership can be tested against artificial prefills.

The recent papers also emphasize unreliability and model dependence—important warnings against assuming H2 will succeed.

<div class="research-callout"><strong>Emergent Introspective Awareness in Large Language Models</strong><p>Lindsey, 2026. arXiv:2601.01828.</p><a href="https://arxiv.org/abs/2601.01828" target="_blank" rel="noopener">Primary source</a></div>

## Memory provenance laundering

Recent persistent-agent security work identifies a failure mode where memory consolidation retains an action trigger but erases the low-trust source that should limit its authority.

That is a powerful applied cousin of H1's source-ledger problem.

S07 shows model-written consolidation can mutate symbolic state.

S09 shows formal provenance channels can be weak relative to narrative cues.

A robust long-term agent may therefore need platform-enforced provenance that the model cannot silently rewrite.

<div class="research-callout"><strong>Memory Provenance Laundering in LLM Agents</strong><p>Xu et al., 2026. arXiv:2607.29167.</p><a href="https://arxiv.org/abs/2607.29167" target="_blank" rel="noopener">Primary source</a></div>

## What H1 contributes to this neighborhood

H1's contribution is not a new memory product.

It is a control program:

- explicit memory before latent memory;
- replay before continuity;
- observer before privileged access;
- cue conflict before ownership;
- intervention before mechanism claims;
- provenance before synthesis.

Those controls can be carried into H2 and into future agent-memory systems.
