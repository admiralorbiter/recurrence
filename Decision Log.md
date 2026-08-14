# Decision Log

This file indexes durable program decisions. Major future decisions should receive their own record copied from `templates/decision_record.md`.

| ID | Decision | Status | Rationale | Revisit when |
|---|---|---|---|---|
| **DR-0001** | Do not use consciousness or phenomenality as the direct dependent variable | accepted | no validated cross-substrate measure; operational properties are testable | a defensible new method emerges |
| **DR-0002** | Use an evidence-gated Level 0→1→2→3 ladder | accepted | prevents architecture/training commitment before measurement validity | a later level becomes dramatically cheaper or a lower level is invalidated |
| **DR-0003** | Use Python as the primary research language | accepted | PyTorch, Transformers, interpretability, and statistics ecosystem | a performance-critical subsystem clearly warrants Rust |
| **DR-0004** | Keep Ollama for behavioral scouting, not core mechanistic claims | accepted | speed and convenience versus limited internal access | Ollama exposes validated intervention/state interfaces |
| **DR-0005** | Develop on 0.1–3B systems and validate on larger local models | accepted | 12 GB GPU, repetition and white-box access matter more than size | external compute or a scale-specific effect is established |
| **DR-0006** | Run Level 1 memory/replay controls before native recurrence | accepted | determines whether the task and construct are worth carrying forward | a direct replication requires native recurrence first |
| **DR-0007** | Introduce language report late in the developmental organism | accepted | reduces human-script and self-description shortcuts | language is required for a specific earlier ground-truth task |
| **DR-0008** | Treat endogenous variables as neutral control variables, not emotions | accepted | representation/regulation does not establish valence | theory-linked valence evidence materially changes |
| **DR-0009** | Keep the first developmental world closed, symbolic, and bounded | accepted | inspectability and safety outweigh realism | core causal results replicate and a richer world tests a clear hypothesis |
| **DR-0010** | Preserve nulls, killed ideas, and superseded drafts | accepted | prevents retrospective story rewriting | never; retention method may change |
| **DR-0011** | Redirect S02 toward task validation and 2x2 factorial diagnostics before observer controls | accepted | S01 scout revealed construct contamination (BPE fragmentation vs binding, prompt recency); measurement validity must precede observer modeling | S02 task validation passes with verified exact scoring |
| **DR-0012** | Select counterbalanced 4-way Forced Choice KV Retrieval as primary behavioral substrate | accepted | Operates in optimal psychophysical 55-65% accuracy regime with zero option-position bias | Replaced by multi-token state task at Level 1+ |
| **DR-0013** | Establish Level-0 Privileged Access Null Baseline (PAI ≈ 0) via Observer Ladder | accepted | Stateless autoregressive models possess zero privileged self-access beyond transcript reconstruction (E02 PAI = -0.025, p = 0.816) | Native recurrent models (Level 2+) exhibit positive PAI |


# Pending decisions

| ID | Decision needed | Dependency | Target sprint |
|---|---|---|---|
| **PD-01** | Primary matched base/instruct model family | model inventory and profile | S01–S03 |
| **PD-03** | Level 1 state representation: JSON, free text, or hybrid | E03 pilot | S04 |
| **PD-04** | First native recurrent substrate | state-access prototype | S10 |
| **PD-05** | Recurrent adapter write/read locations | native-state findings | S13–S15 |
| **PD-06** | Organism v0 core architecture | Level 2 gate | S18 |
| **PD-07** | Initial endogenous variables | regulation pilot | S17–S18 |
| **PD-08** | Public release tier for trained artifacts | safety/welfare review | before release |