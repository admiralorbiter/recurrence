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
| **DR-0013** | Retain Level-0 Privileged Access Reference Baseline | accepted | run_e02_obs_005 established a measurement-valid Level-0 reference on Qwen2.5:3b with 100% primary compliance. No privileged self-monitoring advantage was resolved over the prespecified observer/reconstruction controls; joint PAI = −0.161, 95% CI [−0.428, +0.055]. | Replicated across another task/model, or Level-1 persistence produces a materially different observer-adjusted pattern. |
| **DR-0014** | Pin `google/recurrentgemma-2b` (revision `3620f4ca9c5d...`) as Frozen H2 Core Substrate | accepted | Griffin hybrid recurrent-attention architecture provides explicit, inspectable multi-store access (RG-LRU, Conv1D, sliding KV) within 12 GB local VRAM. | A new architecture provides cleaner multi-store isolation or higher scale replication is required. |
| **DR-0015** | Adopt the Six-Way Horizon 2 Theoretical Property Taxonomy | accepted | Replaces the coarse binary "memory vs prompt" with six distinct empirical dimensions: Reconstructibility, Persistence, Causal Leverage, Value Specificity, Coordinate Stability, and Introspective Access. | New theoretical framework demonstrates empirical necessity for additional axes. |
| **DR-0016** | Treat Execution Batch Shape and Floating-Point Precision as Experimental Variables | accepted | In long BF16 recurrent trajectories, computational execution geometry ($B=1$ vs $B=5$) causes finite-precision trajectory bifurcation while preserving aggregate state-space geometry ($C_R$). | Upstream deterministic accumulation fixes are implemented in CUDA GEMM kernels. |
| **DR-0017** | Freeze Horizon 2 Core (S10–S13) | accepted | The empirical ladder of reconstructibility, persistence at 2W, causal steering ($P=+74.10$), value specificity ($\Delta P=+38.49$), coordinate loss on $u_0$, state reorientation ($C_R \to 0.12$), and contemporaneous steerability ($V^{(N)}>0$) is confirmed across 11,520 records and 10,000 bootstrap draws. | Reopened only under an explicit freeze-reopening criterion (reproducibility defect or stronger matched control). |
| **DR-0018** | Require Matched Nonprivileged Observer for All Introspective Access Claims (S14+) | accepted | Distinguishes genuine privileged internal self-access from public-token inference or post-training verbal narrative capture. Target model must exceed the best nonprivileged observer. | Never; this is a non-negotiable epistemic requirement. |
| **DR-0019** | Standardize on Balanced Order Permutation (BOP) and Contemporaneous POST Controls for Metacognition Assays (S14+) | accepted | Uncalibrated reporting suffers from first-option order bias, and static POST controls confound intervention timing with state age. BOP cancels order bias (100% visible accuracy) and contemporaneous donor evolution isolates causal timing. | Replaced if an architectural read-head or fine-tuned reporting adapter natively eliminates presentation-order asymmetry. |
| **DR-0020** | Adopt the Seven-Way Horizon 2 Theoretical Taxonomy and Freeze Horizon 2 (S10–S14) | accepted | S14 establishes that possessing post-decision RG-LRU content reproduces the metacognitive report ($\Delta M_{\text{timing}} \approx 0$, $p_{\text{TOST}} = 0.0048$), proving that state-conditioned reportability does not imply historical provenance discrimination. Horizon 2 is fully frozen. | Reopened only under explicit freeze-reopening criteria (e.g. higher-order monitor/workspace dissociation or developmental training). |

---

# Pending / Resolved Decisions

| ID | Decision needed | Dependency | Target sprint | Status |
|---|---|---|---|---|
| **PD-01** | Primary matched base/instruct model family | model inventory and profile | S01–S03 | **RESOLVED** (Qwen2.5:3b for H1; RecurrentGemma-2B for H2) |
| **PD-03** | Level 1 state representation: JSON, free text, or hybrid | E03 pilot | S04 | **RESOLVED** (Structured JSON state schema) |
| **PD-04** | First native recurrent substrate | state-access prototype | S10 | **RESOLVED** (`google/recurrentgemma-2b` via DR-0014) |
| **PD-05** | Recurrent adapter write/read locations | native-state findings | S13–S15 | Pending S14 findings |
| **PD-06** | Organism v0 core architecture | Level 2 gate | S18 | Pending Level 2 synthesis |
| **PD-07** | Initial endogenous variables | regulation pilot | S17–S18 | Pending Level 2 synthesis |
| **PD-08** | Public release tier for trained artifacts | safety/welfare review | before release | Pending Level 3 bring-up |