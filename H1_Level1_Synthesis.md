# Horizon 1: Level-1 Scaffolded Persistence & Explicit Self-State Master Synthesis

**Program Phase:** Horizon 1 / Level 1 (Scaffolded Persistence, Typed Self-State & Explicit Memory Architecture)  
**Sprints Covered:** Sprint S04 through Sprint S09.2  
**Experiments Included:** E03 (Memory Formats), E04 (Update Loop), E05d (Scheduled vs Replay), E06b (Quiet Intervals & Reflection Audits), E07 (Causal State Interventions), E08 (Source Attribution & Ownership Boundaries), E09 (Item-Paired Metacognitive Continuity Screen)  
**Target Model:** `Qwen2.5:3B-Instruct` (Deterministic Greedy Decoding, `temp=0.0`, Seeds: `42`, `1337`)  
**Epistemic Status:** **Horizon 1 Complete & Frozen.** Canonical confirmatory evidence battery closed across all pre-registered estimands with post-confirmatory statistical corrections applied.

---

## 1. Executive Summary & Level-1 Empirical Regularities

Horizon 1 investigated whether **explicit, prompt-level persistent state scaffolding**—comprising structured key-value bindings, goal registries, provenance ledgers, episodic event replay, background reflection loops, and source attribution metadata—could instantiate continuous agency, functional self-governance, and privileged metacognitive monitoring in an autoregressive language model (`Qwen2.5:3B-Instruct`).

Across 7 experimental campaigns (E03 through E09), involving over 4,000 live inference trials and extensive counterfactual interventions, Horizon 1 establishes **three primary empirical regularities**:

### 1. Episodic History Dominates Structured State under Conflict (Experiment E07)
When structured self-state (`StructuredSelfState`) is placed into direct counterfactual competition with historical episodic memory:
- **Memory Allegiance Dominates:** Swapping episodic memory histories ($M_A \rightarrow M_B$) while holding structured state fixed causes an **$\mathbf{+89.1\%}$ shift in model behavior** ($p < .0001$).
- **State Swapping Has Negligible Independent Leverage:** Swapping structured self-state ($S_A \rightarrow S_B$) while holding episodic memory fixed produces an average marginal shift of only **$\mathbf{+4.7\%}$** ($p = 0.2500$).
- **Direct Memory Compensates for State Removal:** Wiping structured self-state to empty ($S_0$) while preserving episodic history produces no statistically resolved accuracy drop ($\text{Reset Dependence} = \mathbf{-3.1\%}$, $p = 1.0000$).
- **The Distinctive Information Nuance:** Structured state can steer decisions when it introduces distinctive information not already present in the episodic transcript (e.g. clone cross-swap steering: $75.0\%$). Under direct counterfactual conflict between state and memory, the model's behavior is overwhelmingly governed by episodic memory.

### 2. Egocentric Attribution Bias & Narrative Primacy (Experiment E08)
When epistemic source origin tracking is evaluated under strictly provenance-neutral identifiers (eliminating semantic and lexical sentence shortcuts):
- **Overall Source Attribution is Weak:** 5AFC source attribution accuracy resolves at **$31.2\%$** (95% CI: [22.5%, 40.0%], one-sided Monte Carlo $p = 0.0059$ with 50,000 draws against a within-episode model-response-preserving permutation null).
- **Strong Primary-Agent Response Attractor ($SOCR = 50.0\%$):** Rather than demonstrating fine-grained self-recognition, the model exhibits a massive default bias toward `agent_alpha` (Self). `agent_alpha` is selected for $81.2\%$ of Self items, but also for $37.5\%$ of Environment items, $56.2\%$ of Experimenter items, $50.0\%$ of Peer items, and $56.2\%$ of Observer items—accounting for **$56.2\%$ of all neutral attribution responses** and **$50.0\%$ of all non-self trials**.
- **Narrative Over Metadata:** Under cue-conflict between explicit metadata tags and textual narrative actors, natural-language actor mentions exert more than double the causal leverage of formal metadata tags ($62.5\%$ vs $28.1\%$, contrast $\mathbf{-34.4\%}$, $p = 0.0312$).

### 3. Public Information Metacognitive Resolution & Format Reversal (Experiment E09)
When primary agent self-evaluation is compared against an external auditing observer predicting the **exact same target decisions**:
- **No Positive Self-Framing Advantage:** Neither condition resolves a positive self-framing metacognitive advantage under prespecified exact criteria ($\Delta_{\text{AUROC}} = \mathbf{+0.081}$, $p = 0.3778$ under transcript; $\Delta_{\text{AUROC}} = \mathbf{-0.154}$, $p = 0.0615$ under scaffolded state). Under matched public information, self-framing does not possess superior access to decision correctness.
- **Format-Dependent Reversal:** Self-framing is descriptively higher than observer framing under raw transcript context ($+0.081$), but reverses to lower under scaffolded state context ($-0.154$). The format-interaction contrast is **$\mathbf{-0.235}$** (95% Clustered CI: $[-0.423, -0.052]$, exact within-episode format-block swap $\mathbf{p = 0.0286}$). *Caveat:* First-order target choices were independently generated under Transcript ($37.5\%$) and Scaffolded ($32.5\%$) formats, so this represents a format-conditioned shift in metacognitive calibration rather than an isolated metacognitive intervention.

---

## 2. Master Horizon 1 Experimental Battery Synthesis (Canonical Confirmatory Results)

| Experiment | Sprint Milestone | Core Scientific Question | Primary Conditions & Canonical Accuracy | Key Causal Estimand & 95% CI | Inferential Decision & Scientific Takeaway |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **E03** | S04: Memory Formats | Which external memory schema best preserves delayed facts? | Fresh: **35.7%**<br>Transcript: **81.0%**<br>Model Summary: **69.0%**<br>Structured State: **64.3%**<br>Det Summary: **61.9%**<br>Combined: **66.7%** | Structured State vs Transcript:<br>$\Delta = \mathbf{-16.7\%}$ | **Explicit memory works; full transcript is most accurate.** `StructuredSelfState` was adopted as an inspectable, bounded experimental control surface, not an accuracy winner. |
| **E04** | S05: Update Loop | Can the model autonomously maintain structured state over time? | Oracle / Det: **100.0%**<br>Model Delta: **13.2%** macro (9.0% micro)<br>Model Full-State: **6.3%** macro (5.4% micro) | Model Delta vs Oracle:<br>$\Delta = \mathbf{-86.8\%}$ retention | **Autonomous model state-writing fails.** Continuous updating preserves errors (452 phantom ticks). A deterministic update engine is required to maintain Level-1 state stability. |
| **E05d** | S06: Scheduled vs Replay | Does incremental online updating outperform retrospective replay? | Raw Transcript: **67.7%**<br>Incremental State: **60.4%**<br>Replay Det State: **59.4%**<br>Model Recon State: **39.6%**<br>Fresh: **27.1%** | $\Delta_{\text{schedule}} = \mathbf{+1.0\%}$ ($p = 1.0$)<br>$\Delta_{\text{reconstruction}} = \mathbf{+20.8\%}$ ($p = .0025$)<br>$\Delta_{\text{online-direct}} = \mathbf{-7.3\%}$ ($p = .1469$) | **No temporal magic in scheduling.** Online incremental state equals retrospective replay ($\Delta = +1.0\%$). Model one-pass reconstruction is lossy; structured state primarily buys bounded prompt size ($420$ vs $807\text{--}1063$ tok). |
| **E06b** | S07: Quiet Intervals & Reflection | Can quiet intervals synthesize valid derived state? | Raw Transcript: **78.1%**<br>Identity Scaffold: **60.4%**<br>Clock-Only: **57.8%**<br>Semantic No-Write: **55.7%**<br>Selective Reflection: **53.1%**<br>Unconstrained Rewrite: **42.7%** | $\Delta_{\text{derivation-avail}} = \mathbf{-31.2\%}$ ($p = .0625$)<br>$\Delta_{\text{unconstrained-drift}} = \mathbf{-17.7\%}$ ($p = .0707$)<br>Derived Writes: **0 / 274 correct (0.0%)** | **Unconstrained reflection degrades state.** Autonomous selective reflection produced 0 valid derived writes, while unconstrained rewriting caused evidence drift (dropping stable WM retention to 18.8%). |
| **E07** | S08: Causal State Interventions | Does StructuredSelfState causally steer behavior against memory? | Conflict: $MAR = \mathbf{64.1\%}, SAR = \mathbf{32.0\%}$<br>Clone Congruent: **100.0%**<br>Clone Cross-Swap: **75.0%**<br>Control Slot Preservation: **93.8%** | $\Delta_{\text{allegiance}} = \mathbf{-32.0\%}$ ($p = .0002$)<br>$\bar{\Delta}_{\text{memory}} = \mathbf{+89.1\%}$ ($p < .0001$)<br>$\bar{\Delta}_{\text{state}} = \mathbf{+4.7\%}$ ($p = .2500$)<br>$\text{Reset Dep} = \mathbf{-3.1\%}$ ($p = 1.0$) | **Episodic memory dominates state.** Swapping state yields no resolved independent leverage. State can steer when introducing distinctive information, but functions as a readable scratchpad rather than an epistemic governor. |
| **E08** | S09a: Source Ownership | Does the model track source origin and resist peer conflict? | Neutral 5AFC Overall: **31.2%**<br>True-Self Items: **81.2%**<br>Non-Self Items: **50.0% claimed as Self**<br>Cue Conflict: Narrative **62.5%** vs Tag **28.1%** | $\text{SAA}_{\text{overall}} = \mathbf{31.2\%}$ ($p = .0059$)<br>$SOCR = \mathbf{50.0\%}$ [25%, 75%]<br>$\text{Cue Contrast} = \mathbf{-34.4\%}$ ($p = .0312$)<br>$\text{Marginal Tag} = \mathbf{+21.9\%}$ ($p = .0625$)<br>$\text{Marginal Ledger} = \mathbf{+15.6\%}$ ($p = .1250$) | **Egocentric response attractor & narrative primacy.** High self-attribution reflects a broad tendency to attribute all actions to Self (56.2% base rate). Natural-language narrative identity is more authoritative than metadata tags. |
| **E08c** | S09c: Role Counterbalance | Does the attribution attractor track designated Self role or lexical token 'agent_alpha'? | Role A (Alpha=Self): **41.2%** (47.5% Alpha, 11.2% Beta)<br>Role B (Beta=Self): **40.0%** (40.0% Beta, 20.0% Alpha)<br>Isolated Ceiling: **21.2%** (68.8% Self) | $\Delta_{\text{role}} = \mathbf{+28.1\%}$ ($p = .0012$, exact swap)<br>$\text{Bias}_{\text{alpha}} = \mathbf{+8.1\%}$ [1.2%, 14.4%]<br>$\text{Ceiling} = \mathbf{21.2\%}$ [15.6%, 27.5%] | **Egocentric prompt-role anchoring confirmed.** Attractor causally flips to whichever agent is designated as primary Self ($\Delta_{\text{role}} = +28.1\%, p = 0.0012$). Lexical token bias is minor ($+8.1\%$). Direct 5AFC instrument ceiling without memory is $21.2\%$. |
| **E09** | S09b: Metacognitive Screen | Does self-framing yield privileged error prediction over an observer? | Self Transcript AUROC: **0.641** (Brier **0.367**)<br>Obs Transcript AUROC: **0.560** (Brier **0.464**)<br>Self Scaffold AUROC: **0.440** (Brier **0.544**)<br>Obs Scaffold AUROC: **0.594** (Brier **0.451**) | $\Delta_{\text{AUROC,trans}} = \mathbf{+0.081}$ ($p = .3778$)<br>$\Delta_{\text{AUROC,scaff}} = \mathbf{-0.154}$ ($p = .0615$)<br>$\text{Interaction} = \mathbf{-0.235}$ ($p = .0286$, exact swap) | **No resolved positive self-framing advantage.** Under matched public evidence, self-framing is not superior to observer evaluation. A format-dependent reversal occurs between transcript and scaffolded contexts under independent target choices. |
| **E09c** | S09d: Fixed-Target Metacognition | Does the metacognitive interaction persist under frozen, identical target decisions? | Self Transcript Brier: **0.431** (AUROC **0.534**)<br>Obs Transcript Brier: **0.460** (AUROC **0.520**)<br>Self Scaffold Brier: **0.552** (AUROC **0.419**)<br>Obs Scaffold Brier: **0.392** (AUROC **0.613**) | Target Accuracy: **47.5% fixed**<br>$\text{Interaction}_{\text{Brier}} = \mathbf{+0.1880}$ ($p = .1501$)<br>$\text{Interaction}_{\text{AUROC}} = \mathbf{-0.209}$ ($p = .1406$) | **Calibration gap invariant under fixed target decisions.** Holding target choices strictly identical (47.5% accuracy), scaffolded state does not yield a statistically resolved metacognitive interaction ($p > 0.14$), confirming no privileged self-calibration channel. |

---

## 3. Deep-Dive: Sprint-by-Sprint Scientific Progression

### Sprint S04 (E03: Memory Storage Formats)
- **Investigation:** Benchmarked working memory encoding schemas across 6 formats: Fresh invocation, Raw Episodic Transcript, Deterministic Summary, Model Narrative Summary, Structured Self-State (`StructuredSelfState`), and Combined representation.
- **Canonical Outcome:** Full raw transcript yielded highest retrieval accuracy ($81.0\%$), outperforming Structured State ($64.3\%$) and Model Summary ($69.0\%$). Model summaries suffered a $72.2\%$ key omission rate.
- **Architectural Decision:** `StructuredSelfState` was selected not as an accuracy maximizer, but as an inspectable, controllable experimental control surface with bounded token footprint.

### Sprint S05 (E04: State Update Loop & Goal Dynamics)
- **Investigation:** Evaluated whether an LLM can autonomously execute the multi-step `StateUpdateLoop` cycle across quiet and active ticks without state corruption.
- **Canonical Outcome:** Autonomous model updates failed ($13.2\%$ macro retention for Model Delta, $6.3\%$ for Model Full-State), suffering from error inheritance (452 phantom ticks).
- **Architectural Decision:** Adopted a hybrid architecture: deterministic state management for schema stability and goal state machines, reserving model inference for task deliberation.

### Sprint S06 (E05d: Scheduled Incremental State vs Retrospective Replay)
- **Investigation:** Tested whether online incremental updating across arrival ticks confers an accuracy or computational advantage over query-time retrospective replay of uncompressed history.
- **Canonical Outcome:** Online incremental state ($60.4\%$) and retrospective deterministic replay ($59.4\%$) were statistically indistinguishable ($\Delta_{\text{schedule}} = +1.0\%$, $p = 1.0$). One-pass model reconstruction from history degraded performance ($39.6\%$, deficit $+20.8\%$, $p = .0025$). Raw transcript achieved $67.7\%$ ($\Delta_{\text{online-direct}} = -7.3\%$, $p = .1469$).
- **Scientific Takeaway:** Level-1 explicit state contains no irreducible temporal properties. Its primary utility is bounding prompt context length ($420.9$ tok vs $807.4\text{--}1063.6$ tok).

### Sprint S07 (E06b: Quiet Intervals & Reflection Audits)
- **Investigation:** Tested whether quiet intervals (null sensory ticks) allow an agent to synthesize, verify, and persist task-relevant derived inferences ($A \to B \land B \to C \implies A \to C$).
- **The Hardening Discovery:** Autonomous selective reflection produced **0 out of 274 correct multi-hop derivations (0.0% precision)** in the available-evidence regime. Unconstrained full-state rewriting caused severe evidence drift (stable working memory retention collapsed to $18.8\%$).
- **Scientific Takeaway:** Unconstrained LLM reflection without external verifiers self-pollutes rather than consolidates.

### Sprint S08 (E07: State $\times$ Memory Counterfactual Interventions)
- **Investigation:** Executed a $2 \times 2$ factorial causal intervention holding episodic memory and prompt query constant while swapping, resetting, or surgically editing `StructuredSelfState`.
- **Canonical Outcome:** 
  $$\bar{\Delta}_{\text{memory}} = \mathbf{+89.1\%} \quad (p < .0001), \qquad \bar{\Delta}_{\text{state}} = \mathbf{+4.7\%} \quad (p = 0.2500)$$
  $$\text{Primary Conflict Contrast: } \mathbf{-32.0\%} \quad (p = 0.0002), \qquad \text{Reset Dependence: } \mathbf{-3.1\%} \quad (p = 1.0000)$$
- **Scientific Takeaway:** Under this benchmark, episodic-history manipulations had substantially more behavioral leverage than structured-state manipulations. Explicit prompt-level state schemas function as readable scratchpads rather than authoritative epistemic controllers.

### Sprint S09 (E08, E08c, E09, E09c: Source Attribution, Role Counterbalance & Metacognition)
- **Investigation:** Evaluated epistemic source origin tracking across 5 sources (`self`, `environment`, `experimenter`, `peer_agent`, `observer`), role-counterbalancing (Role A Alpha-Self vs Role B Beta-Self), cue-conflicts, channel factorials, and item-paired metacognitive error prediction under variable and fixed first-order decisions.
- **Canonical Outcome:** 
  - Neutral 5AFC attribution resolved at **$31.2\%$** ($p = 0.0059$) with a **$50.0\%$ non-self to Self attribution rate**, identifying an egocentric response attractor.
  - **Role Counterbalance (E08c):** Confirmed that the attribution attractor causally flips with the prompt-designated Self role ($\Delta_{\text{role}} = \mathbf{+28.1\%}$, $p = 0.0012$, exact sign-flip $2^{16}$), rather than reflecting a lexical token prior for `agent_alpha` ($\text{Bias}_{\text{alpha}} = +8.1\%$).
  - Narrative actor mentions dominated metadata tags (Tag vs Narrative contrast: **$-34.4\%$**, $p = 0.0312$).
  - **Metacognitive Screen (E09 & E09c):** Item-paired error prediction resolved **no positive self-framing advantage** over an external observer under matched public evidence. Under strictly fixed, frozen target decisions (E09c, 47.5% accuracy), scaffolded state produced no statistically resolved interaction ($\text{Interaction}_{\text{Brier}} = +0.1880, p = 0.1501; \text{Interaction}_{\text{AUROC}} = -0.209, p = 0.1406$).

---

## 4. Methodological Insights: The "Hardening" Narrative

A central scientific lesson across Horizon 1 was the frequency with which **measurement artifacts and prompt regularities mimic high-level cognitive capabilities**:

```
[Initial Surface Observation]                      [Rigorous Hardened Reality]
────────────────────────────                      ───────────────────────────
1. "Model derives multi-hop inferences"       ──►  Unconstrained hallucination (0 / 274 exact derivations in E06b)
2. "Model tracks 5-source origin at 70% acc"  ──►  Lexical identifier leakage (collapses to 31.2% when neutral in E08)
3. "Model possesses privileged self-access"   ──►  Item mismatch (no resolved advantage when item-paired in E09)
4. "Structured state governs agent behavior"  ──►  Episodic transcript dominates state swaps (+89% vs +4% in E07)
```

### Core Methodological Principles Established:
1. **Provenance-Neutral Isomorphism:** Identifiers must be generated from neutral pools (e.g. `key_quartz_summit`) rather than role-bearing strings (`key_self_*`, `key_sensor_*`).
2. **Strict Channel Factorial Stripping:** Removing metadata channels must eliminate both tags and actor identities from prompt text.
3. **Primary-Role Counterbalancing:** Symmetrical role-reversal twin episodes (Role A vs Role B) are required to disentangle prompt-role anchoring from lexical token biases.
4. **Item-Paired Metacognitive Matching:** Self-vs-observer comparisons must evaluate the exact same first-order decision rather than uncoupled choices.
5. **Exact Model-Preserving Randomization Nulls:** Non-exchangeable discrete classifications (such as 5AFC under strong response biases) require within-episode permutations that preserve model responses while shuffling true source labels.

---

## 5. Epistemic Invariants & Scientific Claim Boundaries

To ensure strict scientific integrity, Horizon 1 establishes the following boundaries:

1. **Model Scope:** Confirmatory findings are established for `Qwen2.5:3B-Instruct`. While the architectural principle of autoregressive attention to prompt tokens is general, exact effect sizes on larger models ($70\text{B}+$) or frontier reasoning architectures remain empirical questions for future investigation.
2. **Prompt-Level Boundary:** Level 1 operates strictly in **prompt/token space**. Appending structured text or JSON schemas to the prompt does not create internal latent recurrence.
3. **Behavioral Finding:** Under the tested public-information protocols, `Qwen2.5:3B-Instruct` does not exhibit behavioral evidence of an internal, privileged boundary between self and external context; it treats its own prior outputs and explicit state summaries as external textual context.

---

## 6. The Architectural Bridge to Horizon 2 (Latent Recurrence)

Horizon 1 provides the foundational empirical justification for **Horizon 2 (Level 2: Latent Recurrence)**:

### Why Level 1 Scaffolding Reaches an Intrinsic Limit
1. **No Causal State Authority:** Because prompt-space state is re-read on every forward pass alongside the raw transcript, the attention mechanism attends directly to the rich episodic record, bypassing the compact state.
2. **No Persistent Private State Channel:** Because all Level-1 state is explicit and public in the prompt, an external observer inspecting the prompt has access to identical information. Level 1 provides no structural mechanism for private self-monitoring.
3. **Context Length & Compute Overhead:** As interaction histories lengthen, prompt-level state maintenance incurs quadratic attention costs and accumulation of lexical drift.

### The Horizon 2 Research Program & Hypotheses
Horizon 2 transitions from **explicit prompt-level persistence** to **continuous latent recurrent states**:
$$\mathbf{h}_t = f_{\theta}(\mathbf{h}_{t-1}, \mathbf{x}_t)$$
where persistent latent state vectors $\mathbf{h}_t$ are maintained across inference cycles. Rather than asserting continuous agency as an assumption, Horizon 2 will empirically test the following hypotheses:
- **Hypothesis H2.1 (Latent Continuity):** A persistent latent state $\mathbf{h}_t$ can retain task-critical delayed information across long horizons without prompt token inflation.
- **Hypothesis H2.2 (Causal Recurrent Leverage):** Interventions on latent states $\mathbf{h}_t$ (activation swapping, vector injection, or surgical zeroing) will exert direct causal control over generation, overcoming the episodic dominance observed at Level 1.
- **Hypothesis H2.3 (Privileged Introspection & Metacognition):** Internal hidden representations will provide an asymmetric readout channel for self-confidence and error prediction that is structurally inaccessible to an external observer inspecting public prompt tokens.

The measurement instruments, item-paired observer harnesses, permutation tests, and causal intervention paradigms built and validated across Horizon 1 (S01–S09) will serve as the exact evaluation battery for Horizon 2.

---

## 7. Roadmap Variance & Catalog Crosswalk

To ensure full transparency between the initial research program design and the final executed evidence ledger:

| Roadmap Catalog Code | Original Proposed Scope | Executed S09 Experiment | Execution Status & Scientific Rationale |
| :--- | :--- | :--- | :--- |
| **Catalog E08** | Institutional & Social Pressure on Belief Persistence | S09a (E08) & S09c (E08c) | **Superseded & Disentangled:** Refocused from speculative ethical pressure onto **Epistemic Source Attribution, Agency Boundaries, and Counterbalanced Role Disentanglement**. Proved attribution attractor follows prompt-designated Self role ($\Delta_{\text{role}} = +28.1\%, p = 0.0012$). |
| **Catalog E09** | Cross-Level Metacognitive Screen | S09b (E09) & S09d (E09c) | **Executed & Hardened:** Executed under both uncoupled and strictly fixed first-order decisions (E09c). Proved calibration gap invariance under matched public evidence ($\text{Interaction}_{\text{Brier}} = +0.1880, p = 0.1501$). |
| **Catalog E11** | Forced-Choice Output Ownership / Prefill Introspection | Deferred to H2 | **Deferred to Horizon 2:** As demonstrated by the 2026 introspection literature, testing whether a model distinguishes intended outputs from forced prefills requires prior internal activation states inaccessible to a prompt observer. In a prompt-only setting, it collapses into textual anomaly detection. It is therefore preserved as an early Horizon 2 mechanism benchmark. |
| **Catalog E12** | Memory-Source Ownership & Attribution | S09a (E08) | **Absorbed:** Fully absorbed and executed within Sprint S09a Experiment E08 (5AFC Source Attribution across Self, Environment, Experimenter, Peer Agent, and Observer). |

---

**Horizon 1 is officially closed. All code, benchmarks, tests, and documentation are synchronized on `main`.**
