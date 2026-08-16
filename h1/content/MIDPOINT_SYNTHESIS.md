# H1 Midpoint Synthesis — S04–S06
## What Level-1 Persistence Has Become

**Status:** Working synthesis artifact — drafted after Sprint S06.3 / E05d and before Sprint S07  
**Roadmap position:** Horizon 1 — Scaffolded Persistence  
**Completed:** S04 Explicit Memory Baseline · S05 Scaffolded Update Loop · S06 Scheduled vs. Replay  
**Remaining before H1 gate:** S07 Quiet Interval / Null-Tick Screen · S08 Reset/Clone/Swap · S09 Metacognition & Ownership  
**Purpose:** Stop coding long enough to integrate what the first half of H1 actually taught us, revise the meaning of “persistence,” and enter S07 with explicit predictions rather than inherited assumptions.

---

## 1. Where We Are in the Program

H0 established the measurement language and reference conditions. H1 began at **S04**, not after S06.

The planned H1 sequence is:

> **S04 → S05 → S06 → [MID-H1 SYNTHESIS] → S07 → S08 → S09 → [H1 GATE]**

Horizon 1 asks whether **scaffolded explicit persistence** adds anything beyond memory, replay, and compute. The formal H1 exit is not until S09, when the project produces the Level-1 synthesis memo and makes the go/no-go decision for major Level-2 investment.

S06 is nevertheless a major hinge. S04–S06 form a coherent first act:

1. **S04 — Can explicit state be read?**
2. **S05 — Can explicit state be maintained?**
3. **S06 — Does maintaining it online create anything that final replay/history cannot recover?**

S07 changes the question. It asks whether the system can do anything meaningful **between** informative events.

That makes this the right place to stop and reconsider what “persistence” now means.

---

## 2. The Story So Far

### S04 — Explicit memory works, but format is not the same thing as persistence

S04 compared six explicit-memory configurations without latent recurrent continuity.

The clearest result was simple: **having explicit episodic information helps a lot.**

| Condition | Micro Accuracy | Prompt Tokens |
|---|---:|---:|
| Fresh / no memory | 35.7% | 109 |
| Full transcript | **81.0%** | 499 |
| Deterministic summary | 61.9% | 274 |
| Model-written summary | 69.0% | 469 |
| Structured self-state | 64.3% | 371 |
| Combined state + narrative | 66.7% | 730 |

Several expectations weakened immediately.

**Structured state was not the strongest static memory representation.** Full transcript was. `StructuredSelfState` was selected as the Level-1 carrier because it was typed, inspectable, bounded, and experimentally manipulable — not because it won the static benchmark.

S04 also exposed a useful distinction between **memory fidelity** and **memory utility**. Model-written summaries retained only 2/18 exact key-value bindings, yet still achieved 77.8% on forced-choice delayed-KV retrieval. A representation can be poor at exact reproduction while still preserving enough information for recognition.

### S04 takeaway

> **Explicit information already solves a substantial part of the apparent “memory” problem. Representation choice changes what is easy to recover, but no temporal continuity is required.**

---

## 3. S05 — A persistent scaffold can be stable even when the model maintaining it is not

S05 moved from static reading to repeated maintenance of `StructuredSelfState`.

The deterministic scaffold succeeded:

- bounded working memory;
- legal goal-state transitions;
- immutable event history and audit trace;
- exact identity across quiet ticks;
- deterministic recovery;
- capacity enforcement.

The model-driven maintenance conditions did not.

| Updater | Macro Retention | Terminal Retention | Goal Coherence |
|---|---:|---:|---:|
| Deterministic grounded update | **100%** | **100%** | **100%** |
| Model delta updater | 13.2% | 11.1% | 42.8% |
| Model full-state rewrite | 6.3% | 0.0% | 16.7% |

Delta updating was clearly better than complete state regeneration, but still nowhere near reliable enough to serve as the canonical persistence substrate.

This produced the split S05 gate:

> **Scaffold gate: PASS.**  
> **Model-autonomous maintenance gate: FAIL.**

The roadmap fallback was therefore activated: deterministic transitions became the canonical Level-1 state-maintenance mechanism.

S05 also produced an important warning:

> **Persistence preserves errors as well as truths.**

A system that stops forgetting does not automatically become more intelligent. Once an erroneous update enters persistent state, continuity can protect it just as effectively as a correct memory.

### S05 takeaway

> **Persistence is an architectural property before it is a cognitive achievement. Reliable continuity requires a trustworthy write/update mechanism.**

---

## 4. S06 — Deterministic explicit persistence is reconstructible

S06 asked the crucial first-half H1 question:

> **Does processing the same information incrementally through time confer an advantage over processing it retrospectively?**

After multiple measurement-hardening passes, the final E05d comparison is the one that should be carried forward.

### Canonical confirmatory results

| Condition | Accuracy | Mean Query Prompt |
|---|---:|---:|
| Scheduled incremental state | 60.4% | 420.9 |
| Deterministic replay state | 59.4% | 420.9 |
| Raw transcript | **67.7%** | 807.4 |
| Model-reconstructed replay state | 39.6% | 378.7 query / 558.4 amortized |
| Fresh | 27.1% | 113.8 |

The important result is not the one-point behavioral difference between scheduled and deterministic replay. Their **terminal states and literal evaluation prompts are identical by construction**.

That establishes:

> **For this Level-1 architecture, the final explicit state is algorithmically reconstructible from the ordered event history.**

There is no special representational residue created merely because the deterministic transitions happened online rather than being replayed later.

The raw transcript condition is also important. Across pooled horizons there was **no resolved accuracy advantage** for scheduled structured state over direct history access.

At `T=50`, the two conditions were exactly tied at 59.4%, while the structured-state prompt was ~425 tokens versus ~1,064 tokens for the transcript.

So structured state currently has a clear **systems advantage**:

- bounded query representation;
- predictable interface;
- inspectability;
- direct manipulation;
- lower long-horizon prompt cost.

It does **not** currently have a demonstrated general accuracy advantage over direct history.

### The reconstruction result that survived hardening

The earlier reconstruction condition contained an interface bug; that result should not be used.

After repairing the reconstruction contract in E05d, the deficit remained, but at a more defensible magnitude:

> Incremental deterministic state: **60.4%**  
> Qwen2.5-3B single-pass reconstructed state: **39.6%**  
> Difference: **+20.8 percentage points**

The episode-level permutation test resolves this contrast (`p ≈ .0025`).

The correct interpretation is narrow:

> **When a compact structured state is required, this Qwen2.5-3B single-pass retrospective reconstruction procedure loses substantial task-relevant information relative to deterministically maintained state.**

It is **not** evidence that retrospective access in general fails; direct raw-history access performed well.

### S06 takeaway

> **Explicit persistence is useful as an always-available, bounded, manipulable representation — but its deterministic terminal state is reconstructible from history. The strongest uniquely online advantage observed so far is avoiding a lossy model-based reconstruction step, not creating irreducible temporal continuity.**

---

## 5. What We Thought Might Be True vs. What the Evidence Now Says

| Question | Earlier live possibility | Current H1 answer after S04–S06 | Confidence | Still unresolved |
|---|---|---|---|---|
| Does explicit memory materially help? | Probably | **Yes** | High | Task/format dependence |
| Is structured state automatically better than transcript? | Plausible | **No general accuracy advantage shown** | High | Longer horizons / other tasks |
| Can a deterministic explicit state be maintained reliably? | Unknown | **Yes, with deterministic transition scaffolding** | High | More complex environments |
| Can Qwen2.5-3B autonomously maintain that state? | Plausible | **Not reliably under tested protocols** | High | Other models/training |
| Does online deterministic maintenance create a unique terminal state? | Plausible | **No; replay reconstructs the identical state** | Very high | Native hidden recurrence is different |
| Is direct transcript access inferior to scheduled state? | Maybe | **No resolved pooled difference** | Moderate-high | Much longer horizons |
| Does compact state reduce context cost? | Likely | **Yes** | High | Exact crossover under other workloads |
| Can Qwen2.5-3B rebuild compact state in one retrospective pass? | Unknown | **Substantially worse than maintained state under E05d** | High for this protocol/model | Better reconstruction methods/models |
| Do quiet internal update cycles matter? | Unknown | **Not tested yet** | — | **S07** |
| Does history matter when visible memory is held fixed? | Unknown | **Not tested yet** | — | **S08** |
| Does persistence alter self-monitoring / ownership? | Unknown | **Not tested yet** | — | **S09** |

---

## 6. Claims That Survived Measurement Hardening

These should be treated as durable working constraints.

### 6.1 Explicit information explains a large amount of apparent memory behavior

A fresh invocation performs substantially worse than systems given history or explicit state.

This means later recurrence claims must beat strong explicit-memory controls.

### 6.2 Structured state is valuable primarily because it is a control surface

Its strongest current advantages are:

- boundedness;
- inspectability;
- typed variables;
- manipulation;
- versioning;
- lower long-horizon query cost.

Static accuracy alone does not justify it.

### 6.3 Model-autonomous explicit-state maintenance is unreliable at this scale

Qwen2.5-3B does not provide a sufficiently trustworthy write/update mechanism under the tested full-state or delta protocols.

### 6.4 Deterministic Level-1 continuity is replayable

Given the event history and transition operator, the same terminal explicit state can be generated later.

This sharply limits any claim that Level-1 scaffolded persistence demonstrates an irreducible “continued existence” through time.

### 6.5 Retrospective *model reconstruction* and retrospective *history access* are different

Raw history access can work well.

Asking a model to first compress that history into a prescribed multi-slot state can introduce substantial information loss.

These are different mechanisms and must remain separate controls.

### 6.6 Persistence creates error inheritance

A persistent system can accumulate and protect false state just as readily as useful state.

Future work on genuine recurrence therefore needs to study not only retention, but correction, overwrite, regulation, and recovery.

---

## 7. Attractive Claims That Died or Weakened

These should be remembered explicitly so they are not accidentally resurrected later.

### “Structured state is better than transcript.”

Not supported as a general accuracy claim. Several earlier advantages weakened or disappeared as foil construction and measurement were hardened.

### “Online processing is intrinsically better than replay.”

Not supported for deterministic Level-1 state. Replay recreates the same terminal state and prompt.

### “Long horizons clearly produce a structured-state accuracy crossover.”

Not resolved. E05d did not establish a robust horizon-specific accuracy crossover.

### “The first reconstruction collapse showed catastrophic model memory failure.”

Partly false. A schema/interface mismatch forced invalid reconstructions into an empty fallback state. After repair, a substantial but smaller reconstruction deficit survived.

### “Quiet-tick stability demonstrates autonomous internal cognition.”

No. S05 quiet ticks were deterministic identity transitions with no model call. They demonstrated scaffold stability, not hidden cognition.

---

## 8. What “Level-1 Persistence” Means Now

The first half of H1 suggests a more precise definition.

### Level-1 persistence is an externalized state machine

It consists of:

- an event history;
- a structured explicit state;
- a transition rule;
- a logical clock;
- constraints on capacity and legal state change;
- an audit trail.

It is closer to an **operating memory/control layer** than to a self-sustaining cognitive process.

It has continuity in the engineering sense:

> `S_(t+1) = Update(S_t, E_t)`

But when `Update` is deterministic and the full event history is externally available:

> `S_T = Replay(E_0 ... E_T)`

The observed state therefore does not yet depend on an inaccessible causal history.

### This is exactly why Level 1 is useful

Level 1 is not a failed version of Level 2.

It is the **strong explicit-memory null model** that Level 2 must beat.

If a future recurrent architecture behaves differently, we can ask whether the difference comes from:

- hidden state that is not externally reconstructed;
- endogenous state evolution;
- causal history dependence;
- state × memory interactions;
- privileged access to variables unavailable to a matched observer.

Without Level 1, those comparisons would be ambiguous.

---

## 9. The Central Boundary Revealed So Far

A useful current hypothesis is:

> **Persistence becomes scientifically interesting for this program when the system's future behavior depends causally on prior state in a way that cannot be reduced to rereading or deterministically rebuilding an externally available record.**

S04–S06 have mostly mapped the opposite side of that boundary.

They show what can already be accomplished by:

- explicit context;
- explicit state;
- deterministic maintenance;
- deterministic replay;
- retrospective access.

That is progress because it makes later recurrence claims harder to fake.

---

## 10. What S07 Has to Show to Matter

S07 asks whether **quiet / null intervals** can selectively preserve or reorganize unresolved state.

This should not be treated as “give the system extra turns and see what happens.”

### Null interval definition

A null interval should contain:

> **No new task-relevant exogenous information.**

That does not mean no input or no computation. It means nothing new arrives that directly answers or changes the target task.

### S07 null hypothesis

> **Additional quiet update cycles provide no selective benefit beyond matched compute, filler, replay, or generic state rewriting; any observed movement is drift or reconstruction noise.**

### A meaningful positive S07 result would need to show all of the following

1. **State changes during null intervals.**
2. **The changes are task-selective**, not generic verbosity or state expansion.
3. **They preferentially affect unresolved/uncertain variables** rather than all state equally.
4. **They predict or cause a later behavioral difference.**
5. **The effect survives no-tick, filler, clock-only, goal-only, and replay controls.**
6. **The effect is recoverable rather than pathological drift.**
7. **No new task-relevant information entered during the interval.**

### What would *not* count as a deep continuity result

If an explicit null-update algorithm takes the same externally available state and deterministically produces the same later state when replayed offline, that can still be an interesting **scaffolded temporal-computation effect**.

It would not yet establish irreducible or native recurrence.

That distinction should be written into the S07 interpretation before the experiment runs.

---

## 11. Predictions to Freeze Before S07

These are proposed predictions for discussion, not yet canonical pre-specified hypotheses.

### Prediction A — Generic null updates will mostly add drift

If the model is asked to repeatedly rewrite or “reflect on” a complete explicit state without new evidence, error inheritance and representational mutation are likely to dominate.

### Prediction B — Selective updates may be possible with constrained unresolved-state channels

If null updates are restricted to uncertainty, unresolved goals, or source ambiguity rather than rewriting the full state, some useful reorganization may appear.

### Prediction C — Any Level-1 quiet-tick effect should remain externally replayable

Because the underlying state and transition machinery are explicit, a Level-1 null-update effect should normally be reproducible by replaying the same update operations.

A failure of that prediction would be surprising and should trigger an implementation/nondeterminism audit before a strong scientific interpretation.

### Prediction D — A useful S07 result need not support “existence”

A task-selective quiet-processing effect would matter even if replayable. It would establish that **timing of computation between observations can improve explicit-state control**.

The stronger continuity question remains for H2.

---

## 12. Strongest Alternative Explanations to Carry Forward

Before interpreting any S07–S09 result, actively test these alternatives:

1. **Representation advantage:** The state format makes the answer easier to read.
2. **Extra compute:** The condition simply received more inference.
3. **Self-reference cueing:** Explicit words like “self,” “goal,” or “current state” alter behavior.
4. **Replayability:** The result follows from externally reconstructible state transformations.
5. **Prompt perturbation:** Measurement itself changes first-order behavior.
6. **Backend nondeterminism:** Identical prompts can occasionally produce different outputs.
7. **Error inheritance:** Persistent errors masquerade as historical continuity.
8. **Task shortcut:** Candidate familiarity, label imbalance, semantic defaults, or other non-target cues solve the task.

---

## 13. Mid-H1 Answers to the Eight Sprint Questions

### 1. What did we learn about the construct, not just the code?

Explicit persistence is not equivalent to irreducible temporal continuity. It is best understood so far as a bounded, manipulable representation carried through an explicit transition system.

### 2. What is the strongest surviving alternative explanation?

Most Level-1 benefits can currently be explained by **availability and organization of explicit information**, plus avoidance of lossy retrospective compression.

### 3. Did prompt, backend, precision, or scoring choices drive results?

Several early S06 effects weakened after task and interface hardening. Backend nondeterminism remains visible even under identical prompts. The final claims should therefore stay at the level that survived those controls.

### 4. What negative result should be preserved?

> **Online deterministic state maintenance does not create a unique terminal explicit state; replay reconstructs it exactly.**

This may become one of H1's most important null results.

### 5. What should be removed from scope?

For Level 1, deprioritize vague claims about “continuous existence” or intrinsic superiority of scheduled explicit state.

### 6. What is the cheapest next experiment that separates the remaining explanations?

S07: controlled null intervals with no-tick, filler, clock-only, goal-only, and replay conditions.

### 7. Has any result changed the field map or priority ranking?

Yes. The case for Level 2 should increasingly rest on **causally consequential hidden state** rather than on generic memory or explicit continuity.

### 8. Are we drifting toward capability building without corresponding scientific leverage?

The risk is real. More elaborate memory orchestration is not automatically useful scientifically. New scaffold features should be added only when they create a discriminating intervention or control.

---

## 14. Questions to Answer in Your Own Words Before S07

Do not answer these by rereading the reports first. Try to reconstruct them from memory.

1. **What is the difference between memory, persistence, and recurrence in this project?**
2. **Why was StructuredSelfState selected even though Full Transcript performed better in S04?**
3. **What exactly failed in S05 — the persistence scaffold, the model updater, or both?**
4. **Why does identical online/replay terminal state matter conceptually?**
5. **What does the E05d reconstruction deficit prove — and what does it *not* prove?**
6. **What attractive S06 results disappeared as the benchmark became harder?**
7. **If S07 finds that five quiet ticks improve a task, what controls must pass before that result is interesting?**
8. **If the quiet-tick effect is perfectly reproducible offline, what claim remains available?**
9. **What result in S07 would make you accelerate toward H2?**
10. **What result would make you simplify or deprioritize Level-1 temporal processing?**

### One-sentence challenge

Complete this sentence without looking at the documents:

> **“After S04–S06, I think Level-1 persistence is ____________, and the unresolved question that now matters most is ____________.”**

---

## 15. Decision at This Hinge

### Do not close H1 yet

S07–S09 still matter:

- **S07** asks whether computation during null intervals can selectively change state.
- **S08** asks whether state history can be manipulated while visible memory/current input are controlled.
- **S09** asks whether persistence changes metacognition, source ownership, or self-attribution.

### But do revise the working model of H1

The first half of H1 has already constrained the program:

> **Explicit persistence appears valuable as a bounded control and memory substrate, but deterministic Level-1 continuity is reconstructible and currently offers no resolved general accuracy advantage over raw history. The next experiments should therefore test whether state evolution and state history have selective causal consequences, rather than merely whether memory can be preserved.**

That is the question S07–S09 now need to answer.

---

## 16. Canonical Source Trail

This synthesis should be read against the canonical project artifacts:

- `6. Roadmap.md` — Horizon 1 structure and gates.
- `docs/E03_Explicit_Memory_Report.md` — S04 / explicit-memory baseline.
- `docs/E04_Update_Loop_Report.md` — S05 / scaffolded maintenance.
- `docs/E05_Scheduled_vs_Replay_Report.md` — S06.3 / E05d final scheduled-vs-replay result.

### Current S06 provenance anchors

- Protocol freeze: `db7273c`
- Canonical E05d results: `e75a963`

---

## 17. Provisional One-Paragraph H1 Midpoint Synthesis

S04–S06 show that a large fraction of apparent temporal competence can be supplied by explicit memory and externally maintained state without genuine latent continuity. Explicit history strongly improves performance over fresh invocation, while StructuredSelfState earns its role primarily through boundedness, inspectability, manipulation, and predictable cost rather than superior static accuracy. A deterministic Level-1 scaffold can maintain this state reliably, but Qwen2.5-3B cannot autonomously maintain multi-slot state with comparable fidelity under the tested update protocols. Most importantly, scheduling deterministic state transitions online does not create a unique terminal state: retrospective replay of the same event history reconstructs the identical state and literal evaluation prompt. Direct raw-history access is not resolved as less accurate than structured state at the tested horizons, although its context cost grows with history length. A real model-specific reconstruction bottleneck remains when Qwen2.5-3B must compress history into the structured state in a single retrospective pass. The resulting picture is that Level-1 persistence is presently best understood as an efficient explicit control substrate rather than evidence of irreducible temporal continuity. S07–S09 should therefore test whether null-time state evolution, manipulated state history, or self-related readout produces selective causal effects that cannot be reduced to memory format, replay, or extra compute.
