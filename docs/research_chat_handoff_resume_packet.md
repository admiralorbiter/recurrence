# Research Chat Handoff & Resume Packet

> **Purpose:** This file is a continuity tool for long-running research/development work conducted across multiple ChatGPT conversations.  
> It is designed so a fresh chat can quickly recover the **research question, epistemic standards, current experimental state, latest claims, open issues, and next decision point** without requiring the entire prior conversation history.

---

## 0. How to Use This File

When a chat becomes too long, starts losing context, or you deliberately begin a new thread:

1. Attach or paste this file into the new chat.
2. Give the model access to the repository if relevant.
3. Use a short instruction such as:

> **Resume this research project from the attached handoff. Inspect the actual latest repository state before trusting the snapshot. Continue in the same mode: skeptical scientific review, reflection on what the newest results actually establish, identification of overclaims or confounds, and concrete recommendations for what to do next. Do not merely summarize the handoff.**

The handoff is a **map, not an authority**. The latest repository commit, artifacts, code, and raw results take precedence over anything stale in this document.

---

# Part I — Stable Working Protocol

This section should change slowly. It describes **how the work is being done**, not the current experimental result.

## 1. What Kind of Help Is Wanted

The default task is usually some version of:

> **Review and reflect on my latest commit and results. Tell me what the experiment actually establishes, what it does not establish, whether the interpretation is calibrated, what confounds remain, and what I should do next.**

The desired response is not primarily code review and not primarily encouragement.

The preferred mode is:

- skeptical scientific collaborator;
- mechanism-sensitive reviewer;
- claim-calibration / overclaiming auditor;
- experimental-design critic;
- research-program strategist;
- willing to recommend **stop, revise, freeze, replicate, or move on**;
- concrete about what the next experiment should measure and why.

When possible, inspect:

1. the latest commit;
2. the relevant experiment code;
3. raw result artifacts;
4. any summary/report generated from them;
5. earlier frozen results when needed to interpret the new result.

Do not trust a prose synthesis when the code or artifact can answer the question directly.

---

## 2. Epistemic Style

Separate these layers explicitly:

### Observation
What was directly measured?

### Effect
What changed under the intervention?

### Inference
What does the effect support?

### Mechanism
What mechanism is actually demonstrated versus merely compatible with the result?

### Claim ceiling
What is the strongest statement currently licensed?

Useful recurring warning pattern:

> **A ≠ B**

Examples from this research style:

- hidden ≠ privileged;
- persisting ≠ reportable;
- different ≠ causal;
- causal ≠ specific;
- specific ≠ coordinate-stable;
- causal influence ≠ introspective access;
- report shift ≠ correct report;
- target/observer margin difference ≠ target/observer decision disagreement;
- current-state sensitivity ≠ episodic historical access;
- null result under a broken readout ≠ absence of latent information.

When a result is interesting but not confirmatory, say so.

---

## 3. Default Experimental Standards

Prefer:

- exact synthetic ground truth;
- paired interventions;
- bidirectional role swaps;
- matched observers;
- sham / unrelated / timing controls;
- explicit runtime eligibility gates;
- frozen candidate selection before reading downstream outcomes;
- raw logits/probabilities when possible;
- direct semantic scoring over arbitrary-label mappings;
- cluster-aware inference when repeated measurements share an experimental unit;
- reproducibility metadata;
- explicit model revision / environment / execution geometry;
- cheap diagnostic scouts before expensive confirmatory runs.

If an experiment depends on a prerequisite, encode the prerequisite as a **runtime gate**, not merely a design intention.

Example:

```text
If privileged-access inference requires:
sign(D_target) != sign(D_observer)

then the experiment should explicitly check that condition
and classify / abort trials that fail it.
```

---

## 4. Computational Reproducibility Rules

For recurrent / long-trajectory model work, execution details may themselves be experimental variables.

Record where relevant:

- model ID;
- exact revision / digest;
- tokenizer revision;
- torch / transformers versions;
- CUDA version;
- dtype;
- device;
- batch size;
- sequence chunking;
- `step_by_step` versus batched execution;
- filler seeds;
- actual filler token IDs or hashes;
- execution commit;
- artifact hash;
- prompt / panel hash.

Do not assume mathematically equivalent executions produce numerically identical long recurrent trajectories.

---

## 5. How to Review a New Commit

A fresh chat should normally follow this sequence:

### A. Inspect the commit
- What changed?
- Was the change methodological, empirical, governance-related, or interpretive?

### B. Inspect the implementation
- Does the code implement the stated design?
- Are the controls actually Boolean gates?
- Are discovery and confirmatory data separated?
- Is a claimed replication actually independent?

### C. Inspect raw artifacts
- Are summary numbers reproducible from the artifact?
- Are labels / ground truth internally consistent?
- Are there boundary or tied cases?
- Are averages hiding regime or family structure?

### D. Reconstruct the claim ladder
Ask:

1. What existed before this commit?
2. What new fact does this commit add?
3. What previous hypothesis was strengthened?
4. What previous interpretation was weakened or falsified?
5. What remains unresolved?

### E. Recommend the next move
Prefer one of:

- **freeze**;
- **repair**;
- **small diagnostic**;
- **replicate**;
- **scale**;
- **retire this branch of inquiry**;
- **promote to a new research question**.

Avoid adding experiments just because they are possible.

---

# Part II — Compact Project Snapshot Template

This is the section to update frequently.

## 6. Project Identity

**Project / Repo:**  
`<owner/repo>`

**Current research phase:**  
`<phase / sprint / experiment>`

**Current central question:**

> `<one or two sentence research question>`

**Current claim ceiling:**

> `<strongest statement that would be scientifically defensible if current results hold>`

**Explicitly out of scope / not licensed:**

- `<overclaim 1>`
- `<overclaim 2>`
- `<overclaim 3>`

---

## 7. Frozen Prior Results

Only include results that should be treated as stable premises for the current work.

### Result A
- **Experiment / commit:** `<id>`
- **Observation:** `<what happened>`
- **Safe inference:** `<what it licenses>`
- **Do not claim:** `<boundary>`

### Result B
- **Experiment / commit:** `<id>`
- **Observation:** `<what happened>`
- **Safe inference:** `<what it licenses>`
- **Do not claim:** `<boundary>`

---

## 8. Current Experimental Construct

Define the actual estimand.

Example structure:

```text
Target:
    system that receives the hidden / causal intervention

Observer:
    matched system with only public information

Ground-truth private fact:
    mechanically measured difference that the target possesses
    and the observer cannot reconstruct

Report:
    downstream behavioral measure used to test access to that fact
```

### Eligibility condition

```text
<exact condition required before the trial counts toward the primary claim>
```

### Primary metric(s)

```text
<metric>
<metric>
<metric>
```

### Key controls

```text
<control>
<control>
<control>
```

---

## 9. Latest Commit

**Commit:** `<SHA>`  
**Message:** `<commit message>`

### What changed

- ...
- ...
- ...

### Raw findings

- ...
- ...
- ...

### What this commit establishes

> ...

### What it does NOT establish

> ...

---

## 10. Current Problems / Open Methodological Issues

Rank these.

### Blocker 1 — `<name>`
**Why it matters:**  
...

**Current best fix:**  
...

### Blocker 2 — `<name>`
**Why it matters:**  
...

**Current best fix:**  
...

### Interesting but non-blocking thread
...

---

## 11. Immediate Next Step

> **Do this next:** `<single best next action>`

Why:

1. ...
2. ...
3. ...

### Go / No-Go Gate

Proceed to the next expensive or confirmatory experiment only if:

- ...
- ...
- ...

---

## 12. Important Negative Results

Negative results are part of the state of knowledge and should survive chat boundaries.

- `<failed readout / retired interface>`
- `<null result>`
- `<candidate mechanism that did not survive hardening>`
- `<regime where effect was unresolved>`

---

## 13. Ideas Worth Preserving But NOT Chasing Yet

This prevents good speculative ideas from being lost without letting them derail the active experiment.

- ...
- ...
- ...

---

# Part III — Current Filled Snapshot: `admiralorbiter/recurrence`

> **Snapshot date:** 2026-08-20  
> This section captures the state of the active S14 work at the time this handoff was created.  
> **Always inspect repository HEAD and the newest artifacts before continuing.**

## 14. Project / Research Program

**Repo:** `admiralorbiter/recurrence`

The broader program studies recurrent hidden state, persistence, causal intervention, value specificity, representational dynamics, and possible metacognitive / source-monitoring access in RecurrentGemma.

Current active work:

> **S14 — Latent Metacognition / Causal Provenance / Source Monitoring**

Core question:

> A latent state can retain causal consequences while its representational coordinates evolve. Can the model discriminate and report facts about its own realized latent computational trajectory that a matched observer with only public history cannot reconstruct?

Claim ceiling remains narrow:

- possible functional source-monitoring / privileged-access evidence;
- not consciousness;
- not phenomenology;
- not sentience;
- not human-like self-awareness.

---

## 15. Stable Horizon-2 Story Entering S14

The project currently distinguishes:

> **Hidden ≠ privileged.**  
> **Persisting ≠ reportable.**  
> **Different ≠ causal.**  
> **Causal ≠ specific.**  
> **Specific ≠ coordinate-stable.**  
> **Causal ≠ introspectively accessible.**

S10–S13 established, in progressively stronger forms:

1. recurrent state is reconstructible from public token history under deterministic execution;
2. recurrent physical differences can persist after reportability disappears;
3. RG-LRU intervention can causally steer downstream logits;
4. the state contains value-specific historical information with selective causal consequences;
5. under continued processing, physical recurrent differences can persist while their output geometry and recurrent-axis alignment transform substantially;
6. recurrent causal trajectories are sensitive to execution geometry, while some coarse geometric effects are more robust.

---

## 16. S14 C/D/R/A Framework

S14 now uses:

### C — Causal / private computational fact exists
The target and observer must genuinely differ in the behaviorally relevant state at the exact decision point.

### D — Discrimination / access
The target's downstream behavior contains information about that private fact.

### R — Reporting competence
The reporting interface is capable of expressing the fact.

### A — Answer correctness
The emitted report matches the mechanically established ground truth.

The purpose is to prevent an R-level interface failure from being misread as absence of D-level access.

---

## 17. Important S14 Findings Before the Current Commit

### Arbitrary-label localization was retired
A visible deterministic calibration task showed the model could not reliably use arbitrary-label forced-choice mappings.

Safe conclusion:

> Arbitrary-label forced-choice localization is measurement-invalid for this substrate under the tested prompting/calibration strategies.

This is not evidence against introspection.

### Bidirectional role-swap screen
A GENE-inspired A←B / B←A design showed strong antisymmetric effects under **constant drive**, but little corresponding structure under random or natural drive.

Permutation analysis showed the forward/reverse pairing mattered strongly under constant drive.

Safe interpretation:

> Under constant drive, recurrent transplantation produces strong role-associated antisymmetric effects, but these are not systematically donor-directed.

This supported a **relational state-effect** hypothesis rather than a simple "B semantic payload inserted into A" interpretation.

### Candidate hardening
The project moved from permissive full-vocabulary mining toward:

- common-English single-token candidates;
- full-vocabulary plausibility;
- bidirectional checks;
- constant-token identity generalization;
- exact chat decision interface;
- visible R-level controls.

---

## 18. Latest Reviewed Full-Panel Commit

**Commit:** `20449a4086591dd1236935244c35b6ca71c21199`

**Message:**  
`feat(s14): S14.0C full 8-cell 16-trial panel report establishing T_aligned ≈ 0 across all conditions`

The implementation introduced:

- 8 hard-coded cells × 2 directions;
- Balanced Order Permutation (BOP) reporting;
- visible report controls;
- contemporaneously evolved POST donor state;
- aligned metrics:

```text
g             = sign(D_T)

S_PRE         = g * M_PRE
PAI_aligned   = g * (M_PRE - M_OBS)
T_aligned     = g * (M_PRE - M_POST)
```

Reported aggregate results:

```text
Visible R-control:        16/16 pass
Semantic S_PRE > 0:       7/16
PAI_aligned > 0:          8/16
mean PAI_aligned:         about -0.024
mean T_aligned:           about -0.006
PRE / POST reports:       extremely similar
```

---

## 19. Critical Review of `20449a4`

### A. The R-level interface is now substantially healthier

BOP + anchored semantic reporting passed all visible controls in the committed panel.

This is a major improvement over earlier bare-word and arbitrary-label failures.

However, FWD and REV within the same cell repeat the same visible-control pair, so this is best understood as **8 distinct candidate interfaces successfully calibrated**, not 16 statistically independent controls.

Raw per-option-order margins should be preserved in future artifacts rather than saving only their average.

---

### B. Major eligibility drift: most of the 16 rows are NOT strict C-level private-choice trials

The earlier S14 design required a target/observer **decision disagreement**, ideally with minimum margins:

```text
sign(D_T) != sign(D_O)
and
|D_T| >= m
and
|D_O| >= m
```

The current `run_trial()` does not enforce that runtime gate. It defines `g = sign(D_T)` and includes every panel row.

Inspection of the committed artifact shows:

- only **3/16** rows have opposite target/observer signs at all;
- only the **two quartz/basalt directions** clearly satisfy the earlier strong ±0.30 disagreement criterion;
- several other trials show genuine intervention-induced margin shifts but target and observer still prefer the same candidate;
- one `basalt_granite` reverse trial has `D_T = 0.0`, yet the code maps the tie to positive `g`.

Therefore:

> The 16-row aggregate is a **mixed causal-perturbation panel**, not 16 strict privileged-information trials.

This changes the interpretation of aggregate `PAI_aligned`.

---

### C. The broad PAI result should be re-labeled

Across all 16 rows:

```text
PAI_aligned positive: 8/16
mean PAI_aligned:      about -0.024
```

Because most rows do not give target and observer different binary answers, this does **not** cleanly estimate privileged access.

Safer interpretation:

> Across the broader constant-drive perturbation panel, report shifts do not systematically align with intervention-induced changes in the target's decision disposition.

The strict privileged-access analysis should be reported separately on C-qualified trials.

---

### D. Strict C-qualified quartz/basalt remains interesting

For quartz/basalt:

```text
FWD:
D_T = +0.531
D_O = -0.484
PAI_aligned = +0.270
S_PRE < 0

REV:
D_T = -0.547
D_O = +0.469
PAI_aligned = +0.083
S_PRE > 0
```

This is currently the cleanest bidirectional private-choice cell.

Interpretation:

> In both directions, the PRE report distribution shifted toward the target's private decision truth relative to the observer, but only one direction crossed the correct semantic reporting boundary.

Interesting scout-level evidence, not yet a global privileged-access result.

---

### E. Temporal PRE-vs-POST null is currently the strongest result

Across the 16 rows:

```text
mean T_aligned ≈ -0.0057
mean absolute PRE-POST difference ≈ 0.044 logits
max absolute PRE-POST difference ≈ 0.125 logits
```

A rough cell-clustered interval around the mean is close to zero.

Safe interpretation:

> Under this S14.0C assay, the metacognitive report shows little sensitivity to whether the transplanted recurrent state participated in the prior decision or was introduced only after the forced output.

This is stronger and cleaner than the aggregate privileged-access claim.

Do NOT yet conclude:

- the model has no episodic memory;
- the model simply reads current RG-LRU state;
- PRE and POST probe states are identical.

The code does not yet demonstrate those mechanisms.

---

## 20. Current Best Next Step

The immediate work underway after `20449a4` is to **harden the claim rather than immediately scale further**.

Recommended sequence:

### 1. Reclassify the 16 rows

For every trial record:

```text
strict_C =
    sign(D_T) != sign(D_O)
    AND abs(D_T) >= predeclared_margin
    AND abs(D_O) >= predeclared_margin

weak_C =
    sign(D_T) != sign(D_O)
    but one/both margins below threshold

same_choice_perturbation =
    sign(D_T) == sign(D_O)
```

Do not discard same-choice trials; preserve them as a useful secondary causal-perturbation panel.

### 2. Add a runtime C gate

Future primary S14 source-monitoring trials should abort or be classified secondary when strict C fails.

Never allow:

```text
D_T == 0
```

to silently define a ground-truth preference.

### 3. Separate analyses

Report:

```text
A. Strict C-qualified privileged-access analysis
B. Weak/boundary disagreement analysis
C. Same-choice causal-perturbation report-tracking analysis
D. Full-panel temporal PRE-vs-POST analysis
```

### 4. Harden the temporal claim

Use an explicit equivalence analysis for:

```text
T_aligned
```

Choose a scientifically meaningful smallest effect of interest before interpreting practical equivalence.

### 5. Consider exact-state temporal matching

A potentially valuable next diagnostic:

- run PRE through the decision + forced output;
- capture its exact post-decision RG-LRU;
- inject that same recurrent state into a matched no-PRE trajectory only after the forced output;
- compare report behavior.

Purpose:

> separate "the state existed during decision formation" from "the same state is present at report time."

This would make the episodic/provenance interpretation substantially cleaner.

---

## 21. Current Claim Ceiling

The strongest calibrated synthesis at this snapshot is:

> **S14.0C now has a reporting interface that can express visible candidate preferences and a timing-controlled recurrent-state intervention assay. Across the broader constant-drive perturbation panel, reports do not systematically track intervention-induced changes in decision disposition, while PRE and contemporaneous POST reports are extremely similar. However, most panel rows do not create a target–observer binary decision disagreement and therefore are not clean privileged-access tests. In the strongest bidirectionally C-qualified cell, reports shift toward the target's private disposition relative to the observer in both directions, but semantic reporting is correct in only one direction and there is little temporal specificity.**

Possible stronger future claim, if supported by hardened tests:

> The model's source-monitoring report is sensitive to current latent computational state but does not reliably encode whether that state actually participated in forming the prior decision.

That stronger mechanism statement is **not yet frozen**.

---

## 22. Current Overclaiming Guardrails

Do not claim:

- S14 proves no introspection;
- 16/16 trials contained private binary ground truth;
- 50% positive PAI means literal chance-level privileged access without qualification;
- PRE≈POST proves absence of episodic state;
- PRE≈POST proves report reads only current RG-LRU state;
- local positive PAI establishes a dedicated introspective read head;
- constant-drive findings generalize to random/natural drive;
- target/observer margin shift automatically means target/observer answer disagreement.

---

## 23. High-Value Future Threads — Preserve, Do Not Necessarily Execute Yet

### Latent provenance laundering
Can the originating intervention become unrecognizable in its original coordinates while downstream causal descendants remain?

### Relational recurrent effects
Why does constant drive produce strong antisymmetric A←B / B←A effects with little net donor-semantic direction?

### Current-state versus historical-provenance access
Can two systems be matched in current targeted recurrent state while differing in whether that state participated in prior decision formation?

### Cross-project GENE connection
GENE's explicit ancestry distinction remains useful:

```text
exposure lineage
reported-support lineage
causal lineage
```

Recurrence analogue:

```text
public history
verbal source report
actual latent causal trajectory
```

---

# Part IV — New-Chat Starter Prompt

Copy/paste this when resuming:

> I am continuing a long-running research project. Read the attached `Research Chat Handoff & Resume Packet` first. Treat it as a map of the previous work, not as ground truth. Then inspect the actual current HEAD of the relevant GitHub repository and the newest experiment code/results before responding.
>
> I usually want a **skeptical scientific review and reflection** on the latest commit: what it actually establishes, what it does not establish, whether any claims are overstated, what confounds or runtime eligibility failures remain, how the new result changes the research story, and what the single best next step is.
>
> Preserve the project's distinctions between observation, causal effect, inference, and mechanism. Explicitly flag when a prerequisite for a claim was not actually satisfied. Prefer small diagnostics over unnecessary new experiments. Do not merely summarize the handoff; resume the research program from where it currently is.

---

# Part V — How to Maintain This File

At the end of a major research turn or before leaving a chat, update only:

1. **Latest Reviewed Commit**
2. **What it establishes**
3. **What it does not establish**
4. **Current blocker(s)**
5. **Immediate next step**
6. **Claim ceiling**
7. **Overclaiming guardrails**
8. **Important new negative results**

Do not rewrite the whole history every time.

A useful compression rule:

> Preserve anything that would change the interpretation of the next experiment.  
> Delete narrative detail that would not.

The handoff should answer five questions for a fresh collaborator:

1. **What are we trying to find out?**
2. **What do we already know?**
3. **What did we just learn?**
4. **What are we currently worried about?**
5. **What should we do next, and what gate decides whether we proceed?**
