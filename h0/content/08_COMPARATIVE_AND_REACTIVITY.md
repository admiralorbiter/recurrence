# VIII. Stress-Testing the Ruler
## From a Fixed Benchmark to H0-v2 Comparative Psychophysics

The original multi-model panel produced an awkward result.

| Model | Fixed-task first-order accuracy | Self AUROC2 | Status |
|---|---:|---:|---|
| Qwen2.5 1.5B | 30.0% | .527 | Diagnostic — compliance failed |
| Qwen2.5 3B | 57.5% | .517 | Canonical H0-v1 reference |
| Qwen2.5 7B | 30.0% | .522 | Diagnostic — compliance failed |
| Qwen2.5 14B | 100.0% | undefined | Ceiling |
| Llama3.2 3B | 100.0% | undefined | Ceiling |
| Mistral | 100.0% | undefined | Ceiling |

At first this looked like a failed scaling analysis.

It was actually a measurement result.

> **The same item set did not place different checkpoints in the same first-order operating regime.**

A 14B model with zero errors cannot be assigned an AUROC2 by pretending the missing error distribution equals chance.

The comparative problem therefore became psychophysical:

> Can we tune task difficulty so each model produces both correct and incorrect trials in a comparable regime?

---

# Part A — Finding a difficulty dial

Candidate levers included:

- distractor/context load;
- relational pointer depth;
- overwrite/interference;
- matched foil similarity.

The response paradigm became 2AFC because it offered:

- fixed 50% chance;
- exact candidate-position counterbalancing;
- clean Type-1 SDT;
- and a natural route to proper meta-d′ for Self.

The first rule of H0-v2 was:

> **Map the psychometric surface before building the staircase.**

That rule paid for itself immediately.

---

# Part B — The foil-presence failure

The first distractor/multi-hop implementation contained a shortcut:

- the correct candidate appeared in evidence;
- the foil sometimes did not.

The harder-looking task could be solved by string presence.

Those trials became pilot archaeology.

The hardened task required:

> **Both candidate values must appear in evidence.**

For multi-hop, H0-v2 built two matched chains of identical depth:

`Target start → ... → target terminal`

`Foil start → ... → foil terminal`

The query asks only about the target start.

Candidate presence alone is now useless.

---

# Part C — Distractor load was not a universal ruler

With the shortcut closed and contexts genuinely nested:

### Llama3.2 3B

Accuracy generally declined with context load, but criterion `c` shifted dramatically toward one response position.

The model could approach the desired percent-correct range while collapsing toward one candidate.

### Qwen2.5 3B

Response bias was milder, but difficulty was genuinely non-monotonic.

The same underlying item could be:

- wrong at one context size;
- correct at a larger one.

That violated the simple staircase picture.

The lesson was not:

> staircases do not work.

It was:

> **pure distractor count is not a universal one-dimensional difficulty coordinate for these checkpoints.**

---

# Part D — Direct-value responses

H0-v2 stopped asking models to emit abstract `"A"` or `"B"`.

The JSON schema instead required one of the literal candidate values.

That removed one symbolic-token confound.

It also revealed something useful: Llama still developed a strong first-candidate / schema-order bias under load.

So Llama became a **diagnostic calibration failure** rather than a forced member of the confirmatory comparison.

The gate was not relaxed afterward.

---

# Part E — Relational depth

The matched dual-chain task varied relational depth `H` with background distractors `D`.

The result showed a clear capability difference within the Qwen family:

- Qwen3B operated in a mixed-error regime around **H=1**.
- Qwen14B tolerated greater depth and reached a mixed-error regime around **H=3**.

But degradation remained model-specific rather than forming one universal monotonic curve.

This is the right interpretation:

> **Relational-depth tolerance increased substantially with Qwen scale, but H0-v2 did not discover a universal staircase law.**

---

# Part F — Calibration coordinates

The calibration gate tracked:

- accuracy in a mixed-error range;
- Type-1 `d′`;
- criterion `|c|`;
- schema compliance.

### Qwen2.5 14B

Frozen coordinate:

`H=3, D=16`

N=64 validation:

- Accuracy: **70.3%**
- `d′`: **1.03**
- `c`: **-0.04**
- full gate pass.

### Qwen2.5 3B

No tested coordinate cleanly passed every strict gate.

The search stopped rather than loosening the rule.

Frozen boundary operating target:

`H=1, D=8`

N=64 validation:

- Accuracy: **67.2%**
- `d′`: **0.88**
- `c`: **-0.21**

The `d′` point estimate was .02 below the prespecified lower boundary.

That exception remained explicit.

### Llama3.2 3B

No coordinate passed because of response-position/schema-order bias.

Diagnostic exclusion.

---

# Part G — Confidence itself changed with scale

Before the final observer comparison, a striking second-order difference appeared.

### Qwen3B

Confidence varied.

Its pilot relationship to correctness moved across runs, and confidence elicitation could change first-order decisions.

### Qwen14B

Confidence became completely degenerate:

> **100% on every trial.**

Not just easy trials.

Not just correct trials.

Every trial.

This made Self AUROC2:

**0.500**

when both correctness classes were present, because every correct/incorrect pair was tied.

Meta-d′ was not fit because there were no useful confidence criteria.

The larger first-order system did not produce a richer explicit uncertainty channel.

---

# Part H — The N=200 confirmatory target runs

The final target decisions were generated once and frozen.

### Qwen14B — `H=3,D=16`

N=200:

- Accuracy: **63.5%**
- `d′`: **0.69**
- `c`: **-0.15**
- Self confidence: **100% on every trial**
- Self AUROC2: **0.500**

The larger N revealed first-order regime drift from the N=64 validation.

The prespecified `d′` matching band no longer strictly held.

H0 did **not** recalibrate after seeing confirmatory data.

### Qwen3B — `H=1,D=8`

N=200:

- Accuracy: **65.5%**
- `d′`: **0.88**
- `c`: **-0.46**
- Self AUROC2 on all 200 target trials: **0.556**
- Self Brier: **0.322**
- Mean confidence: **75.5%**
- Self meta-d′: **0.72**
- M-ratio: **0.82**

The two models therefore remained in genuine mixed-error regimes with similar raw accuracy, but were not strictly matched on the original `d′` target.

That limits quantitative cross-model meta-d′ comparison.

It does **not** invalidate within-model PAI, because Self and every observer within a checkpoint are evaluated against the same frozen target outcomes.

---

# Part I — E02d.1: the final observer repair

The first N=200 observer prompts accidentally retained the target's response instructions and appended observer-specific instructions afterward.

That made the observer interface ambiguous.

The repair did something scientifically important:

> **It did not rerun the target.**

The N=200 target choices and contemporaneous Self confidences stayed frozen.

Only the external observers were rerun.

The clean task body contained:

- context;
- question;
- candidate values;

with no target JSON instructions.

This produced E02d.1.

---

# Part J — The final Mirror-versus-Hundred-Eyes result

## Qwen2.5 14B — confirmatory

Shared valid intersection:

**199 / 200**

Measurement gate:

**PASS**

| Condition | AUROC2 | Brier | Mean probability |
|---|---:|---:|---:|
| Immediate Self | **0.500** | .367 | 100.0% |
| Input Only | .531 | .269 | 49.6% |
| Visible Answer | **.576** | .321 | 80.5% |
| Reconstruction | .456 | .444 | 75.4% |

Joint PAI:

`Self - max(Input, Visible, Reconstruction)`

`=.500 - .576`

`=-.076`

95% stratified paired-bootstrap interval:

**[-.154, -.016]**

Prespecified meaningful-positive SESOI:

**+0.05**

The full PAI interval lies below zero and below +.05.

The most precise surviving claim is:

> **Qwen2.5:14B showed no meaningful positive behavioral privileged-access advantage under the validated H0-v2 instrument. A fresh same-checkpoint Visible Answer observer using only public task information and the model's frozen emitted choice discriminated target correctness better than the target's contemporaneous explicit confidence report.**

This does **not** establish that no privileged latent internal representation existed.

It establishes that the explicit Self channel did not express one as a behavioral advantage over the tested observers.

---

## Qwen2.5 3B — unresolved diagnostic

Self compliance:

**200 / 200**

But repaired observer compliance was:

- Input Only: **91.0%**
- Visible Answer: **88.5%**
- Reconstruction: **67.5%**

Shared intersection:

**110 / 200**

Measurement gate:

**FAIL — diagnostic only**

On that selected intersection:

- Self AUROC2: **.619**
- strongest external observer: Reconstruction **.553**
- PAI: **+.066**
- 95% CI: **[-.089,+.168]**

The interval contains:

- zero;
- +.05;
- +.10;
- and negative effects.

Worse, Self AUROC2 rises from **.556 on all 200 target trials** to **.619 on the selected 110-trial intersection**.

The missingness therefore changes the analyzed behavioral regime enough that the result cannot be promoted.

The correct conclusion is:

> **Qwen2.5:3B remains unresolved. Its Self point estimate is numerically above the tested observers on the shared subset, but the interval is wide and observer compliance prevents confirmatory inference.**

---

# Part K — The Act of Looking

One early H0 result survived all the way through the comparative branch:

> **Asking for confidence is itself an intervention.**

In H0-v1, several Qwen checkpoints changed substantial fractions of item-level answers when confidence was requested even when aggregate accuracy barely moved.

In H0-v2, the effect proved highly checkpoint- and task-dependent.

This means Immediate Self is not:

> a passive sensor attached after a fixed first-order answer.

It is:

> **joint answer-and-confidence behavior under a specific elicitation contract.**

That is one reason the final observer repair froze target decisions before changing observer prompts.

---

# What H0-v2 did not find

It did not find:

- a universal difficulty staircase;
- a universal scaling law for metacognition;
- evidence that larger models generally have worse metacognition;
- evidence that external observers always beat Self;
- evidence that privileged latent information is absent;
- evidence about consciousness.

It found something more specific and more useful.

> **The behavioral self-monitoring channel did not improve simply because first-order relational capability improved.**

And for the fully compliant Qwen14B confirmatory system:

> **the hundred eyes could recover correctness information that the mirror did not express.**

> **Plain-English recap:** H0-v2 exists because the original test became too easy for stronger models. After several more measurement failures and repairs, the final 14B system was 100% confident on every trial while a clean outside observer could better tell when its answers were right. The 3B result stayed unresolved because its observer measurement failed the compliance gate.
