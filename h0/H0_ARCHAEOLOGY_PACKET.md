# H0 Archaeology Packet — v3
## How the Recurrence Project Learned to Distrust Its Own Results

> **Warning:** This packet intentionally preserves superseded analyses, failed runs, premature interpretations, task shortcuts, statistical mistakes, and interface bugs. Historical claims lose authority when later hardened evidence contradicts them.

H0 has two histories:

1. the history of the models;
2. the history of the measurement instrument.

The second is the point of this packet.

## Archaeology timeline

| Stage | What looked plausible | What broke | Durable lesson |
|---|---|---|---|
| S01 | Opaque failure vs semantic success reveals mechanism | Too many confounds | Plausibility is not mechanistic evidence |
| S02 | Free-generation failure means missing association | Forced choice is much stronger | Recognition ≠ reproduction |
| Context scout | 4/5 suggests state tracking | Hardened version falls to 3/20 | Easy prompt recency can imitate state |
| Early observers | Confidence can be compared directly | Probability semantics reversed | All observers must estimate the same event |
| Early PAI | Separate AUROCs can be subtracted | Different valid subsets | Pair on shared items |
| Early Brier | Hard classifications can stand in for probabilities | Calibration meaning disappears | Use real continuous forecasts |
| Reconstruction v1 | `1-p` recovers target probability | 4AFC residual mass is not binary | Require full multiclass distribution |
| Parser rescue | Missing fields can be repaired | Experimenter manufactures data | Invalid stays invalid |
| Metadata fallback | Recover target option from metadata | Ground truth leaks | Metadata is an experimental side channel |
| `run_004` | Inferential analysis still possible | Min compliance 37.5% | Sometimes the result is measurement failure |
| `run_005` | First trustworthy fixed-task result | — | Promote only after validity gates pass |
| Multi-model panel | Same task fairly compares scale | Floors and ceilings | Match operating regimes, not only items |
| H0-v2 initial 2AFC | Distractor load is clean | Correct candidate uniquely appears | Hard tasks can have easy shortcuts |
| “Nested” sweep | Only D changes | order/position also changed | Freeze nuisance dimensions |
| Universal staircase | D is one shared dial | Qwen non-monotonic; Llama biased | Difficulty can be multidimensional |
| Symbolic A/B | Position counterbalancing solves bias | response labels themselves matter | Direct-value response removes one token confound |
| Direct value | Bias fixed | Llama first-candidate/schema-order collapse | Surface repair can expose deeper policy bias |
| Accuracy matching | 70% means matched regime | criterion collapse | Match sensitivity and bias too |
| Type-2 helper | pseudo-meta-d′ convenient | formula not Maniscalco–Lau | Scientific names require scientific estimators |
| Confidence bins | 50–100 enough | legal <50 responses uncovered | Freeze full-domain preprocessing before data |
| E02d observers | observer prompts are detached | target instructions remained embedded | Observer interfaces are part of the construct |
| E02d.1 3B | shared subset yields PAI estimate | Reconstruction compliance 67.5% | selected missingness can change the behavioral regime |
| E02d.1 14B | Self may retain hidden advantage | Visible Answer beats Self; PAI CI negative | public behavioral traces can exceed explicit Self report |

## Canonical endpoints

### H0-v1

`run_e02_obs_005`

- Qwen2.5:3B
- 40 counterbalanced 4AFC items
- first-order accuracy: 57.5%
- Self AUROC2: .517
- Visible Answer: .678
- Reconstruction: .573
- Input Only: .527
- PAI: -.161
- 95% CI: [-.428,+.055]

Surviving claim:

> No positive privileged Self-monitoring advantage was resolved on this model/task; the joint result excluded a +.10 meaningful Self advantage.

### H0-v2 confirmatory endpoint

E02d.1 — frozen-target repaired observer battery.

#### Qwen2.5:14B

- target N=200
- shared valid N=199
- Self AUROC2: .500
- Visible Answer: .576
- PAI: -.076
- 95% CI: [-.154,-.016]
- measurement gate: PASS

Surviving claim:

> A meaningful positive **behavioral** privileged-access advantage was excluded under this instrument. The explicit Self channel was less discriminative than a clean Visible Answer observer.

#### Qwen2.5:3B

- target N=200
- shared valid N=110
- full-target Self AUROC2: .556
- shared-subset Self AUROC2: .619
- shared PAI: +.066
- 95% CI: [-.089,+.168]
- Reconstruction compliance: 67.5%
- measurement gate: FAIL

Surviving claim:

> Unresolved / diagnostic.

## Museum rules for future horizons

1. **Never average superseded and canonical runs.**
2. **A failed gate changes epistemic status, not merely presentation.**
3. **Do not repair missing confirmatory outputs after seeing them.**
4. **Keep target behavior frozen when repairing observer-only interfaces.**
5. **Separate target performance from shared-intersection performance.**
6. **Do not call AUROC2 calibration.**
7. **Do not use meta-d′ for an evaluator that has no matched first-order response distribution.**
8. **Do not claim latent absence from behavioral non-advantage.**
9. **Do not retrofit a new threshold after confirmatory results arrive.**
10. **Preserve the embarrassing version. It may teach the next horizon more than the polished one.**

> The scientific question is not whether an interpretation is interesting.
>
> It is whether the interpretation survives an instrument designed to prove it wrong.
