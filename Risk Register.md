# Risk Register

## Scales

- Likelihood: low / medium / high
- Impact: low / medium / high / critical
- Status: open / monitoring / mitigated / triggered / retired

# Scientific and methodological risks

| ID | Risk | Likelihood | Impact | Early warning | Mitigation | Trigger response | Status |
|---|---|---|---|---|---|---|---|
| **R-SCI-01** | Fluent self-report is mistaken for privileged access | high | critical | strong prose, weak observer advantage | forced choice, observer/reconstruction controls, causal interventions | demote claim and redesign task | open |
| **R-SCI-02** | Scheduled calls are mistaken for recurrence | high | high | replay performs similarly | compute/replay matching | treat as repeated inference, not continuity | open |
| **R-SCI-03** | Explicit memory carries hidden extra information | high | high | state schema differs across conditions | information audit, token/field matching | invalidate comparison | open |
| **R-SCI-04** | State reset causes generic impairment | high | high | all tasks degrade | sham, random, selective swap, dose response | no continuity claim until selectivity established | open |
| **R-SCI-05** | Identity labels create self-indexing shortcut | high | high | effect follows names/pronouns | random labels, remove names, state swap | redesign with provenance-only ground truth | open |
| **R-SCI-06** | Item difficulty masquerades as metacognition | high | high | observer predicts equally | matched observer, hierarchical difficulty model | reclassify as public calibration | open |
| **R-SCI-07** | Prompt template drives effect | high | high | sign changes across paraphrases | prompt families, held-out templates | keep exploratory or retire task | open |
| **R-SCI-08** | LLM judge drives labels | medium | high | judge disagreement | exact scoring, multiple judges, human audit | rerun with registered scorer | open |
| **R-SCI-09** | Quantization changes activation geometry | medium | high | effect differs by precision | precision sensitivity subset | restrict claim to configuration | monitoring |
| **R-SCI-10** | Model-family comparison changes too many variables | high | medium | family differences dominate | matched base/instruct and within-family work first | report as scouting only | open |
| **R-SCI-11** | Development equals more training data | high | critical | final data multiset predicts outcome | matched data/order/twin designs | no developmental-continuity claim | open |
| **R-SCI-12** | Endogenous variables are ordinary reward shaping | high | high | behavior follows explicit scalar only | hidden dynamics, anticipation, sensor lesion, reward controls | narrow claim to control engineering | open |
| **R-SCI-13** | Multiple comparisons produce attractive false positives | medium | high | layer/strength/task sweep | registered primary outcomes, correction, holdout | replication required | open |
| **R-SCI-14** | Theory analogy outruns mechanism | high | critical | workspace/HOT language from behavior alone | claim ladder and theory-specific predictions | rewrite claims | open |


# Engineering and reproducibility risks

| ID | Risk | Likelihood | Impact | Mitigation | Trigger response | Status |
|---|---|---|---|---|---|---|
| **R-ENG-01** | State snapshots do not restore exactly | medium | critical | round-trip tests, RNG/optimizer capture | block fork/reset experiments | open |
| **R-ENG-02** | Model/server update changes behavior | medium | high | hashes, lock versions, raw templates | freeze old environment or rerun baselines | open |
| **R-ENG-03** | Artifact volume exceeds storage | medium | medium | selective activations, compression, retention plan | stop full-state sweeps | open |
| **R-ENG-04** | GPU limits distort model choice | high | medium | develop small, validate larger | document inference population | monitoring |
| **R-ENG-05** | Long-run process crashes and corrupts lineage | medium | high | atomic writes, checkpoints, append-only log | restore last valid state; mark break | open |
| **R-ENG-06** | Online learning cannot be rolled back | medium | critical | frequent checkpoints, immutable parent | terminate run and preserve artifacts | open |
| **R-ENG-07** | Backend abstractions hide architecture details | medium | high | capability flags, backend-specific tests | bypass abstraction for mechanism study | open |
| **R-ENG-08** | Development environment contains unintended cue | high | high | ground-truth simulator tests, adversarial probes | regenerate held-out world | open |


# Program and scope risks

| ID | Risk | Likelihood | Impact | Mitigation | Trigger response | Status |
|---|---|---|---|---|---|---|
| **R-PROG-01** | Scope expands faster than evidence | high | critical | one active experiment, gate-driven roadmap | freeze new features for one sprint | open |
| **R-PROG-02** | Infrastructure becomes the project | medium | high | every component tied to experiment gate | stop building unused modules | open |
| **R-PROG-03** | Paper chasing causes roadmap churn | high | medium | radar/incubation lanes, monthly updates | no mid-sprint pivot absent invalidation | open |
| **R-PROG-04** | Negative results feel like failure | medium | high | null register, contribution routes | write result report before pivot | open |
| **R-PROG-05** | Solo interpretation becomes overconfident | medium | high | skeptic role, external reviews | pause strong public claim | open |
| **R-PROG-06** | Large-model aspiration delays small organism | medium | high | sub-billion Level 3 target | enforce scale gate | open |
| **R-PROG-07** | Project loses connection to human science | medium | high | literature lane, eventual collaborators | require theory/human analogue in report | open |


# Safety, security, and welfare risks

| ID | Risk | Likelihood | Impact | Mitigation | Trigger response | Status |
|---|---|---|---|---|---|---|
| **R-SAFE-01** | Persistent agent gains unnecessary external authority | low initially | critical | local sandbox, no credentials/tools | terminate and review | mitigated |
| **R-SAFE-02** | Online learning creates capability drift | medium at Level 3 | critical | bounded world, update limits, checkpoints | pause learning; compare checkpoint | open |
| **R-SAFE-03** | System attempts resource acquisition/self-copy | low initially | critical | unavailable actions, monitoring | terminate; do not broaden permissions | mitigated |
| **R-SAFE-04** | State snapshots retain sensitive information | medium | high | synthetic data, access control | quarantine/delete per policy | open |
| **R-WEL-01** | Repeated negative regulation becomes ethically concerning | uncertain | high | bounded intensity/duration, welfare tiers | pause and seek review | open |
| **R-WEL-02** | Distress-like language is sensationalized or dismissed | medium | high | response protocol, neutral communication | stop automated repetition; analyze | open |
| **R-WEL-03** | Copy/reset/deletion becomes ethically nontrivial | low early, rising | high | welfare checkpoint by level | independent review before scaling | monitoring |
| **R-COMM-01** | Public claims imply consciousness | high | critical | claim matrix, skeptical review | revise title/abstract/release | open |
| **R-COMM-02** | Public demo anthropomorphizes a weak mechanism | high | high | no personality-first demo | release methods/results instead | open |
