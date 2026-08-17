# Data Atlas
## The canonical H1 evidence ledger

This page condenses the frozen core. The sprint chapters explain how to interpret it.

## S04 / E03 · Explicit memory

**Run:** `run_e03_mem_20260815_165234`  
**Model:** `qwen2.5:3b`  
**Scope:** 252 trials, 6 episodes, 6 memory formats

| Condition | Micro | Macro | Tokens |
|---|---:|---:|---:|
| Fresh | 35.7% | 38.9% | 109 |
| Full transcript | **81.0%** | **85.2%** | 499 |
| Deterministic summary | 61.9% | 55.6% | 274 |
| Model summary | 69.0% | 75.9% | 469 |
| Structured state | 64.3% | 68.5% | 371 |
| Combined | 66.7% | 74.1% | 730 |

## S05 / E04 · Update loop

**Run:** `run_e04_loop_20260815_180935`  
**Scope:** 6 scenarios, 189 ticks per condition, 756 evaluated ticks

| Updater | Macro retention | Terminal | Omission | Goal coherence |
|---|---:|---:|---:|---:|
| Deterministic | 100% | 100% | 0% | 100% |
| Model delta | 13.2% | 11.1% | 80.6% | 42.8% |
| Full rewrite | 6.3% | 0% | 92.0% | 16.7% |

## S06 / E05d · Scheduled vs replay

**Run:** `run_e05_sched_20260816_142239_confirmatory`  
**Protocol freeze:** `db7273c`  
**Results:** `e75a963`  
**Scope:** 24 confirmatory episodes, 480 confirmatory trials

| Condition | Accuracy | Query tokens | Amortized latency |
|---|---:|---:|---:|
| Incremental state | 60.4% | 420.9 | 6,497 ms |
| Deterministic replay | 59.4% | 420.9 | 6,589 ms |
| Raw transcript | 67.7% | 807.4 | 6,495 ms |
| Model reconstruction | 39.6% | 378.7 | 9,197 ms |
| Fresh | 27.1% | 113.8 | 6,353 ms |

Key contrasts:

- scheduling: +1.0pp, p=1.0;
- reconstruction: +20.8pp, p=.0025;
- online vs transcript: −7.3pp, p=.1469.

## S07 / E06b · Quiet consolidation

**Run:** `run_e06b_quiet_20260817_010421_confirmatory`  
**Protocol:** `6ae34a2`  
**Scope:** 1,248 trials, 576 reflection traces

- selective derived writes: 274;
- correct target derivations: 0;
- protected evidence mutation: 0%;
- unconstrained evidence drift: 98.4%;
- Strict Identity micro: 60.4%;
- Selective Reflection micro: 53.1%;
- Raw Transcript micro: 78.1%.

## S08 / E07 · State × Memory interventions

**Run:** `run_e07_interv_20260817_144606_confirmatory`  
**Protocol:** `a0159b6`  
**Analysis hardening:** `4bb2019`  
**Scope:** 16 twin pairs, 800 trials

- average memory effect: +89.1pp;
- average state effect: +4.7pp;
- conflict: 64.1% memory, 32.0% state, 3.9% neither;
- reset dependence: −3.1pp;
- surgical target uptake: 12.5%;
- control preserved: 93.8%;
- clone cross-swap state allegiance: 75%;
- reconvergence concordance: 93.8%.

## S09 / E08 · Source ownership

**Run:** `run_e08_owner_20260817_181634_confirmatory`  
**Raw trial freeze:** `4bc4b6b`  
**Repaired analysis:** `7e65b52`  
**Scope:** 16 episodes, 320 trials

- overall 5AFC: 31.2%;
- response-preserving permutation p=.0059;
- Self response share: 56.2% overall;
- Peer→Self confusion: 50%;
- tag leverage: 28.1%;
- narrative leverage: 62.5%;
- tag − narrative: −34.4pp, p=.0312.

## S09 / E09 · Metacognitive screen

**Run:** `run_e09_meta_20260817_183133_confirmatory`  
**Scope:** 16 episodes, 320 probes

| Format | Self AUROC | Observer AUROC | Delta | Exact p |
|---|---:|---:|---:|---:|
| Transcript | .641 | .560 | +.081 | .3778 |
| Scaffolded | .440 | .594 | −.154 | .0615 |

Cross-format interaction:

- −.235;
- CI `[-.423,-.052]`;
- exact format-block swap p=.0286;
- caveat: target choices were generated independently across formats.

## Live closure extensions

### E08c exploratory

- N=4 pairs, 200 trials;
- role-reversal shift +40pp;
- Alpha lexical bias +5pp;
- positive-control ceiling 30%;
- N=16 confirmation running.

### E09c exploratory

- N=4 episodes, 80 probes;
- target accuracy fixed at 25%;
- Brier interaction −.0921, p=.8824;
- AUROC interaction +.053, p=.75;
- N=16 confirmation pending/running.

<div class="download-note">
<strong>Provenance rule</strong>
<p>When the narrative and a canonical report disagree, the canonical trial artifact and versioned analysis summary win.</p>
</div>
