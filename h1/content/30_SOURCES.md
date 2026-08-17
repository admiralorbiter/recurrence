# Sources & Provenance
## Project source hierarchy

When this site disagrees with a project artifact, use this order:

1. canonical `trials.jsonl` / raw result artifact;
2. versioned machine-readable `summary.json`;
3. canonical experiment report;
4. analysis code and protocol freeze;
5. this teaching site;
6. roadmap or speculative prose.

## S04 / E03

- Report: `docs/E03_Explicit_Memory_Report.md`
- Run: `run_e03_mem_20260815_165234`
- Model: `qwen2.5:3b`
- Scope: 252 trials

## S05 / E04

- Report: `docs/E04_Update_Loop_Report.md`
- Run: `run_e04_loop_20260815_180935`
- Scope: 756 evaluated ticks

## S06 / E05d

- Report: `docs/E05_Scheduled_vs_Replay_Report.md`
- Protocol freeze: `db7273c`
- Results commit: `e75a963`
- Run: `run_e05_sched_20260816_142239_confirmatory`

## S07 / E06b

- Report: `docs/E06_Quiet_Intervals_Report.md`
- Protocol freeze: `6ae34a2`
- Results: `c79f881`
- Run: `run_e06b_quiet_20260817_010421_confirmatory`

## S08 / E07

- Report: `docs/E07_State_Interventions_Report.md`
- Protocol freeze: `a0159b6`
- Analysis hardening: `4bb2019`
- Final cleanup: `6e1ee4c`
- Run: `run_e07_interv_20260817_144606_confirmatory`

## S09 / E08 and E09

- Protocol refreeze: `e8850c6`
- Raw trial freeze / confirmatory results: `4bc4b6b`
- Post-confirmatory repaired analysis: `7e65b52`
- E08 run: `run_e08_owner_20260817_181634_confirmatory`
- E09 run: `run_e09_meta_20260817_183133_confirmatory`

The repaired reports should identify:

- raw trial freeze;
- analysis version;
- correction commit;
- unchanged trial provenance.

## Live extensions

### E08c

- Primary-role counterbalance;
- exploratory N=4 shown in this site;
- N=16 confirmatory running at the time of this build.

### E09c

- fixed-target metacognitive interaction;
- exploratory N=4 shown in this site;
- N=16 confirmatory pending/running at the time of this build.

## External research references

### Cognitive offloading

Risko, E. F., & Gilbert, S. J. (2016). *Cognitive Offloading*. Trends in Cognitive Sciences, 20(9), 676–688. DOI: `10.1016/j.tics.2016.07.002`.

### Extended mind

Clark, A., & Chalmers, D. (1998). *The Extended Mind*. Analysis, 58(1), 7–19. DOI: `10.1111/1467-8284.00096`.

### Source monitoring

Johnson, M. K., Hashtroudi, S., & Lindsay, D. S. (1993). *Source monitoring*. Psychological Bulletin, 114(1), 3–28. DOI: `10.1037/0033-2909.114.1.3`.

### Metacognition

Koriat, A. (1993). *How do we know that we know? The accessibility model of the feeling of knowing*. Psychological Review, 100(4), 609–639. DOI: `10.1037/0033-295X.100.4.609`.

### Long context

Liu, N. F., et al. (2024). *Lost in the Middle: How Language Models Use Long Contexts*. TACL, 12, 157–173. DOI: `10.1162/tacl_a_00638`.

### LLM memory management

Packer, C., et al. (2023). *MemGPT: Towards LLMs as Operating Systems*. arXiv:2310.08560.

### Role confusion

Ye, C., Cui, J., & Hadfield-Menell, D. (2026). *Prompt Injection as Role Confusion*. arXiv:2603.12277.

### Activation introspection

Lindsey, J. (2026). *Emergent Introspective Awareness in Large Language Models*. arXiv:2601.01828.

### Persistent-memory provenance

Xu, J., et al. (2026). *Memory Provenance Laundering in LLM Agents*. arXiv:2607.29167.

## How this site handles research connections

External sources are used to:

- explain neighboring concepts;
- provide historical vocabulary;
- identify alternative mechanisms;
- motivate stronger controls.

They are not used to claim that H1 reproduces human cognition or independently validates the project's empirical conclusions.

## Final provenance rule

> **The narrative may change. The trial artifact must not.**
