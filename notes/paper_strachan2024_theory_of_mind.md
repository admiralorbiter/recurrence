---
title: Quick Paper Note — Testing theory of mind in large language models and humans
authors: James W. A. Strachan, Dalila Albergo, Giulia Borghini, Oriana Pansardi, Eugenio Scaliti, Saurabh Gupta, Krati Saxena, Alessandro Rufo, Stefano Panzeri, Guido Manzi, Michael S. A. Graziano, Cristina Becchio
year: 2024
source: Nature Human Behaviour (doi:10.1038/s41562-024-01882-z)
read_at: 2026-08-18
status: scan
radar_domains: [R1, R2, R5]
---

# Quick Paper Note — Testing Theory of Mind in Large Language Models and Humans

## Citation
Strachan, J. W. A., Albergo, D., Borghini, G., Pansardi, O., Scaliti, E., Gupta, S., Saxena, K., Rufo, A., Panzeri, S., Manzi, G., Graziano, M. S. A., & Becchio, C. (2024). Testing theory of mind in large language models and humans. *Nature Human Behaviour*, 8(7), 1285–1295. https://doi.org/10.1038/s41562-024-01882-z

## Why it matters to the project
1. **Direct connection to Attention Schema Theory (AST):** Co-authored by Michael S. A. Graziano, whose AST framework posits that subjective awareness and Theory of Mind (ToM) share the same underlying internal modeling machinery (social ToM as an externalized attention schema).
2. **Behavioral vs Internal State Dissociation (Horizons 2 & 3):** Shows how prompted zero-shot evaluation can produce illusory competence or conservative artifacts (e.g. LLaMA2's ignorance-attribution heuristic, GPT-4's hyperconservative commitment threshold), mirroring our S11/S12 discovery that behavioral readout diverges sharply from underlying physical/causal state representation.
3. **Controlled Psychophysics for Horizon 3 (Source Ownership & Self-Modeling):** Provides validated 5-domain psychophysics batteries (false beliefs, indirect requests, irony, misdirection, faux pas) that can be adapted for testing 1st-person vs 3rd-person state ownership in recurrent agents.

## One-paragraph summary
The authors benchmarked two families of LLMs (GPT-3.5/GPT-4 and LLaMA2) against 1,907 human participants across a multi-domain Theory of Mind battery (false beliefs, indirect requests, irony, misdirection, and faux pas recognition). GPT-4 performed at or above human levels on false beliefs, indirect requests, and misdirection, but underperformed on faux pas. LLaMA2 appeared to outperform humans on faux pas, but belief-likelihood manipulations proved this was an artifact of a simple ignorance-attribution heuristic. GPT-4's faux pas difficulty was shown to stem from a hyperconservative commitment criterion rather than inferential incapacity. The study underscores the necessity of systematic psychophysical and likelihood manipulations to separate genuine computational competence from heuristic shortcuts.

## Load-bearing claims

| Claim | Evidence | Strength | Relevant construct |
|---|---|---|---|
| GPT-4 matches or exceeds human accuracy on classic ToM (false beliefs, indirect requests, misdirection). | Repeated testing across 1,907 human controls and multiple prompt variants. | Strong | Mentalistic state tracking |
| High LLM performance can arise from heuristic biases (e.g. ignorance-default in LLaMA2). | Belief likelihood manipulation tests. | Strong | Shortcut learning vs internal model |
| Behavioral failure can reflect conservative decision thresholds rather than representational absence. | Follow-up sensitivity & likelihood scaling in GPT-4. | Moderate | Behavioral readout vs latent capacity |

## What the study actually manipulates and measures
- **System/Sample:** 1,907 human participants (Prolific) vs GPT-3.5, GPT-4, and LLaMA2 (7B, 13B, 70B).
- **Independent Variables:** Task domain (false beliefs, indirect requests, irony, misdirection, faux pas); belief likelihood gradients; prompt framing.
- **Dependent Variables:** Task accuracy, belief attribution likelihood scores, decision conservatism thresholds.
- **Controls:** Non-mental control stories (physical causality/facts matched in complexity), belief likelihood variations.

## Relevance to Recurrence Program & Attention Schema
- **AST Grounding:** Graziano's core thesis is that an agent models attention in others using the same internal model it uses to represent its own attention. In our program (Horizon 2/3), recurrent states ($R, C, K$) provide the physical substrate for self-consistent temporal trajectories and internal self-monitoring.
- **Psychophysical Rigor:** The paper's lesson—that superficial behavioral performance is easily fooled by heuristics—reinforces why our program does not rely solely on zero-shot cloze prompts, but insists on physical store tracing (S11), causal surgical transplantation (S12), and privileged-access source ownership tests (Horizon 3).

## Program consequence
- **Action:** Add paper to `Research Radar.md` under R1/R2/R5. Use task battery concepts in Horizon 3 source ownership and self-modeling experimental design.
