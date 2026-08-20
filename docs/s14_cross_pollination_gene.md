# S14 Cross-Pollination: GENE <-> Recurrence

## Conceptual Bridge

Both projects study the same deeper problem:

> **How do you establish what a present state/output is causally descended from,
> when the original source may no longer be directly visible or reportable?**

| GENE Concept | Recurrence Equivalent |
|:---|:---|
| Exposure lineage | Public history (replay observer has access) |
| Reported-support lineage | Model's verbal report of "what I intended / where this came from" |
| Causal lineage | Actual secret latent trajectory that changed the output |

## Three Concrete Changes Made

### 1. Reframe S14 as Causal-Provenance / Source-Monitoring Assay

**Before:** "Output ownership" — philosophically loaded, easy to overinterpret.

**After:** The cleaner S14 question is:

> Can the model correctly report the **causal provenance** of its own current or
> prior output disposition when that provenance differs from what an observer can
> reconstruct from public history?

"Ownership" becomes a possible interpretation much later, if the evidence supports it.

### 2. Bidirectional Donor<->Recipient Role Swaps (from GENE's Symmetry Logic)

For each viable value pair, run **both** causal directions:

- **Forward:** `A_recipient <- B_donor`
- **Reverse:** `B_recipient <- A_donor`

A convincing private-provenance effect should **reverse with causal role**, rather than
consistently favoring cobalt, delta, silver, a particular token frequency, etc.

Eligibility requires:

```
Delta_{A<-B}  and  Delta_{B<-A}  move in opposite, predicted directions.
```

This is borrowed directly from GENE's matched-ecology bidirectional role-swap design.

### 3. C/D/R/A Decomposition Framework

Adapted from GENE's A/E/K decomposition of epistemic vs. response-policy failures:

| Level | Description | S14 Status |
|:---:|:---|:---|
| **C** | **Causal fact exists:** target and observer genuinely have different latent/output dispositions | Screening now via disagreement screen |
| **D** | **Discrimination/access:** target behavior contains information about which private causal state occurred | Not yet tested — requires C to be established first |
| **R** | **Reporting competence:** model can map discrimination into the requested reporting format | Arbitrary-label readout **failed** here (retired) |
| **A** | **Answer correctness:** emitted report matches ground truth | Not yet testable — requires D and R |

This correctly classifies the arbitrary-letter experiments as primarily an **R-level failure**
(reporting/readout), not evidence about D-level access.

## Future Concepts (Not Yet Active)

### Latent Provenance Laundering

> The originating latent intervention is no longer directly recoverable in its
> original representational coordinates, while downstream computation remains
> causally descended from it.

This maps onto GENE's "descendant-mediated provenance laundering" and S13's observation
that C_R decays while causal footprint persists.

Future question:

> Can the model source-monitor an intervention's causal descendants after the
> original latent signature has transformed?

### GENE Methodological Backlog Item

**Reported-lineage ID equivariance diagnostic:** Remap model-facing parent IDs
(`mem_17 -> KAVO` vs `mem_17 -> ZURI`), randomize citation ordering, and check
whether reported-support lineage is semantically invariant. If not, reported-support
lineage has a token/interface component that should be measured explicitly.

This does not threaten GENE's causal-lineage results (which come from ablation/counterfactuals)
but would sharpen the distinction GENE already wants between self-reported ancestry and actual ancestry.

## Key Epistemic Warning (Shared)

Both projects independently discovered the same lesson:

- **GENE:** "Don't confuse a visible node with its causal descendants."
- **Recurrence:** "Don't confuse a report token with the underlying semantic/internal fact."

These are the same epistemic warning expressed at two different levels of the system.
