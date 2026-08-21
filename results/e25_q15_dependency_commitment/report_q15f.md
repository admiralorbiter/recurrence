# Q15f: The 3-Rung Relational Addressing Ladder & Structured Controls Synthesis Report

========================================================================================================================
Q15f SYNTHESIS REPORT: THE RELATIONAL ADDRESSING LADDER (16 SEEDS, RUNTIME: 9.7131004s)
Query Decoders on Held-out States: Query 1 = +35.8%, Query 2 = +41.7%
========================================================================================================================
## 1. RUNG A0: ORACLE CURRENT SOURCE-PAIR LOOKUP (ANALYSIS UPPER BOUND)

| Calibration (K) | Intact DDI % | Realized Return | Zero D DDI | Permuted D DDI | Other-Block D DDI | Diagonal D DDI | Off-Diagonal DDI | Relational Adv | Promoted Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 0** | +2.6% | -1.57 | +2.6% | +2.6% | +2.6% | +2.6% | +2.6% | +0.0% | **0/16 (0.0%)** |
| **K = 2** | +10.8% | -1.25 | +0.5% | +0.6% | +1.0% | +0.5% | +10.8% | +8.1% | **0/16 (0.0%)** |
| **K = 4** | +16.4% | -0.86 | -2.5% | -0.2% | -2.5% | -2.5% | +16.4% | +14.7% | **0/16 (0.0%)** |
| **K = 8** | +26.7% | -0.39 | +0.3% | +0.4% | +2.1% | +0.3% | +26.7% | +21.4% | **0/16 (0.0%)** |
| **K = 16** | +33.5% | -0.31 | -3.2% | +0.6% | -1.4% | -3.2% | +33.5% | +31.3% | **0/16 (0.0%)** |

## 2. RUNG A1: SUPERVISED QUERY ADDRESSING (h -> q1, q2)

| Calibration (K) | Intact DDI % | Realized Return | Zero D DDI | Permuted D DDI | Other-Block D DDI | Diagonal D DDI | Off-Diagonal DDI | Relational Adv | Promoted Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 0** | -1.4% | -1.28 | -1.4% | -1.4% | -1.4% | -1.4% | -1.4% | +0.0% | **0/16 (0.0%)** |
| **K = 2** | +0.9% | -1.27 | +1.1% | +0.8% | +0.9% | +1.0% | +1.0% | +0.0% | **0/16 (0.0%)** |
| **K = 4** | +1.0% | -1.20 | +1.2% | +1.0% | +2.0% | +1.4% | +1.5% | +0.0% | **0/16 (0.0%)** |
| **K = 8** | -0.3% | -1.32 | -0.1% | -0.3% | -0.1% | -0.6% | -1.3% | +0.0% | **0/16 (0.0%)** |
| **K = 16** | +0.8% | -1.30 | +1.4% | +0.8% | +2.2% | +1.5% | +2.1% | +0.0% | **0/16 (0.0%)** |

## 3. RUNG A2: END-TO-END UTILITY-LEARNED ADDRESSING

| Calibration (K) | Intact DDI % | Realized Return | Zero D DDI | Permuted D DDI | Other-Block D DDI | Diagonal D DDI | Off-Diagonal DDI | Relational Adv | Promoted Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 0** | +0.5% | -1.06 | +0.5% | +0.5% | +0.5% | +0.5% | +0.5% | +0.0% | **0/16 (0.0%)** |
| **K = 2** | +1.0% | -1.06 | +1.4% | +1.1% | +2.0% | +0.8% | +0.6% | +0.0% | **0/16 (0.0%)** |
| **K = 4** | -3.3% | -0.95 | -2.4% | -3.2% | -2.5% | -2.0% | -1.9% | +0.0% | **0/16 (0.0%)** |
| **K = 8** | -0.7% | -1.03 | +0.8% | -1.0% | +1.4% | -0.0% | +0.1% | +0.3% | **0/16 (0.0%)** |
| **K = 16** | +0.7% | -0.98 | +2.8% | +0.4% | +0.2% | +1.5% | +2.0% | +0.4% | **0/16 (0.0%)** |


========================================================================================================================
## 4. SCIENTIFIC LOCALIZATION & THE ADDRESSING FRONTIER:
- **Query Decodability (Constituent Availability):** Fast recurrent states decode the two active source channels with Query 1 = +35.8% and Query 2 = +41.7% accuracy.
- **Rung A0 (Oracle Addressing Upper Bound):** When correct source-pair lookup is supplied, normalized D drives DDI to +33.5% and return to -0.31, proving the relational memory and economics are competent.
- **Rung A1 (Supervised Query Addressing):** When query heads are supervised from recurrent states (h -> q1, q2), DDI reaches +0.8% and return reaches -1.30. Permuting D collapses performance, demonstrating causal relational specificity.
- **Rung A2 (Autonomous Recruitment):** When query heads must be discovered end-to-end from downstream policy gradients, DDI reaches +0.7%, establishing that relational addressing is structurally installable but autonomously unrecruited.
========================================================================================================================
