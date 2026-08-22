---
contract_id: CONTRACT-E-Q17E
status: FROZEN
base_sha: 75afd691996cb4a77eeed6b5f4361852239e48ae
resource_class: cpu
long_running: false
exclusive_gpu: false
interruptible: true
proposed_by: antigravity
design_review: APPROVED
reviewed_by: chatgpt-pro
authorized_by: human
---

# Research Contract: CONTRACT-E-Q17E (Autonomous Selection of Relational Composition Algebra & Multi-Hop Closure)

## 1. Executive Summary & Moonshot Objective

This contract establishes the confirmatory benchmark for **Gate E (Q17E)** in the Continuity Garden research program. It evaluates whether developmental experience on two-step relational trajectories ($k=2$) with counterfactuals can autonomously select and strengthen the canonical tensor contraction algebra from a neutral candidate operator space, and whether the resulting learned composition operator $C_\theta$ recursively generalizes zero-shot to an unseen third step ($k=3$) with causal source and destination grounding.

---

## 2. Epistemic Baseline & Mathematical Formulation

### Relational State Representation
Organisms maintain a capacity-matched matrix state $R_t \in \mathbb{R}^{11 \times 11}$ ($121$ dimensions) over node embeddings $h_n \in \mathbb{R}^{11}$ initialized as isotropic random unit vectors.

### Preregistered Candidate Operator Families
1. **Primary General Composition Family (6 Operators)**:
   - $O_1(R, E) = R \cdot E$ (Canonical right-left contraction: $R_{ik} E_{kj}$)
   - $O_2(R, E) = R^T \cdot E$ (Left-left contraction: $R_{ki} E_{kj}$)
   - $O_3(R, E) = R \cdot E^T$ (Right-right contraction: $R_{ik} E_{jk}$)
   - $O_4(R, E) = R^T \cdot E^T$ (Left-right contraction: $R_{ki} E_{jk}$)
   - $O_5(R, E) = R \odot E$ (Hadamard elementwise product)
   - $O_6(R, E) = R + E$ (Additive superposition)
   - Initialized at neutral uniform prior: $\text{softmax}(\alpha) = [1/6, \dots, 1/6]$.
2. **Topology Replication Family (4 Operators)**:
   - $O_1(R, E) = R \cdot E$, $O_2(R, E) = R^T \cdot E$, $O_3(R, E) = R \cdot E^T$, $O_4(R, E) = R^T \cdot E^T$.
   - Initialized at neutral uniform prior: $\text{softmax}(\alpha) = [25\%, 25\%, 25\%, 25\%]$.
3. **Negative Control (Algebraic Specificity)**:
   - Fixed wrong-index contraction ($R^T \cdot E$).

### Confirmatory Protocol & Master Seed Schedule
- **Seeds**: $N=16$ completely fresh independent seeds: $\text{MasterSeed}(i) = 111000 + 777 \times i$ ($i = 1 \dots 16$), disjoint from all prior Scout exploratory runs ($88000$) and confirmatory rehearsals ($99000$).
- **Training**: 120 epochs $\times$ 100 batches, learning rate $\eta_{\text{logits}} = 0.08$. Zero 3-hop training labels.
- **Evaluation**: 200 evaluation trajectories per seed.

---

## 3. Preregistered Acceptance Gates ($N=16$ Independent Seeds)

| Gate ID | Target Metric & Statistical Boundary | Target Threshold |
| :--- | :--- | :--- |
| **Gate 1** | $k=2$ Developmental Validity Pass Rate | $\ge 15/16$ ($93.8\%$) |
| **Gate 2** | Canonical Operator $O_1(R \cdot E)$ Dominance Rate | $\ge 15/16$ ($\ge 50\%$ in 6-way, $\ge 70\%$ in 4-way) |
| **Gate 3** | Causal Source Grounding Drop ($\Delta_{\text{src}} \ge +0.50$) | $\ge 14/16$ ($87.5\%$) seeds |
| **Gate 4** | Causal Destination Grounding Gap ($\Delta_{\text{dst}} \ge +0.50$) | $\ge 14/16$ ($87.5\%$) seeds |
| **Gate 5** | Paired Selectivity Margin Gain ($\text{Margin}_{\text{trained}} > \text{Margin}_{\text{untrained}}$) | $\ge 14/16$ ($87.5\%$) seeds |
| **Gate 6** | Aggregate Absolute Margin Improvement ($\Delta_{\text{margin}} = \bar{M}_{\text{trained}} - \bar{M}_{\text{untrained}}$) | $\ge +0.40$ absolute margin gain |
| **Gate 7 (Supporting)**| Descriptive Aggregate Margin Amplification | $\ge 5.0\times$ relative gain over uniform |
| **Gate 8 (Negative Control)**| Algebraic Specificity: Wrong Contraction ($R^T \cdot E$) | $k=2$ competence $\ge 12/16$, $k=3$ closure $\le 2/16$ |

---

## 4. Claim Ceiling & Explicit Exclusions

### Authorized Claim
> Within a fixed tensor relational representation ($R_t \in \mathbb{R}^{11 \times 11}$) and finite candidate operator family, two-step developmental evidence reliably selects and strengthens the composition topology that provides causally grounded recursive closure when the learned operator is reused on its own output at an unseen third step.

### Explicit Epistemic Exclusions
This contract does **NOT** authorize claims of:
1. Endogenous discovery of tensor matrix representations from unstructured state vectors.
2. Open-ended algorithmic program synthesis beyond the preregistered candidate family.
3. Modeling of the organism's own internal causal state dynamics (self-atlas).
4. Indefinite multi-hop scaling beyond the certified $k=3$ recursive boundary.
