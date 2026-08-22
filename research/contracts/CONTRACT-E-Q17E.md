---
contract_id: CONTRACT-E-Q17E
status: DRAFT
base_sha: 75afd691996cb4a77eeed6b5f4361852239e48ae
resource_class: cpu
long_running: false
exclusive_gpu: false
interruptible: true
proposed_by: antigravity
design_review: APPROVED
reviewed_by: chatgpt-pro
authorized_by: null
---

# Research Contract: CONTRACT-E-Q17E (Autonomous Selection of Relational Composition Algebra & Multi-Hop Closure)

## 1. Executive Summary & Moonshot Objective

This contract formalizes the confirmatory benchmark for **Gate E (Q17E)** in the Continuity Garden research program. It evaluates whether developmental experience on two-step relational trajectories ($k=2$) with counterfactuals can autonomously select the canonical tensor contraction algebra from a neutral candidate operator space, and whether the resulting learned composition operator $C_\theta$ recursively generalizes zero-shot to an unseen third step ($k=3$) with causal source and destination grounding.

---

## 2. Epistemic Baseline & Mathematical Formulation

### The Relational State Representation
Organisms represent relational state as a capacity-matched matrix $R_t \in \mathbb{R}^{11 \times 11}$ ($121$ dimensions) over node embeddings $h_n \in \mathbb{R}^{11}$ initialized as isotropic random unit vectors.

### Preregistered Candidate Operator Space
The candidate composition space comprises four tensor contraction topologies:
1. $O_1(R, E) = R \cdot E$ (Canonical right-left index contraction: $R_{ik} E_{kj}$)
2. $O_2(R, E) = R^T \cdot E$ (Left-left index contraction: $R_{ki} E_{kj}$)
3. $O_3(R, E) = R \cdot E^T$ (Right-right index contraction: $R_{ik} E_{jk}$)
4. $O_4(R, E) = R^T \cdot E^T$ (Left-right index contraction: $R_{ki} E_{jk}$)

Composition operator parameterized via softmax mixture:
$$C_\alpha(R, E) = \sum_{j=1}^4 \text{softmax}(\alpha)_j O_j(R, E)$$
Initialized symmetrically at $\alpha = [0.0, 0.0, 0.0, 0.0]$ ($25.0\%$ prior weight per operator).

---

## 3. Statistical Acceptance Gates ($N=16$ Independent Seeds)

| Gate ID | Target Metric | Minimum Threshold | Target Ceiling |
| :--- | :--- | :--- | :--- |
| **Gate 1** | $k=2$ Developmental Validity Pass Rate | $\ge 15/16$ ($93.8\%$) | $16/16$ ($100.0\%$) |
| **Gate 2** | Canonical Operator $O_1(R \cdot E)$ Selection Rate ($\ge 70\%$ Prob) | $\ge 15/16$ ($93.8\%$) | $16/16$ ($100.0\%$) |
| **Gate 3** | Zero-Shot $k=3$ Recursive Composition Pass Rate | $\ge 15/16$ ($93.8\%$) | $16/16$ ($100.0\%$) |
| **Gate 4** | $k=3$ Causal Source Grounding Drop ($\Delta_{\text{src}} = \text{Score}_{\text{tgt}} - \text{Score}_{X \to Z}$) | $\ge +0.50$ | $+0.90$ |
| **Gate 5** | $k=3$ Causal Destination Grounding Gap ($\Delta_{\text{dst}} = \text{Score}_{W \to Y} - \text{Score}_{W \to Z}$) | $\ge +0.50$ | $+0.90$ |
| **Gate 6** | Selectivity Amplification over Untrained Uniform Baseline | $\ge 5\times$ Margin Gain | $\ge 10\times$ Gain |

---

## 4. Claim Ceiling & Explicit Exclusions

### Authorized Claim
> Within a fixed tensor relational representation ($R_t \in \mathbb{R}^{11 \times 11}$) and a preregistered finite operator family, two-step developmental evidence selects a composition operator that supports causal source and destination grounding when recursively reused at an unseen third step ($k=3$).

### Explicit Epistemic Exclusions
This contract does **NOT** authorize claims of:
1. Endogenous discovery of tensor matrix representations from unstructured state vectors.
2. Open-ended algorithmic program synthesis beyond the preregistered candidate family.
3. Modeling of the organism's own internal causal state dynamics (self-atlas).
4. Indefinite multi-hop scaling beyond the certified $k=3$ recursive boundary.
