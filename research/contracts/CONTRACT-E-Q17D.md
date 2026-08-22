---
contract_id: CONTRACT-E-Q17D
status: DRAFT
proposed_by: antigravity
design_review: null
reviewed_by: chatgpt-pro
authorized_by: null
base_sha: 1cb1ccce3df3f2ebfef7d53399d1f5dc242b7f0d
execution_base_sha: null
resource_class: cpu
long_running: false
exclusive_gpu: false
interruptible: true
---

# Research Contract Proposal: CONTRACT-E-Q17D (Incubation / Next Frontier)

## Title
Gate E Frontier: Unseen Multi-Hop Depth Generalization (3-Hop to 5-Hop) from Endogenous Recurrent Causal History

## 1. Context & Research Question
In exploration stage Q17B, self-supervised 2-hop composition was established under an external transition table. Stage Q17C isolates endogenous memory migration ($z_t$ as the sole history carrier) and is currently undergoing confirmatory review.

Stage Q17D germinates the next logical scaling frontier: **Out-of-Distribution Compositional Depth Generalization**.

```
Q17B: validated external-memory 2-hop composition
Q17C: candidate endogenous-memory migration
Trajectory (A->B, B->C) -> Persistent z_t -> Readout predicts A->C reachability

Q17D (Multi-Hop Generalization):
Trajectory (A->B, B->C, C->D, D->E) -> Persistent z_t -> Readout predicts A->D (3-hop), A->E (4-hop) reachability
(Model weights trained strictly on 2-step sequences; zero multi-hop exposure during meta-training)
```

### Core Research Question
Can an endogenous recurrent memory representation trained exclusively on short 2-hop transitions generalize zero-shot to predict compositional reachability across unseen 3-hop, 4-hop, and 5-hop causal chains?

---

## 2. Conditional Hypotheses & Research Branches

Following the continuous incubation rule, Q17D maintains an explicit **promotion fence**:
- **Promotion Dependency**: Branch Q17D-A becomes eligible for freezing and execution **ONLY IF** `CONTRACT-E-Q17C` is formally promoted following scientific review. Current Q17C observations remain exploratory and are not treated as established baseline truth.

### Branch Q17D-A: Depth Generalization (Primary Path upon Q17C Promotion)
- **Hypothesis**: The recurrent dynamic update $z(t+1) = g_\theta(z(t), x(t))$ forms an iterative compositional operator such that multi-step reachability naturally emerges from recursive state integration.
- **Protocol**: Expose organism to $k$-hop paths ($k \in \{3, 4, 5\}$); evaluate directional margins $m_{k} = \text{score}(\text{Start} \to \text{End}) - \text{score}(\text{End} \to \text{Start})$.
- **Assay**: Multi-hop conflict resolution and transposition discrimination across depth $k \in \{3, 4, 5\}$.

### Branch Q17D-B: Exploratory Bottleneck Scouting (Contingency Path)
- **Hypothesis**: If depth generalization decays sharply at $k \ge 3$, conduct a cheap development-only exploratory scout to locate the bottleneck (e.g. state saturation vs. readout projection) prior to formulating any subsequent confirmatory contract.

---

## 3. Structural Scaffolding & Zero-Sidecar Invariant
- **Zero Query-Time Ledger**: All multi-hop paths must be encoded sequentially into $z_t$ during online life experience.
- **Fixed Synaptic Parameters**: Weights $\theta$ remain frozen at test time.
- **Structural Interface**: Query function receives strictly `query(z_t, (start_node, target_node), weights)`.

---

## 4. Epistemic Scope Ceilings
- **Claim Ceiling**: Explores zero-shot depth composition in synthetic graph topologies.
- **Exclusions**: Does NOT claim general algorithmic graph-search capabilities or open-domain semantic reasoning.
