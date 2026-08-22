# Literature Delta: Recurrence (2026-08-22)

## 1. Zhu et al. (2021) — *Neural Bellman-Ford Networks: A General Graph Neural Network Framework for Link Prediction*

- **Connection class**: DIRECT
- **What it actually established**: Formulates multi-hop relational path reasoning as generalized Bellman-Ford path solving, where pair representations are updated via learned composition operators $\mathbf{h}_{u, v} = \bigoplus_{(w, r, v)} \mathbf{h}_{u, w} \otimes \mathbf{e}_r$.
- **Why it became relevant**: Scout G & Scout H investigate the exact parameterization of path composition: how an accumulated historical state $m(u \to w)$ combines with a new edge $e(w \to v)$ to form $m(u \to v)$.
- **Hypothesis it suggests for us**: Composition operators must explicitly parameterize intermediate node compatibility (matching $w == w$) to prevent unbound reachability shortcuts.
- **What our evidence does NOT yet establish**: Whether an additive residual accumulator can enforce intermediate node matching without explicit multi-relational tensor operators.

---

## 2. Palm et al. (2018) — *Recurrent Relational Networks*

- **Connection class**: DIRECT
- **What it actually established**: Iterated message-passing over relational representations across multiple sequential steps to perform multi-step relational reasoning on graphs without depth-specific parameters.
- **Why it became relevant**: Tests whether a single recurrent relational operator can generalize across reasoning depths ($k=2 \to 3 \to 4 \to 5$).
- **Hypothesis it suggests for us**: Message passing between explicit relation slots stabilizes deeper composition.
- **What our evidence does NOT yet establish**: Whether slot-based message passing is necessary for continuous 1D spatial trajectories.

---

## 3. Elhage et al. (2021) — *A Mathematical Framework for Transformer Circuits*

- **Connection class**: ANALOGY (Mechanistic Analogy)
- **What it actually established**: The residual stream in Transformers acts as a shared linear communication bus where layers read from and additively write features without destructive overwriting.
- **Why it became relevant**: Provides an architectural analogy suggesting why non-destructive additive residual accumulation ($m + \Delta m$) enables history retention where convex squashing fails.
- **Hypothesis it suggests for us**: Additive state updates preserve earlier features in orthogonal linear subspaces.
- **What our evidence does NOT yet establish**: Does not prove that the internal dynamics of our RNN mirror Transformer attention heads.

---

## 4. Dayan (1993) / Stachenfeld et al. (2017) — *The Successor Representation in Reinforcement Learning and the Hippocampus*

- **Connection class**: ANALOGY (Conceptual Analogy)
- **What it actually established**: Multi-hop transition graphs decompose into an expected future occupancy matrix $M = \sum \gamma^t T^t$ that updates recursively via $M = I + \gamma T M$.
- **Why it became relevant**: Illustrates that multi-step reachability naturally admits a recursive additive structure.
- **Hypothesis it suggests for us**: State $m_t$ could function as a continuous learned successor representation.
- **What our evidence does NOT yet establish**: We have not demonstrated that $m_t$ computes discounted future state occupancies; it currently produces directional query margins.
