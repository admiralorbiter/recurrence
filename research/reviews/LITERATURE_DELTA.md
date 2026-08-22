# Literature Delta: Recurrence (2026-08-22)

### 1. Elhage et al. (2021) — *A Mathematical Framework for Transformer Circuits*
- **Relevance**: Explains why the residual stream acts as a communication bus where features add linearly rather than destructive non-linear replacement.
- **Mechanism Shift**: Explains why additive residual updates ($m + \Delta m$) succeed where convex gated updates ($\lambda m + (1-\lambda)\tilde{m}$) failed.

### 2. Liška et al. (2018) — *Memorize or Generalize? Searching for a Compositional RNN in a Small Sequence-to-Sequence Task*
- **Relevance**: Demonstrates that standard RNNs can achieve compositional solutions when given intermediate structural incentives, but default to memorization without them.
- **Connection to Recurrence**: Explains why prefix supervision must be paired with additive linear accumulation to prevent the recency attractor.

### 3. Dayan (1993) / Stachenfeld et al. (2017) — *Improving Generalization for Temporal Difference Learning: The Successor Representation*
- **Relevance**: Multi-hop transition maps decompose additively into current transition plus expected future transitions.
- **Connection**: Relational accumulator $m_t$ serves as an endogenous, learned continuous successor representation.
