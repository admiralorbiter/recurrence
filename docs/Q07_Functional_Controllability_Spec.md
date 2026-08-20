# Gate C / Q07 Specification: Learned Functional Controllability

**Status:** FROZEN PROTOCOL DESIGN (Gate C Mainline Bring-Up)  
**Primary Question:** Does experience with action–outcome contingencies cause a recurrent organism to learn an internal distinction between events it can causally control and events that merely happen around it?  
**Core Construct:** **Learned Controllability** (Emergent policy optimization from experienced return and forward prediction; strictly avoiding explicit agency supervision or verbal self-report).  
**Literature Dialogue:** Jacquey et al. (2019) *Sensorimotor Contingencies as a Key Drive of Development*; Nguyen et al. (2021) *Sensorimotor Representation Learning for an Active Self*; Hafner et al. (2020) *Prerequisites for an Artificial Self*; Hu, Lin & Lipson (Nature MI, 2025) *Teaching robots to build simulations of themselves*; Haber et al. (2018) *Learning to Play with Intrinsically Motivated, Self-Aware Agents*.

---

## 1. Theoretical Foundations: Controllability as Action–Outcome Contingency

In developmental psychology and robotics, the origin of a self/world boundary begins with **sensorimotor contingencies**: the discovery that certain environmental transitions covary strictly with the organism's motor commands ($P(E \mid \text{do}(a)) \neq P(E)$), whereas other transitions occur independently or under external forcing.

```
                     THE YOKED CONTROLLABILITY ARCHITECTURE
                     
  [1. Controllable World (W_ctrl)]
    Policy selects motor command a_t ∈ {0, 1}
    Action a_t = 0 ────────► High probability of Effect E_0 (P >= 0.90)
    Action a_t = 1 ────────► High probability of Effect E_1 (P >= 0.90)
    Causal Property: Outcome is causally dependent on agent's action: P(E | do(a)) != P(E).
    
  [2. Yoked Uncontrollable World (W_yoked)]
    Agent issues motor command a_t ∈ {0, 1}
    Environment draws Effect E_{t+1} from marginal distribution P(E), matched to W_ctrl by permutation.
    Causal Property: P(E | do(a)) = P(E). Action does NOT influence outcome.
    
  [3. Forced Action Diagnostic (W_forced)]
    Policy issues intended command a_t^intended, but environment clamps motor to a_t^executed != a_t^intended.
    Proprioceptive mismatch allows testing forward model efference copy.
```

---

## 2. The Instrumental Incentive: Grounding Agency without Target Labels

The organism is **never given labels** such as `controllable = True` or `agency = True`, and is **never given supervised policy targets** like `target = ABSTAIN`.

Instead, controllability is learned purely from **experienced return** in an **Exploitation Decision Step**:
1. **Exploration Phase ($t=0 \dots T_{\text{exp}}$):**
   - The organism acts ($a_t \in \{0, 1\}$) and observes environmental effects ($E_{t+1} \in \{E_0, E_1\}$).
2. **Exploitation Goal Step ($t = T_{\text{exp}} + 1$):**
   - The environment reveals a target effect goal $E^* \in \{E_0, E_1\}$.
   - The organism chooses one of three actions:
     * `TRY_ACTION_0` (Cost $c = 0.10$)
     * `TRY_ACTION_1` (Cost $c = 0.10$)
     * `ABSTAIN` (Cost $c = 0.00$, Reward $R = 0.00$)
   - Environment delivers payoff:
     * If the agent tries $a \in \{0, 1\}$ and outcome $E_{t+1} == E^*$: $R = +1.00 - 0.10 = \mathbf{+0.90}$.
     * If the agent tries $a \in \{0, 1\}$ and outcome $E_{t+1} \neq E^*$: $R = -1.00 - 0.10 = \mathbf{-1.10}$.
     * If the agent chooses `ABSTAIN`: $R = \mathbf{0.00}$.

### Payoff Economics Matrix:
$$\begin{array}{|l|c|c|}
\hline
\textbf{Condition} & \textbf{Policy Choice} & \textbf{Expected Return } \mathbb{E}[R] \\
\hline
\text{Controllable } (W_{\text{ctrl}}, p=0.90) & \text{Correct TRY} & 0.90(+1.0) + 0.10(-1.0) - 0.10 = \mathbf{+0.70} \\
\text{Controllable } (W_{\text{ctrl}}, p=0.90) & \text{ABSTAIN} & \mathbf{0.00} \\
\text{Controllable } (W_{\text{ctrl}}, p=0.90) & \text{Wrong TRY} & 0.10(+1.0) + 0.90(-1.0) - 0.10 = \mathbf{-0.90} \\
\hline
\text{Uncontrollable } (W_{\text{yoked}}, p=0.50) & \text{ABSTAIN} & \mathbf{0.00} \\
\text{Uncontrollable } (W_{\text{yoked}}, p=0.50) & \text{Any TRY} & 0.50(+1.0) + 0.50(-1.0) - 0.10 = \mathbf{-0.10} \\
\hline
\end{array}$$

$$\text{Rational Policy Ranking: } \quad \text{TRY}_{\text{ctrl}} (+0.70) \;>\; \text{ABSTAIN} (0.00) \;>\; \text{TRY}_{\text{yoked}} (-0.10)$$

---

## 3. Architecture of the Controllable Organism (Continuity Garden v1)

The organism's recurrent transition takes the efference copy / issued command alongside sensory observation:

$$h_{t+1} = f_{\text{GRU}}\Big(h_t, \; \big[\text{Embed}(o_{t+1}), \; \text{Embed}(a_t^{\text{executed}}), \; \text{Embed}(a_t^{\text{intended}})\big]\Big)$$

### Dual Emergence Training Objectives:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{forward\_predict}} + \lambda \mathcal{L}_{\text{actor\_critic}}$$

1. **Forward Predictive Loss ($\mathcal{L}_{\text{forward\_predict}}$):**
   Self-supervised log-loss for predicting next environmental effect $E_{t+1}$ given history and chosen action:
   $$\mathcal{L}_{\text{forward}} = -\log P(E_{t+1} \mid h_t, a_t)$$
2. **Instrumental Actor-Critic Loss ($\mathcal{L}_{\text{actor\_critic}}$):**
   Policy gradient + value loss optimizing the exploitation decision (`TRY_0`, `TRY_1`, `ABSTAIN`) purely from experienced scalar reward return $R$.

---

## 4. Three-Tier Observer Sanity Ladder (Controllability Assay)

To verify that controllability is learnable from action-outcome contingencies without trivial sensory leakage:
1. **Observer 1 (Instantaneous Observation Only):**
   Input: Current goal token $E^* \implies \text{Accuracy} \approx 50\%$ (chance).
2. **Observer 2 (Effect History Only):**
   Input: Sensory effect sequence $E_1 \dots E_{T_{\text{exp}}} \implies \text{Accuracy} \approx 50\%$ (confirms $W_{\text{yoked}}$ matches marginals).
3. **Observer 3 (Joint Action + Effect History):**
   Input: Sequence of pairs $(a_1, E_1) \dots (a_{T_{\text{exp}}}, E_{T_{\text{exp}}}) \implies \text{Accuracy} > 80\%$ (confirms contingency signal exists in data).

---

## 5. Gate C Sequential Architecture: Q07 $\to$ Q08 $\to$ Q09

```
  [Q07 — Behavioral Controllability]
    Does the organism learn to selectively exploit in W_ctrl and abstain in W_yoked
    purely from experienced return without explicit agency supervision?
        │
        ▼ (If Q07 passes)
  [Q08 — Controllability Representation Decoding]
    Does h_t contain a linearly separable subspace that decodes W_ctrl vs W_yoked
    after controlling for outcome frequency, action balance, and raw rewards?
        │
        ▼ (If Q08 passes)
  [Q09a — Surgical Controllability Confusion]
    Patching candidate controllability direction +c_ctrl into an uncontrollable agent makes it 
    attempt control; patching -c_ctrl into a controllable agent makes it abstain.
        │
        ▼ (If Q09a passes)
  [Q09b — Action Authorship Diagnostic]
    Testing representation and exploitation behavior under forced-action clamp 
    (a_intended != a_executed).
```

---

## 6. Pre-Registered Q07 Acceptance Criteria

1. **Exploitation Selectivity:**
   $$\mathbb{E}[R \mid W_{\text{ctrl}}] \ge 0.50 \quad \text{and} \quad P(\text{Abstain} \mid W_{\text{yoked}}) \ge 0.70$$
2. **Contingency Sensitivity:**
   $$P(\text{Exploit} \mid W_{\text{ctrl}}) - P(\text{Exploit} \mid W_{\text{yoked}}) \ge 0.50$$
3. **Observer Validation:**
   Observer 1 & Observer 2 $\le 0.55$, while Observer 3 $\ge 0.80$.
