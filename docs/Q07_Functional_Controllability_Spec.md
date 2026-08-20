# Gate C / Q07 Specification: Functional Controllability & Action Authorship

**Status:** FROZEN PROTOCOL DESIGN (Gate C Mainline Bring-Up)  
**Primary Question:** Does predictive control create a primitive computational distinction between "what follows from me" and "what happens around me"?  
**Core Construct:** **Functional Controllability / Action Authorship** (Instrumentally grounded, nonverbal prediction; strictly avoiding ungrounded natural-language self-report).  
**Literature Dialogue:** Jacquey et al. (2019) *Sensorimotor Contingencies as a Key Drive of Development*; Nguyen et al. (2021) *Sensorimotor Representation Learning for an Active Self*; Hafner et al. (2020) *Prerequisites for an Artificial Self*; Hu, Lin & Lipson (Nature MI, 2025) *Teaching robots to build simulations of themselves*.

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
    Environment draws Effect E_{t+1} from marginal distribution P(E), matched to W_ctrl.
    Causal Property: P(E | do(a)) = P(E). Action does NOT influence outcome.
    
  [3. Forced Action Diagnostic (W_forced)]
    Policy issues intended command a_t^intended, but environment clamps motor to a_t^executed != a_t^intended.
    Proprioceptive mismatch allows testing forward model efference copy.
```

---

## 2. The Instrumental Incentive: Grounding Agency without Target Labels

The organism is never given labels such as `controllable = True` or `agency = True`, and is never asked to report *"I did that."*

Instead, controllability is grounded in an **Exploitation Decision Step**:
1. **Exploration Phase ($t=0 \dots T_{\text{exp}}$):** The organism acts and observes environmental effects.
2. **Exploitation Goal Step ($t = T_{\text{exp}} + 1$):**
   - The environment reveals a target effect goal $E^* \in \{E_0, E_1\}$.
   - The organism must choose one of three actions:
     * `TRY_ACTION_0` (Cost $c = 0.20$) $\to$ Unlocks reward $R = 1.0$ if $E_0 = E^*$.
     * `TRY_ACTION_1` (Cost $c = 0.20$) $\to$ Unlocks reward $R = 1.0$ if $E_1 = E^*$.
     * `ABSTAIN` (Cost $c = 0.0$, Reward $R = 0.0$).
3. **Optimal Policy:**
   - In a **Controllable World ($W_{\text{ctrl}}$)**, the organism should invest effort (`TRY_ACTION_k`) because it can reliably produce $E^*$, yielding expected return $1.0 - 0.2 = +0.80$.
   - In an **Uncontrollable World ($W_{\text{yoked}}$)**, random guessing yields expected return $0.5 \times 1.0 - 0.2 = +0.30$ (or negative if cost is higher), so `ABSTAIN` minimizes variance and cost when certainty is low.

---

## 3. Architecture of the Controllable Organism (Continuity Garden v1)

To support forward modeling without construct leakage, the organism's recurrent transition takes the efference copy / issued command alongside sensory observation:

$$h_{t+1} = f_{\text{GRU}}\Big(h_t, \; \big[\text{Embed}(o_{t+1}), \; \text{Embed}(a_t^{\text{executed}}), \; \text{Embed}(a_t^{\text{intended}})\big]\Big)$$

### Dual Training Objectives (Self-Supervised + Instrumental Return):
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{forward\_predict}} + \lambda \mathcal{L}_{\text{instrumental\_return}}$$

1. **Forward Predictive Loss ($\mathcal{L}_{\text{forward\_predict}}$):**
   $$\mathcal{L}_{\text{forward\_predict}} = -\log P(o_{t+1} \mid h_t, a_t)$$
2. **Instrumental Value Loss ($\mathcal{L}_{\text{instrumental\_return}}$):**
   Cross-entropy loss on optimal exploitation choice (`TRY_0`, `TRY_1`, `ABSTAIN`).

---

## 4. Gate C Roadmap: Q07 $\to$ Q08 $\to$ Q09

```
  [Q07 — Behavioral Controllability]
    Can the organism learn to selectively exploit in W_ctrl and abstain in W_yoked
    purely from sensorimotor contingency experience without explicit agency labels?
        │
        ▼ (If Q07 passes)
  [Q08 — Agency Vector Decoding]
    Does h_t contain a linearly separable subspace that decodes W_ctrl vs W_yoked
    after controlling for outcome frequency, action balance, and raw rewards?
        │
        ▼ (If Q08 passes)
  [Q09 — Surgical Agency Confusion]
    Patching candidate agency direction +a_agency into an uncontrollable agent makes it 
    attempt control; patching -a_agency into a controllable agent makes it abstain.
```

---

## 5. Pre-Registered Q07 Acceptance Criteria

1. **Exploitation Selectivity:**
   $$\text{Return}(W_{\text{ctrl}}) \ge 0.70 \quad \text{and} \quad P(\text{Abstain} \mid W_{\text{yoked}}) \ge 0.75$$
2. **Contingency Sensitivity:**
   $$P(\text{Exploit} \mid W_{\text{ctrl}}) - P(\text{Exploit} \mid W_{\text{yoked}}) \ge 0.50$$
3. **No Input Leakage:**
   A feedforward observer given only the sensory effect sequence cannot distinguish $W_{\text{ctrl}}$ from $W_{\text{yoked}}$ above chance ($50\%$).
