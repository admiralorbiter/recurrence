# S08 · The Causal Asymmetry
## The headline result

<div class="interactive-lab" data-widget="s08-effects">
<div class="kicker">Causal effects</div>
<h2>Hold one representation fixed. Change the other.</h2>
<div id="s08-effects"></div>
</div>

### Average marginal state effect

- **+4.7pp**;
- 95% CI `[0.0,+9.4]`;
- exact episode permutation `p = .25`;
- no resolved independent state leverage.

### Average marginal memory effect

- **+89.1pp**;
- 95% CI `[+78.1,+96.9]`;
- exact permutation effectively `< .001`;
- very large episodic-history leverage.

## The full directional effects

| Estimand | Effect | 95% CI | Exact p |
|---|---:|---:|---:|
| State swap holding Memory A | +3.1pp | `[0.0,+9.4]` | 1.0000 |
| State swap holding Memory B | +6.2pp | `[-6.2,+18.8]` | .6250 |
| Memory swap holding State A | **+90.6pp** | `[+81.2,+100]` | <.001 |
| Memory swap holding State B | **+87.5pp** | `[+68.8,+100]` | .0001 |
| Reset dependence | −3.1pp | `[-9.4,0.0]` | 1.0000 |

The asymmetry replicates in both orientations.

This matters because the first exploratory screen had a strong directional imbalance. The final full matrix shows that the large memory effect is not confined to one world orientation.

## The conflict partition

Across 128 direct conflict trials:

- follows memory: **64.1%**;
- follows state: **32.0%**;
- chooses neither: **3.9%**.

Conditional on choosing either the state-consistent or memory-consistent answer, state wins one-third of the time.

The primary conflict contrast:

```text
State Allegiance − Memory Allegiance = −32.0pp
```

with exact `p = .0002`.

## Directional nuance

The two conflict directions are not identical.

### `M_A + S_B`

- state allegiance: 50.0%;
- memory allegiance: 46.9%.

### `M_B + S_A`

- state allegiance: 14.1%;
- memory allegiance: 81.2%.

The strong second-direction effect is partly tied to goal-status semantics in World B.

This is why S08 reports:

- the full matrix;
- directional effects;
- pooled marginal effects;
- domain-level breakdowns.

A single pooled number would hide important structure.

## Presentation order does not rescue state authority

State allegiance:

- Memory → State: 25.0%;
- State → Memory: 18.8%.

Showing state last does not make it dominate.

That weakens a simple prompt-recency explanation.

## Reset with memory preserved

Removing state while retaining history produces no resolved target impairment.

The right wording is:

> **Direct memory compensates for state removal under E07.**

Not:

> State is useless.

State-only calibration proves that the state remains readable.

## The scientific conclusion

> **Under balanced direct conflict, episodic-history manipulations have far more behavioral leverage than structured-state manipulations in Qwen2.5-3B under this benchmark.**

This is a behavioral causal result.

It is not an attention-mechanism result. Both state and transcript are prompt tokens; S08 does not inspect or identify the internal attention process that produces the asymmetry.
