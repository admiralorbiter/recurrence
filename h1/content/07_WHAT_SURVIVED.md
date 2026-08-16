# What Survived
## The durable first-half H1 findings

After S04–S06 and repeated measurement hardening, several conclusions still stand.

### 1. Explicit memory explains a lot

History dramatically improves performance over fresh invocation.

Future recurrence claims must beat strong explicit-memory baselines.

### 2. Structured state is primarily a control surface

Its strongest demonstrated advantages are:

- bounded size;
- inspectability;
- typed variables;
- direct intervention;
- lower long-horizon prompt cost.

It is not currently established as generally more accurate than raw history.

### 3. A deterministic Level-1 scaffold can be stable

The explicit state architecture itself works.

### 4. Model-autonomous maintenance did not work reliably

Qwen2.5-3B lost most multi-slot state under the tested update protocols.

### 5. Deterministic Level-1 continuity is replayable

The event history plus deterministic transition rule rebuilds the same terminal explicit state.

### 6. Single-pass model reconstruction remains lossy

After the interface was repaired, Qwen2.5-3B still performed substantially worse when asked to reconstruct the compact state retrospectively in one pass.

Final E05d:

- maintained state: 60.4%;
- model reconstruction: 39.6%;
- difference: +20.8 percentage points;
- primary episode-level permutation p ≈ .0025.

### 7. Persistence can inherit errors

A memory architecture needs correction and regulation, not merely retention.

## What did not survive

The project should **not** carry forward these stronger claims:

- structured state is generally more accurate than transcript;
- online deterministic processing creates an irreducible explicit state;
- long horizons already show a resolved structured-state accuracy crossover;
- the first reconstruction collapse was entirely a model-memory failure;
- identity quiet ticks demonstrate autonomous cognition.
