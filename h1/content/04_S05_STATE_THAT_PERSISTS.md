# S05 — A State That Persists
## Can the system maintain memory through time?

S04 let the model read memory.

S05 asks whether a system can **carry a state forward**.

The Level-1 state contains:

- working-memory bindings;
- goals and goal status;
- source information;
- unresolved items;
- a logical clock.

## Two very different questions were hidden inside S05

### Can the scaffold maintain state?

Yes.

The deterministic state manager was stable, bounded, auditable, and recoverable.

### Can Qwen2.5-3B maintain the state autonomously?

Not reliably under the tested protocols.

| Update method | Macro retention | Terminal retention | Goal coherence |
|---|---:|---:|---:|
| Deterministic update | **100%** | **100%** | **100%** |
| Model delta update | 13.2% | 11.1% | 42.8% |
| Model full-state rewrite | 6.3% | 0.0% | 16.7% |

This produced the S05 split gate:

> **Scaffold: PASS. Model-autonomous maintenance: FAIL.**

The project therefore activated its fallback and used deterministic state transitions as the canonical Level-1 substrate.

## Why delta updates helped

Asking the model to rewrite the entire state every time was destructive.

A delta update asks only:

> What changed?

That roughly doubled retention.

But “better” did not mean “good enough.”

## Error inheritance

S05 produced one of the most important conceptual lessons in H1:

> **Persistence preserves errors as well as truths.**

A system that remembers forever can remember a mistake forever.

If an incorrect key enters a persistent state, deterministic maintenance can faithfully preserve the error.

So persistence needs more than storage.

It eventually needs:

- correction;
- conflict resolution;
- garbage collection;
- overwrite policies;
- regulation.

Memory is not automatically intelligence.
