window.H1_DATA = {
  "s04": [
    {
      "name": "Fresh",
      "acc": 35.7,
      "tok": 109
    },
    {
      "name": "Full transcript",
      "acc": 81.0,
      "tok": 499
    },
    {
      "name": "Deterministic summary",
      "acc": 61.9,
      "tok": 274
    },
    {
      "name": "Model summary",
      "acc": 69.0,
      "tok": 469
    },
    {
      "name": "Structured state",
      "acc": 64.3,
      "tok": 371
    },
    {
      "name": "Combined",
      "acc": 66.7,
      "tok": 730
    }
  ],
  "s05": {
    "Deterministic": {
      "retention": 100,
      "terminal": 100,
      "goal": 100
    },
    "Model delta": {
      "retention": 13.2,
      "terminal": 11.1,
      "goal": 42.8
    },
    "Full-state rewrite": {
      "retention": 6.3,
      "terminal": 0,
      "goal": 16.7
    }
  },
  "s06": {
    "horizons": [
      {
        "t": 10,
        "incremental": 65.6,
        "transcript": 71.9,
        "incTok": 418.4,
        "transTok": 576.2
      },
      {
        "t": 25,
        "incremental": 56.2,
        "transcript": 71.9,
        "incTok": 419.5,
        "transTok": 782.4
      },
      {
        "t": 50,
        "incremental": 59.4,
        "transcript": 59.4,
        "incTok": 424.7,
        "transTok": 1063.6
      }
    ]
  },
  "s09": {
    "e08c": {
      "roleShift": 40.0,
      "alphaBias": 5.0,
      "roleA_self": 55.0,
      "roleA_peer": 10.0,
      "roleB_self": 50.0,
      "roleB_peer": 15.0
    }
  }
};
window.GLOSSARY = {
  "explicitmemory": {
    "term": "explicit memory",
    "short": "History stored outside the model in an inspectable form such as a transcript, summary, or structured state.",
    "long": "H1 uses explicit memory as a strong control: if written history already explains a behavior, hidden recurrence should not receive credit for it."
  },
  "structuredstate": {
    "term": "StructuredSelfState",
    "short": "The typed Level-1 state object for working memory, goals, sources, unresolved items, and derived inferences.",
    "long": "It is not assumed to be a mind or self. It is a bounded, inspectable, clonable experimental control surface."
  },
  "persistence": {
    "term": "persistence",
    "short": "A later state depends on an earlier state rather than every episode beginning from scratch.",
    "long": "H1 studies scaffolded persistence implemented with explicit memory and deterministic state management."
  },
  "recurrence": {
    "term": "recurrence",
    "short": "Earlier internal state is fed back to influence later processing.",
    "long": "The project reserves the stronger native recurrence question for H2, where hidden recurrent state itself becomes the causal object."
  },
  "latentstate": {
    "term": "latent state",
    "short": "A hidden internal representation rather than visible prompt text or JSON.",
    "long": "H2 asks whether latent state carries history that cannot be reduced to rereading or deterministically reconstructing an external record."
  },
  "transcript": {
    "term": "episodic transcript",
    "short": "The raw chronological event history shown to the model.",
    "long": "The transcript is a strong control because it preserves observable history without requiring a separate compact representation."
  },
  "detreplay": {
    "term": "deterministic replay",
    "short": "Rebuilding state later by applying the same transition rules to the same ordered history.",
    "long": "If online maintenance and later replay produce the same state, deterministic Level-1 continuity is algorithmically reconstructible."
  },
  "modelrecon": {
    "term": "model reconstruction",
    "short": "Asking the model to compress history into structured state in one retrospective pass.",
    "long": "In S06 this was substantially lossier than deterministic incremental maintenance."
  },
  "nulltick": {
    "term": "null interval",
    "short": "A period with no new task-relevant exogenous information.",
    "long": "A null interval may contain computation; it simply does not add new task evidence."
  },
  "errorinheritance": {
    "term": "error inheritance",
    "short": "Persistence can preserve a false state just as faithfully as a true state.",
    "long": "S05 showed that reliable storage is not equivalent to reliable cognition."
  },
  "derivedwrite": {
    "term": "derived-state write failure",
    "short": "Failure to correctly externalize a valid conclusion into persistent derived state.",
    "long": "S07 found 0 exact correct target derivations among 274 available-inference writes under the tested selective reflection mechanism."
  },
  "epistemicquality": {
    "term": "epistemic-state quality",
    "short": "Whether the full state remains useful, coherent, and appropriately organized for later reasoning.",
    "long": "Protected facts can remain intact while derived or unresolved channels become noisy enough to interfere with readout."
  },
  "stateallegiance": {
    "term": "state allegiance",
    "short": "Under State \u00d7 Memory conflict, choosing the answer implied by structured state.",
    "long": "S08 compares state allegiance with memory allegiance to estimate which representation governs behavior under conflict."
  },
  "memoryallegiance": {
    "term": "memory allegiance",
    "short": "Under State \u00d7 Memory conflict, choosing the answer implied by episodic history.",
    "long": "In S08 balanced conflicts, memory allegiance substantially exceeded state allegiance."
  },
  "causalleverage": {
    "term": "causal leverage",
    "short": "How much a targeted intervention changes downstream behavior.",
    "long": "S08 holds one representation fixed while manipulating the other to separate state leverage from history leverage."
  },
  "sourceattribution": {
    "term": "source attribution",
    "short": "Identifying who or what originated an event or binding.",
    "long": "S09 separates remembering content from remembering provenance."
  },
  "sourceledger": {
    "term": "source ledger",
    "short": "A structured mapping from a binding to its recorded source class.",
    "long": "S09 compares this formal channel with transcript tags and narrative actor cues."
  },
  "selfattractor": {
    "term": "primary-agent response attractor",
    "short": "A tendency to attribute events to the actor designated as the primary/Self agent.",
    "long": "Canonical E08 showed frequent agent_alpha responses for non-self sources. E08c tests whether the bias follows the designated Self role."
  },
  "metacognition": {
    "term": "metacognition",
    "short": "Here: confidence that tracks whether a first-order decision is actually correct.",
    "long": "E09 compares Self-framed and Observer-framed correctness prediction on the same target decisions."
  },
  "brier": {
    "term": "Brier score",
    "short": "Mean squared error of a probability forecast against a binary outcome. Lower is better.",
    "long": "Brier captures calibration and accuracy of confidence forecasts."
  },
  "auroc": {
    "term": "AUROC",
    "short": "How often correct decisions receive higher confidence than incorrect decisions.",
    "long": "AUROC measures resolution, not absolute calibration."
  },
  "observer": {
    "term": "matched observer",
    "short": "An external evaluator given the same public evidence as the target.",
    "long": "A strong observer control prevents public cues from being misinterpreted as privileged self-access."
  },
  "noresolved": {
    "term": "no resolved difference",
    "short": "The prespecified inferential test did not distinguish the conditions.",
    "long": "This is not the same as proving equivalence or zero effect."
  },
  "equivalence": {
    "term": "equivalence",
    "short": "A stronger claim requiring a prespecified equivalence margin and appropriate test.",
    "long": "H1 avoids calling null results equivalent unless that test was actually designed."
  },
  "hardening": {
    "term": "measurement hardening",
    "short": "Removing shortcuts, bugs, weak controls, and scoring ambiguities until the construct is cleaner.",
    "long": "A central H1 result is that several attractive effects weakened or changed once the ruler improved."
  },
  "claimceiling": {
    "term": "claim ceiling",
    "short": "The strongest conclusion the evidence is allowed to support.",
    "long": "H1 supports engineering and behavioral conclusions about explicit state, not phenomenal-consciousness claims."
  },
  "rolecounterbalance": {
    "term": "role counterbalance",
    "short": "Swap which actor is designated Self while keeping names and task structure matched.",
    "long": "E08c distinguishes a primary-role anchor from a lexical preference for agent_alpha."
  },
  "fixedtarget": {
    "term": "fixed-target metacognition",
    "short": "Hold the exact same first-order decision fixed across assessment formats.",
    "long": "E09c removes a remaining confound in the original transcript-vs-scaffolded interaction."
  }
};
