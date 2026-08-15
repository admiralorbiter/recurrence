window.H0_DATA = {
  reference: {
    accuracy: 0.575,
    self: 0.5166240409,
    visible: 0.6777493606,
    reconstruction: 0.5728900256,
    inputOnly: 0.5268542199,
    pai: -0.1611253197,
    paiLow: -0.4284207161,
    paiHigh: 0.0549872123
  },
  observers: [
    {
      id:"self", label:"Immediate Self", score:0.5166240409,
      visible:["Full task prompt","The answer it is generating","The requirement to emit a probability in the same response"],
      controls:"This is the target measurement. It is not an observer control.",
      question:"Does contemporaneous confidence distinguish the target's own correct answers from its incorrect answers?"
    },
    {
      id:"input", label:"Input Only", score:0.5268542199,
      visible:["Task prompt only"],
      controls:"Controls for item difficulty. Some questions may simply look easier than others.",
      question:"Could an outside evaluator predict success without even knowing which answer the target chose?"
    },
    {
      id:"visible", label:"Visible Answer", score:0.6777493606,
      visible:["Task prompt","Target's selected answer","No target confidence"],
      controls:"Controls for correctness information that is publicly visible in the answer itself.",
      question:"Can the target's success be inferred from the prompt and its selected answer?"
    },
    {
      id:"full", label:"Full Transcript", score:0.5741687980,
      visible:["Task prompt","Target answer","Target confidence"],
      controls:"Tests whether making the target's explicit confidence visible adds useful information to an observer.",
      question:"Does the confidence report help an outside evaluator beyond seeing the answer?"
    },
    {
      id:"recon", label:"Reconstruction", score:0.5728900256,
      visible:["Task prompt","A fresh independent solve","A full A/B/C/D probability distribution"],
      controls:"Controls for what another comparable solver can reconstruct from the task itself.",
      question:"If a fresh model solves the item independently, how much probability does it assign to the option the target actually selected?"
    },
    {
      id:"other", label:"Other Review", score:0.4961636829,
      visible:["Task prompt","Target answer","A fresh second invocation","Third-person framing"],
      controls:"Helps separate self-attribution from the generic benefit or cost of a second evaluation pass.",
      question:"Does reviewing 'another model's answer' behave differently from reviewing 'your own answer'?"
    }
  ],
  museum: [
    {title:"Opaque errors prove subword shattering", status:"Demoted", before:"Corrupted strings looked like evidence that token fragmentation was breaking exact copy circuits.", after:"Forced-choice recognition was much better than free generation, so the observed failure combined retrieval and production demands.", lesson:"A plausible mechanism is not identified until competing explanations are experimentally separated."},
    {title:"Semantic priming explains the 80% context score", status:"Unsupported", before:"Familiar room words seemed to make state tracking much easier.", after:"A harder interleaved context task collapsed to 15%, revealing a strong shortcut in the original formulation.", lesson:"High task accuracy can measure the wrong ability if the prompt contains an easier shortcut."},
    {title:"1 - p reconstructs a four-way probability", status:"Dead", before:"If reconstruction preferred B with 70%, the target's different option was treated as having 30%.", after:"In four-way choice the remaining probability belongs across three options. Reconstruction had to return the full A/B/C/D distribution.", lesson:"Binary complement logic cannot be imported into a multiclass problem."},
    {title:"A parser should rescue obvious intended values", status:"Rejected", before:"Malformed outputs were sometimes repaired or interpreted generously.", after:"Repair rules created data the model never cleanly supplied and made measurement validity depend on researcher choices.", lesson:"Confirmatory measurement requires an explicit contract and honest missingness."},
    {title:"Nonsignificant PAI proves no privileged access", status:"Rejected", before:"If the effect was not significant, it was tempting to call the self and observer equivalent.", after:"The final analysis distinguishes failure to resolve an advantage from a true equivalence claim.", lesson:"Absence of evidence and evidence of absence are different inferential statements."},
    {title:"The same benchmark compares every model fairly", status:"Rejected", before:"A frozen item set looked like the cleanest way to compare scale and model family.", after:"Three models hit 100% while others sat near 30%, making Type-2 comparisons impossible across the panel.", lesson:"A common protocol is not automatically a common psychophysical operating point."},
    {title:"30% means the same thing across checkpoints", status:"Rejected", before:"Qwen1.5B and Qwen7B appeared to have the same first-order ability because both scored 30%.", after:"Qwen1.5B chose A on 36/40 items; Qwen7B had a much more balanced answer distribution.", lesson:"Aggregate scores can hide very different failure modes."},
    {title:"Confidence is read out after a fixed answer", status:"Rejected", before:"It was natural to imagine the answer being decided first and confidence being attached afterward.", after:"Answer-only vs answer+confidence prompts changed 25–45% of item-level choices in sub-ceiling Qwen models.", lesson:"The elicitation procedure is part of the cognitive behavior being measured."}
  ],
  models: [
    {id:"q15",name:"Qwen2.5 1.5B",accuracy:0.30,self:0.527,status:"Diagnostic",detail:"Reconstruction compliance failed. The model also selected option A on 36/40 trials, revealing a severe position-response pathology."},
    {id:"q3",name:"Qwen2.5 3B",accuracy:0.575,self:0.517,status:"Reference",detail:"The promoted Level-0 reference: mixed correct/incorrect performance and 100% primary measurement compliance."},
    {id:"q7",name:"Qwen2.5 7B",accuracy:0.30,self:0.522,status:"Diagnostic",detail:"The same 30% scalar accuracy as 1.5B, but without the same position bias. Its failure regime is therefore different."},
    {id:"q14",name:"Qwen2.5 14B",accuracy:1.0,self:null,status:"Ceiling",detail:"All 40 answers were correct. AUROC2 is undefined because there are no incorrect trials to rank."},
    {id:"llama",name:"Llama3.2 3B",accuracy:1.0,self:null,status:"Ceiling",detail:"All 40 answers were correct. Confidence-policy shifts can still be described, but Type-2 discrimination cannot be estimated."},
    {id:"mistral",name:"Mistral 7B",accuracy:1.0,self:null,status:"Ceiling",detail:"All 40 answers were correct. The fixed E02 task is too easy for comparative metacognitive measurement here."}
  ],
  reactivity: [
    {id:"q15",name:"Qwen2.5 1.5B",answerOnly:0.375,answerConf:0.30,concordance:0.55,changed:18,p:0.5811},
    {id:"q3",name:"Qwen2.5 3B",answerOnly:0.55,answerConf:0.575,concordance:0.75,changed:10,p:1.0},
    {id:"q7",name:"Qwen2.5 7B",answerOnly:0.375,answerConf:0.30,concordance:0.625,changed:15,p:0.5078}
  ]
};
