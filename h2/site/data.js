window.H2 = {
  timeline: [
    {n:0, label:'Origin', conv:true, kv:true, r:.3384, recall:1.0, note:'The event is fresh. All stores can still carry direct traces.'},
    {n:3, label:'Conv turnover', conv:false, kv:true, r:.25, recall:1.0, note:'The four-token convolution buffer no longer directly contains the event.'},
    {n:256, label:'256', conv:false, kv:true, r:.12, recall:.55, note:'Behavioral retrieval is already deteriorating while recurrent separation remains measurable.'},
    {n:1024, label:'1024', conv:false, kv:true, r:.08, recall:.55, note:'The recurrent difference survives continued processing.'},
    {n:2048, label:'1W', conv:false, kv:false, r:.06, recall:.50, note:'The original event has turned over from the local attention window.'},
    {n:4096, label:'2W', conv:false, kv:false, r:.3384, recall:.50, note:'At 2W, branch-specific RG-LRU separation is resolved while the paired cloze probe is unresolved.'}
  ],
  transplant: [
    {id:'match', label:'Matching RG-LRU', short:'RIGHT PAST', estimate:74.10, ci:[46.79,106.72], copy:'Matching recurrent history strongly steers the recipient toward the donor trajectory.'},
    {id:'wrong', label:'Same-template wrong value', short:'WRONG VALUE', estimate:83.13, ci:[71.77,95.19], copy:'The wrong value still steers strongly. Causal influence alone is not specificity.'},
    {id:'cross', label:'Cross-template donor', short:'CROSS', estimate:75.75, ci:[58.92,92.08], copy:'Structured cross-history states also exert a large shared steering component.'},
    {id:'noise', label:'Matched noise', short:'NOISE', estimate:48.23, ci:[15.57,74.76], copy:'Matched noise perturbs the system, but structured historical states are more selective.'},
    {id:'whole', label:'Whole-state swap', short:'WHOLE', estimate:218.76, ci:[197.13,241.80], copy:'Whole-state transplantation is the positive-control ceiling.'}
  ],
  specificity: {
    value:38.49,
    ci:[25.82,50.85],
    projection:.1744,
    projectionCI:[.1001,.2536]
  },
  dynamics: [
    {n:0, v0:39.58, vn:39.58, cr:1.0000, qr:1.00, angle:0},
    {n:16, v0:4.85, vn:11.44, cr:.6092, qr:1.10, angle:22},
    {n:64, v0:-2.96, vn:31.96, cr:.4843, qr:1.10, angle:39},
    {n:256, v0:1.45, vn:7.21, cr:.3306, qr:1.24, angle:58},
    {n:1024, v0:-16.19, vn:9.57, cr:.1660, qr:1.79, angle:76},
    {n:2048, v0:4.70, vn:13.95, cr:.1238, qr:4.85, angle:83}
  ],
  strictC: {
    fwd:{truth:'alkali', dt:.531, d0:-.484, delta:1.016, mpre:-1.770, mobs:-2.039, pai:.270, matched:-1.789, timing:.020},
    rev:{truth:'antonio', dt:-.547, d0:.469, delta:-1.016, mpre:-1.742, mobs:-1.659, pai:.083, matched:-1.795, timing:.053}
  },
  tost:{
    evolved:{mean:.0033, ci90:[-.0222,.0288], ci95:[-.0285,.0351], p:8.99e-5},
    matched:{mean:.0348, ci90:[-.0002,.0698], ci95:[-.0089,.0785], p:.0048}
  },
  tiers:[
    {name:'Strict-C', n:2, detail:'quartz_basalt FWD & REV', tone:'acid'},
    {name:'Boundary / weak', n:3, detail:'marble_quartz FWD; basalt_granite REV; amber_garnet REV', tone:'amber'},
    {name:'Clear same-choice controls', n:11, detail:'continuous causal shifts without discrete target/observer disagreement', tone:'cyan'}
  ],
  ladder:[
    {n:'01', left:'Hidden', right:'Privileged', answer:'NO', detail:'S10: deterministic public-history replay reconstructs hidden state.'},
    {n:'02', left:'Persistent', right:'Reportable', answer:'NO', detail:'S11b: RG-LRU separation survives after direct local residency and factual retrieval resolve apart.'},
    {n:'03', left:'Different', right:'Causal', answer:'NO', detail:'S12: only surgical transplantation establishes leverage.'},
    {n:'04', left:'Causal', right:'Specific', answer:'NO', detail:'S12c: matching history adds a selective increment over wrong-value and cross-history states.'},
    {n:'05', left:'Specific', right:'Coordinate-stable', answer:'NO', detail:'S13: the historical axis dissolves while contemporaneous steerability survives.'},
    {n:'06', left:'State-sensitive report', right:'Generic read head', answer:'NO', detail:'S14: report modulation is local to the strict-C cell, not globally coupled to arbitrary perturbations.'},
    {n:'07', left:'State-conditioned reporting', right:'Historical provenance', answer:'NO', detail:'S14: matched POST reproduces the report without prior decision participation.'}
  ]
};
