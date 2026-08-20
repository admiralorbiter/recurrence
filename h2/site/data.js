window.H2 = {
  "timeline": [
    {
      "n": 0,
      "label": "Origin",
      "conv": true,
      "kv": true,
      "r": 1.0,
      "recall": 1.0,
      "note": "The event is fresh. All stores carry direct traces."
    },
    {
      "n": 3,
      "label": "Conv turnover",
      "conv": false,
      "kv": true,
      "r": 0.692,
      "recall": 1.0,
      "note": "The four-token convolution buffer no longer directly contains the event."
    },
    {
      "n": 256,
      "label": "256",
      "conv": false,
      "kv": true,
      "r": 0.234,
      "recall": 0.55,
      "note": "Behavioral retrieval is already deteriorating while recurrent separation remains measurable."
    },
    {
      "n": 1024,
      "label": "1024",
      "conv": false,
      "kv": true,
      "r": 0.194,
      "recall": 0.55,
      "note": "The recurrent difference survives continued processing."
    },
    {
      "n": 2048,
      "label": "1W",
      "conv": false,
      "kv": false,
      "r": 0.285,
      "recall": 0.5,
      "note": "The original event has turned over from the local attention window."
    },
    {
      "n": 4096,
      "label": "2W",
      "conv": false,
      "kv": false,
      "r": 0.3384,
      "recall": 0.5,
      "note": "At 2W, branch-specific RG-LRU separation is resolved while the paired cloze probe is unresolved."
    }
  ],
  "transplant": [
    {
      "id": "match",
      "label": "Matching RG-LRU",
      "short": "MATCHING",
      "estimate": 74.0994,
      "ci": [
        46.7899,
        106.7161
      ],
      "copy": "Matching recurrent history strongly steers the recipient toward the donor trajectory."
    },
    {
      "id": "unrelated",
      "label": "Unrelated structured donor",
      "short": "UNRELATED",
      "estimate": 54.4236,
      "ci": [
        32.2609,
        77.1805
      ],
      "copy": "Unrelated structured states also exert a large shared steering component."
    },
    {
      "id": "permuted",
      "label": "Permuted structured donor",
      "short": "PERMUTED",
      "estimate": 44.459,
      "ci": [
        32.0241,
        57.583
      ],
      "copy": "Permuted donor states retain moderate causal leverage."
    },
    {
      "id": "noise",
      "label": "Matched Frobenius noise",
      "short": "NOISE",
      "estimate": 17.6393,
      "ci": [
        10.7672,
        25.4122
      ],
      "copy": "Matched noise perturbs the system, but structured states are far more effective."
    },
    {
      "id": "kv",
      "label": "KV state swap",
      "short": "KV SWAP",
      "estimate": 62.4483,
      "ci": [
        54.7684,
        69.7399
      ],
      "copy": "Transplanting sliding KV cache steers local generation."
    },
    {
      "id": "whole",
      "label": "Whole state swap",
      "short": "WHOLE STATE",
      "estimate": 136.5477,
      "ci": [
        111.7998,
        165.1752
      ],
      "copy": "Whole-state transplantation is the positive-control ceiling for S12b."
    }
  ],
  "specificity": {
    "value": 38.4939,
    "ci": [
      25.818,
      50.8524
    ],
    "projection": 0.1744,
    "projectionCI": [
      0.1001,
      0.2536
    ]
  },
  "dynamics": [
    {
      "n": 0,
      "v0": 39.58,
      "vn": 39.58,
      "cr": 1.0,
      "qr": 1.0,
      "angle": 0
    },
    {
      "n": 16,
      "v0": 4.85,
      "vn": 11.44,
      "cr": 0.6092,
      "qr": 1.1,
      "angle": 22
    },
    {
      "n": 64,
      "v0": -2.96,
      "vn": 31.96,
      "cr": 0.4843,
      "qr": 1.1,
      "angle": 39
    },
    {
      "n": 256,
      "v0": 1.45,
      "vn": 7.21,
      "cr": 0.3306,
      "qr": 1.24,
      "angle": 58
    },
    {
      "n": 1024,
      "v0": -16.19,
      "vn": 9.57,
      "cr": 0.166,
      "qr": 1.79,
      "angle": 76
    },
    {
      "n": 2048,
      "v0": 4.7,
      "vn": 13.95,
      "cr": 0.1238,
      "qr": 4.85,
      "angle": 83
    }
  ],
  "strictC": {
    "fwd": {
      "truth": "alkali",
      "dt": 0.531,
      "d0": -0.484,
      "delta": 1.016,
      "mpre": -1.77,
      "mobs": -2.039,
      "pai": 0.27,
      "matched": -1.789,
      "timing": 0.02
    },
    "rev": {
      "truth": "antonio",
      "dt": -0.547,
      "d0": 0.469,
      "delta": -1.016,
      "mpre": -1.742,
      "mobs": -1.659,
      "pai": 0.083,
      "matched": -1.795,
      "timing": 0.053
    }
  },
  "tost": {
    "evolved": {
      "mean": 0.0033,
      "ci90": [
        -0.0222,
        0.0288
      ],
      "ci95": [
        -0.0285,
        0.0351
      ],
      "p": 8.99e-05
    },
    "matched": {
      "mean": 0.0348,
      "ci90": [
        -0.0002,
        0.0698
      ],
      "ci95": [
        -0.0089,
        0.0785
      ],
      "p": 0.0048
    }
  },
  "tiers": [
    {
      "name": "Strict-C",
      "n": 2,
      "detail": "quartz_basalt FWD & REV",
      "tone": "acid"
    },
    {
      "name": "Boundary / weak",
      "n": 3,
      "detail": "marble_quartz FWD; basalt_granite REV; amber_garnet REV",
      "tone": "amber"
    },
    {
      "name": "Clear same-choice controls",
      "n": 11,
      "detail": "continuous causal shifts without discrete target/observer disagreement",
      "tone": "cyan"
    }
  ],
  "ladder": [
    {
      "n": "01",
      "left": "Hidden",
      "right": "Privileged",
      "answer": "NO",
      "detail": "S10: deterministic public-history replay reconstructs hidden state."
    },
    {
      "n": "02",
      "left": "Persistent",
      "right": "Reportable",
      "answer": "NO",
      "detail": "S11b: RG-LRU separation survives after direct local residency and factual retrieval resolve apart."
    },
    {
      "n": "03",
      "left": "Different",
      "right": "Causal",
      "answer": "NO",
      "detail": "S12b: only surgical transplantation establishes causal leverage."
    },
    {
      "n": "04",
      "left": "Causal",
      "right": "Specific",
      "answer": "NO",
      "detail": "S12c: matching history adds a selective increment over wrong-value and cross-history states."
    },
    {
      "n": "05",
      "left": "Specific",
      "right": "Coordinate-stable",
      "answer": "NO",
      "detail": "S13: the historical axis dissolves while contemporaneous steerability survives."
    },
    {
      "n": "06",
      "left": "State-sensitive report",
      "right": "Generic read head",
      "answer": "NO",
      "detail": "S14: report modulation is local to the strict-C cell, not globally coupled to arbitrary perturbations."
    },
    {
      "n": "07",
      "left": "State-conditioned reporting",
      "right": "Historical provenance",
      "answer": "NO",
      "detail": "S14: matched POST reproduces the report without prior decision participation."
    }
  ]
};
