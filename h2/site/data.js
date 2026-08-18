window.H2_DATA = {
  "meta": {
    "title": "The Memory That Couldn't Remember",
    "subtitle": "A journey beyond the attention window",
    "model": "google/recurrentgemma-2b",
    "attention_window": 2048,
    "conv_width": 4,
    "frozen_core": ["S10", "S11b", "S12b"],
    "live_extension": "S12c",
    "next_major_sprint": "S13"
  },
  "architecture": [
    {
      "id": "conv",
      "label": "CONV",
      "name": "Conv1D buffer",
      "capacity": "4-token local buffer",
      "direct_residency_ends_at": 3,
      "status_at_2w": "no direct residency"
    },
    {
      "id": "kv",
      "label": "KV CACHE",
      "name": "Sliding attention KV cache",
      "capacity": "2048-token local window",
      "direct_residency_ends_at": 2047,
      "status_at_2w": "no direct residency"
    },
    {
      "id": "rglru",
      "label": "RG-LRU",
      "name": "Real-Gated Linear Recurrent Unit",
      "capacity": "continuous recurrent state",
      "direct_residency_ends_at": null,
      "status_at_2w": "branch-specific trace remains"
    }
  ],
  "s10": {
    "status": "FROZEN",
    "claim": "Under deterministic execution, hidden recurrent state is exactly reconstructible from the same public token history.",
    "guardrail": "Hidden does not imply informationally privileged.",
    "source": "docs/H2_Core_Retrospective_Memo.md"
  },
  "s11": {
    "status": "FROZEN",
    "lags": [0, 1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2040, 2047, 2048, 2049, 4096],
    "constant": {
      "rglru_retention": [1.0, 0.842, 0.767, 0.692, 0.633, 0.46, 0.386, 0.307, 0.292, 0.254, 0.234, 0.196, 0.194, 0.291, 0.286, 0.285, 0.285, 0.34],
      "kv_retention": [1.0, 0.962, 0.925, 0.885, 0.849, 0.748, 0.623, 0.494, 0.399, 0.309, 0.244, 0.196, 0.153, 0.124, 0.106, 0.105, 0.105, 0.086],
      "cloze_margin": [10.78, 11.03, 11.14, 10.86, 10.47, 12.26, 11.11, 7.85, 3.73, 4.96, 0.54, 1.85, 0.62, 0.5, 0.47, 0.52, 0.52, -0.03],
      "cloze_accuracy": [1.0, 1.0, 0.97, 1.0, 1.0, 1.0, 0.95, 0.95, 0.82, 0.9, 0.55, 0.72, 0.55, 0.57, 0.5, 0.5, 0.5, 0.5]
    },
    "two_window_regimes": [
      {
        "id": "constant",
        "label": "Constant",
        "rglru_retention": 0.3384,
        "retention_ci": [0.2484, 0.4401],
        "cloze_margin": -0.0265,
        "cloze_ci": [-0.1109, 0.0656]
      },
      {
        "id": "interfering",
        "label": "Interfering",
        "rglru_retention": 0.0798,
        "retention_ci": [0.0734, 0.0864],
        "cloze_margin": 0.0044,
        "cloze_ci": [-0.075, 0.0844]
      },
      {
        "id": "natural",
        "label": "Natural prose",
        "rglru_retention": 0.0514,
        "retention_ci": [0.0461, 0.0571],
        "cloze_margin": 0.0123,
        "cloze_ci": [-0.0125, 0.0375]
      },
      {
        "id": "random",
        "label": "Random tokens",
        "rglru_retention": 0.0453,
        "retention_ci": [0.0402, 0.0501],
        "cloze_margin": -0.003,
        "cloze_ci": [-0.0203, 0.0125]
      }
    ],
    "source": "docs/S11_Latent_Impulse_Retention_Report.md"
  },
  "s12": {
    "status": "FROZEN",
    "conditions": [
      {
        "id": "match",
        "label": "Matching RG-LRU",
        "short": "RIGHT PAST",
        "estimate": 74.0994,
        "ci": [46.7899, 106.7161]
      },
      {
        "id": "unrelated",
        "label": "Unrelated structured donor",
        "short": "OTHER PAST +1",
        "estimate": 54.4236,
        "ci": [32.2609, 77.1805]
      },
      {
        "id": "permuted",
        "label": "Permuted structured donor",
        "short": "OTHER PAST +7",
        "estimate": 44.459,
        "ci": [32.0241, 57.583]
      },
      {
        "id": "noise",
        "label": "Matched Frobenius noise",
        "short": "MATCHED NOISE",
        "estimate": 17.6393,
        "ci": [10.7672, 25.4122]
      },
      {
        "id": "kv",
        "label": "KV state swap",
        "short": "KV",
        "estimate": 62.4483,
        "ci": [54.7684, 69.7399]
      },
      {
        "id": "whole",
        "label": "Whole state swap",
        "short": "WHOLE STATE",
        "estimate": 136.5477,
        "ci": [111.7998, 165.1752]
      }
    ],
    "specificity": {
      "matching_minus_unrelated": 19.6759,
      "matching_minus_unrelated_ci": [1.8384, 39.1219],
      "matching_minus_permuted": 29.6404,
      "matching_minus_permuted_ci": [11.8234, 52.4697],
      "matching_minus_noise": 56.4601,
      "matching_minus_noise_ci": [29.449, 89.4735]
    },
    "store_race": {
      "alpha_kv": 0.632,
      "alpha_rglru": 0.368,
      "absolute_kv_minus_rglru": -11.6512,
      "absolute_ci": [-49.0193, 20.1108],
      "inference": "UNRESOLVED"
    },
    "growth": {
      "p_rglru_l8": 18.2964,
      "p_rglru_w_plus_1": 21.4408,
      "p_rglru_2w": 74.0994,
      "delta_2w_minus_w_plus_1": 52.6587,
      "delta_ci": [26.6603, 83.781]
    },
    "mediation": [
      {
        "tokens_after_graft": 512,
        "m_post": -0.451,
        "interpretation": "strongly recipient-anchored"
      },
      {
        "tokens_after_graft": 2048,
        "m_post": -0.23,
        "interpretation": "less recipient-anchored; grand mean still recipient-side"
      }
    ],
    "source": "docs/S12_Surgical_Store_Swaps_Report.md"
  },
  "frontier": {
    "s12c": {
      "status": "LIVE",
      "title": "Specificity Microscope",
      "question": "How much of cross-history steering is value-specific binding versus shared template/event geometry?",
      "implemented_panel": "24 value pairs × 4 filler regimes × 14 intervention conditions",
      "results": "not frozen in the core evidence contract"
    },
    "s13": {
      "status": "OPEN QUESTION",
      "title": "Null-Observation / Controlled Recurrent Dynamics",
      "question": "What happens to surviving recurrent history when no new task-relevant semantic information enters?"
    }
  },
  "taxonomy": [
    {
      "property": "Reconstructible",
      "answer": "YES",
      "status": "FROZEN",
      "note": "Public-history replay reconstructs deterministic hidden state."
    },
    {
      "property": "Persistent",
      "answer": "YES",
      "status": "FROZEN",
      "note": "Branch-specific RG-LRU traces remain at 2W."
    },
    {
      "property": "Causally operative",
      "answer": "YES",
      "status": "FROZEN",
      "note": "Matching RG-LRU transplantation causally steers downstream logits."
    },
    {
      "property": "Specific",
      "answer": "PARTIALLY",
      "status": "LIVE",
      "note": "Matching history adds a selective increment, but structured cross-history donors also steer substantially."
    },
    {
      "property": "Owned / privileged",
      "answer": "UNKNOWN",
      "status": "OPEN QUESTION",
      "note": "The frozen core does not establish introspective access or source ownership."
    }
  ]
};
