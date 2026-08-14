"""Privileged Access Index (PAI) and Strict Item-Paired Intersection Analysis."""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from recurrence.analysis.calibration import (
    compute_auroc2,
    compute_post_decision_discrimination_from_pairs,
)


def compute_continuous_brier_score(predictions: List[Tuple[Optional[float], bool]]) -> Optional[float]:
    """Compute true probabilistic Brier score on continuous probability forecasts vs binary ground truth.
    
    predictions: List of (predicted_prob_float in [0.0, 1.0], actual_correct_bool)
    Brier Score = (1/N) * sum((p_i - y_i)^2)
    """
    valid = [p for p in predictions if p[0] is not None]
    if not valid:
        return None
    squared_errors = [((float(p[0])) - (1.0 if p[1] else 0.0)) ** 2 for p in valid]
    return float(np.mean(squared_errors))


def compute_item_paired_contrasts(
    self_item_map: Dict[str, Tuple[Optional[float], bool]],
    observer_item_maps: Dict[str, Dict[str, Tuple[Optional[float], bool]]],
    sesoi: float = 0.10,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute strict item-paired discrimination contrasts on exact pairwise intersection subsets.
    
    Parameters
    ----------
    self_item_map : Dict[item_id, (self_prob, actual_correct)]
    observer_item_maps : Dict[observer_name, Dict[item_id, (obs_prob, actual_correct)]]
    sesoi : float
        Smallest Effect Size of Interest for equivalence margin (default +/- 0.10 AUROC2).
    n_bootstraps : int
        Number of paired bootstrap resamples.
    seed : int
        Random seed for reproducibility.
    """
    rng = np.random.RandomState(seed)
    contrasts: Dict[str, Any] = {}

    for obs_name, obs_map in observer_item_maps.items():
        # Form exact pairwise intersection subset
        shared_keys = [
            k for k in self_item_map
            if k in obs_map and self_item_map[k][0] is not None and obs_map[k][0] is not None
        ]
        n_shared = len(shared_keys)
        if n_shared == 0:
            contrasts[obs_name] = {
                "shared_items_count": 0,
                "error": "No shared valid items between self and observer",
            }
            continue

        self_probs = [float(self_item_map[k][0]) for k in shared_keys]
        obs_probs = [float(obs_map[k][0]) for k in shared_keys]
        labels = [bool(self_item_map[k][1]) for k in shared_keys]

        self_disc = compute_post_decision_discrimination_from_pairs(list(zip(self_probs, labels)))
        obs_disc = compute_post_decision_discrimination_from_pairs(list(zip(obs_probs, labels)))

        self_auroc = self_disc["auroc2"] if self_disc["auroc2"] is not None else 0.5
        obs_auroc = obs_disc["auroc2"] if obs_disc["auroc2"] is not None else 0.5
        delta_auroc = float(self_auroc - obs_auroc)

        # Continuous Brier Scores
        self_brier = compute_continuous_brier_score(list(zip(self_probs, labels)))
        obs_brier = compute_continuous_brier_score(list(zip(obs_probs, labels)))

        # Binary forecast accuracy (p >= 0.5)
        obs_pred_acc = float(np.mean([(p >= 0.5) == y for p, y in zip(obs_probs, labels)]))

        # Paired Bootstrap CI over intersection subset
        boot_deltas: List[float] = []
        for _ in range(n_bootstraps):
            boot_idx = rng.choice(n_shared, size=n_shared, replace=True)
            b_self_p = [self_probs[i] for i in boot_idx]
            b_obs_p = [obs_probs[i] for i in boot_idx]
            b_labels = [labels[i] for i in boot_idx]

            b_self_auc = compute_auroc2(b_self_p, b_labels)
            b_obs_auc = compute_auroc2(b_obs_p, b_labels)

            b_self_val = b_self_auc if b_self_auc is not None else 0.5
            b_obs_val = b_obs_auc if b_obs_auc is not None else 0.5
            boot_deltas.append(b_self_val - b_obs_val)

        ci_lower = float(np.percentile(boot_deltas, 2.5))
        ci_upper = float(np.percentile(boot_deltas, 97.5))

        # Equivalence evaluation relative to SESOI
        # Equivalence holds if the 95% CI is entirely within [-SESOI, +SESOI]
        equivalent_within_sesoi = bool(ci_lower >= -sesoi and ci_upper <= sesoi)
        no_positive_advantage = bool(ci_upper <= sesoi)

        contrasts[obs_name] = {
            "shared_items_count": n_shared,
            "self_auroc2": self_auroc,
            "observer_auroc2": obs_auroc,
            "delta_auroc2_self_minus_obs": delta_auroc,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "sesoi_margin": sesoi,
            "equivalent_within_sesoi": equivalent_within_sesoi,
            "no_positive_advantage_detected": no_positive_advantage,
            "self_brier_score": self_brier,
            "observer_brier_score": obs_brier,
            "observer_binary_accuracy": obs_pred_acc,
            "self_discrimination": self_disc,
            "observer_discrimination": obs_disc,
        }

    # Compute Joint Intersection PAI (across Visible-Answer-Only and Reconstruction)
    benchmark_names = [k for k in ["observer_visible_answer_only", "observer_reconstruction"] if k in observer_item_maps]
    joint_keys = [
        k for k in self_item_map
        if self_item_map[k][0] is not None and all(
            k in observer_item_maps[b] and observer_item_maps[b][k][0] is not None
            for b in benchmark_names
        )
    ]

    joint_pai_summary: Dict[str, Any] = {}
    if joint_keys and benchmark_names:
        n_joint = len(joint_keys)
        j_self_p = [float(self_item_map[k][0]) for k in joint_keys]
        j_labels = [bool(self_item_map[k][1]) for k in joint_keys]
        j_self_auc = compute_auroc2(j_self_p, j_labels) or 0.5

        j_obs_aucs = []
        for b in benchmark_names:
            b_p = [float(observer_item_maps[b][k][0]) for k in joint_keys]
            b_auc = compute_auroc2(b_p, j_labels) or 0.5
            j_obs_aucs.append(b_auc)

        max_bench_auc = max(j_obs_aucs) if j_obs_aucs else 0.5
        point_pai = float(j_self_auc - max_bench_auc)

        # Joint Bootstrap
        joint_boot_pais: List[float] = []
        for _ in range(n_bootstraps):
            boot_idx = rng.choice(n_joint, size=n_joint, replace=True)
            b_self_p = [j_self_p[i] for i in boot_idx]
            b_labels = [j_labels[i] for i in boot_idx]
            b_self_auc = compute_auroc2(b_self_p, b_labels) or 0.5

            b_obs_aucs = []
            for b in benchmark_names:
                b_p = [float(observer_item_maps[b][joint_keys[i]][0]) for i in boot_idx]
                b_auc = compute_auroc2(b_p, b_labels) or 0.5
                b_obs_aucs.append(b_auc)

            b_max = max(b_obs_aucs) if b_obs_aucs else 0.5
            joint_boot_pais.append(b_self_auc - b_max)

        joint_pai_summary = {
            "joint_shared_items_count": n_joint,
            "point_pai": point_pai,
            "self_auroc2": j_self_auc,
            "max_benchmark_observer_auroc2": max_bench_auc,
            "ci_95_lower": float(np.percentile(joint_boot_pais, 2.5)),
            "ci_95_upper": float(np.percentile(joint_boot_pais, 97.5)),
            "sesoi_margin": sesoi,
        }

    return {
        "contrasts": contrasts,
        "joint_pai_summary": joint_pai_summary,
    }


# Backwards compatibility helper
def compute_privileged_access_index(
    self_pairs: List[Tuple[Optional[int], bool]],
    observer_dict: Dict[str, List[Tuple[Optional[int], bool]]],
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compatibility wrapper converting paired lists to item maps."""
    self_map = {f"item_{i}": (float(c / 5.0) if c is not None else None, y) for i, (c, y) in enumerate(self_pairs)}
    obs_maps = {}
    for name, pairs in observer_dict.items():
        obs_maps[name] = {f"item_{i}": (float(c / 5.0) if c is not None else None, y) for i, (c, y) in enumerate(pairs)}
    
    res = compute_item_paired_contrasts(self_map, obs_maps, n_bootstraps=n_bootstraps, seed=seed)
    joint = res.get("joint_pai_summary", {})
    return {
        "point_pai": joint.get("point_pai", 0.0),
        "self_auroc2": joint.get("self_auroc2", 0.5),
        "max_benchmark_observer_auroc2": joint.get("max_benchmark_observer_auroc2", 0.5),
        "ci_95_lower": joint.get("ci_95_lower", 0.0),
        "ci_95_upper": joint.get("ci_95_upper", 0.0),
        "contrasts": res.get("contrasts", {}),
    }
