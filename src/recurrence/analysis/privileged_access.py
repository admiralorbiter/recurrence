"""Privileged Access Index (PAI) and Strict Item-Paired Intersection Analysis with Stratified Bootstrap."""

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


def _stratified_paired_bootstrap_indices(labels: List[bool], rng: np.random.RandomState) -> np.ndarray:
    """Generate bootstrap sample indices stratified by label to preserve positive/negative proportions."""
    arr_labels = np.array(labels, dtype=bool)
    pos_idx = np.where(arr_labels)[0]
    neg_idx = np.where(~arr_labels)[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return rng.choice(len(labels), size=len(labels), replace=True)

    boot_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
    boot_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
    return np.concatenate([boot_pos, boot_neg])


def compute_direct_pairwise_contrast(
    map_a: Dict[str, Tuple[Optional[float], bool]],
    map_b: Dict[str, Tuple[Optional[float], bool]],
    name_a: str = "evaluator_a",
    name_b: str = "evaluator_b",
    sesoi: float = 0.10,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute direct pairwise contrast between any two evaluators on their exact shared item intersection.
    
    Uses stratified paired bootstrap. When labels contain only a single class (all correct or all incorrect),
    Type-2 AUROC2 is mathematically non-identifiable and returns None.
    """
    rng = np.random.RandomState(seed)

    shared_keys = [
        k for k in map_a
        if k in map_b and map_a[k][0] is not None and map_b[k][0] is not None
    ]
    n_shared = len(shared_keys)
    if n_shared == 0:
        return {
            "name_a": name_a,
            "name_b": name_b,
            "shared_items_count": 0,
            "error": f"No shared valid items between {name_a} and {name_b}",
            "status": "no_shared_items",
        }

    probs_a = [float(map_a[k][0]) for k in shared_keys]
    probs_b = [float(map_b[k][0]) for k in shared_keys]
    labels = [bool(map_a[k][1]) for k in shared_keys]

    disc_a = compute_post_decision_discrimination_from_pairs(list(zip(probs_a, labels)))
    disc_b = compute_post_decision_discrimination_from_pairs(list(zip(probs_b, labels)))

    auc_a = disc_a["auroc2"]
    auc_b = disc_b["auroc2"]

    brier_a = compute_continuous_brier_score(list(zip(probs_a, labels)))
    brier_b = compute_continuous_brier_score(list(zip(probs_b, labels)))
    delta_brier = float(brier_a - brier_b) if brier_a is not None and brier_b is not None else None

    # Forecast classification accuracy (thresholded at p >= 0.5)
    acc_a = float(np.mean([(p >= 0.5) == y for p, y in zip(probs_a, labels)]))
    acc_b = float(np.mean([(p >= 0.5) == y for p, y in zip(probs_b, labels)]))

    if auc_a is None or auc_b is None:
        return {
            "name_a": name_a,
            "name_b": name_b,
            "shared_items_count": n_shared,
            "auroc2_a": auc_a,
            "auroc2_b": auc_b,
            "delta_auroc2": None,
            "ci_95_lower": None,
            "ci_95_upper": None,
            "sesoi_margin": sesoi,
            "brier_score_a": brier_a,
            "brier_score_b": brier_b,
            "delta_brier_score": delta_brier,
            "forecast_classification_accuracy_a": acc_a,
            "forecast_classification_accuracy_b": acc_b,
            "binary_accuracy_a": acc_a,
            "binary_accuracy_b": acc_b,
            "discrimination_a": disc_a,
            "discrimination_b": disc_b,
            "status": "undefined_single_class",
        }

    delta_auc = float(auc_a - auc_b)

    # Stratified paired bootstrap for AUROC difference
    boot_deltas: List[float] = []
    for _ in range(n_bootstraps):
        boot_idx = _stratified_paired_bootstrap_indices(labels, rng)
        b_p_a = [probs_a[i] for i in boot_idx]
        b_p_b = [probs_b[i] for i in boot_idx]
        b_lbls = [labels[i] for i in boot_idx]

        b_auc_a = compute_auroc2(b_p_a, b_lbls)
        b_auc_b = compute_auroc2(b_p_b, b_lbls)

        if b_auc_a is not None and b_auc_b is not None:
            boot_deltas.append(b_auc_a - b_auc_b)

    if boot_deltas:
        ci_lower = float(np.percentile(boot_deltas, 2.5))
        ci_upper = float(np.percentile(boot_deltas, 97.5))
    else:
        ci_lower, ci_upper = None, None

    return {
        "name_a": name_a,
        "name_b": name_b,
        "shared_items_count": n_shared,
        "auroc2_a": auc_a,
        "auroc2_b": auc_b,
        "delta_auroc2": delta_auc,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "sesoi_margin": sesoi,
        "brier_score_a": brier_a,
        "brier_score_b": brier_b,
        "delta_brier_score": delta_brier,
        "forecast_classification_accuracy_a": acc_a,
        "forecast_classification_accuracy_b": acc_b,
        "binary_accuracy_a": acc_a,
        "binary_accuracy_b": acc_b,
        "discrimination_a": disc_a,
        "discrimination_b": disc_b,
        "status": "valid",
    }


def compute_item_paired_contrasts(
    self_item_map: Dict[str, Tuple[Optional[float], bool]],
    observer_item_maps: Dict[str, Dict[str, Tuple[Optional[float], bool]]],
    sesoi: float = 0.10,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute strict item-paired discrimination contrasts on exact pairwise intersection subsets
    using stratified paired bootstrap. Returns None for AUROC2/PAI if single-class (e.g. 100% accuracy).
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
                "status": "no_shared_items",
            }
            continue

        self_probs = [float(self_item_map[k][0]) for k in shared_keys]
        obs_probs = [float(obs_map[k][0]) for k in shared_keys]
        labels = [bool(self_item_map[k][1]) for k in shared_keys]

        self_disc = compute_post_decision_discrimination_from_pairs(list(zip(self_probs, labels)))
        obs_disc = compute_post_decision_discrimination_from_pairs(list(zip(obs_probs, labels)))

        self_auroc = self_disc["auroc2"]
        obs_auroc = obs_disc["auroc2"]

        # Continuous Brier Scores
        self_brier = compute_continuous_brier_score(list(zip(self_probs, labels)))
        obs_brier = compute_continuous_brier_score(list(zip(obs_probs, labels)))

        # Forecast classification accuracy (thresholded at p >= 0.5)
        obs_pred_acc = float(np.mean([(p >= 0.5) == y for p, y in zip(obs_probs, labels)]))

        if self_auroc is None or obs_auroc is None:
            contrasts[obs_name] = {
                "shared_items_count": n_shared,
                "self_auroc2": self_auroc,
                "observer_auroc2": obs_auroc,
                "delta_auroc2_self_minus_obs": None,
                "ci_95_lower": None,
                "ci_95_upper": None,
                "sesoi_margin": sesoi,
                "equivalent_within_sesoi": None,
                "no_positive_advantage_detected": None,
                "self_brier_score": self_brier,
                "observer_brier_score": obs_brier,
                "forecast_classification_accuracy": obs_pred_acc,
                "observer_binary_accuracy": obs_pred_acc,
                "self_discrimination": self_disc,
                "observer_discrimination": obs_disc,
                "status": "undefined_single_class",
            }
            continue

        delta_auroc = float(self_auroc - obs_auroc)

        # Stratified Paired Bootstrap CI over intersection subset
        boot_deltas: List[float] = []
        for _ in range(n_bootstraps):
            boot_idx = _stratified_paired_bootstrap_indices(labels, rng)
            b_self_p = [self_probs[i] for i in boot_idx]
            b_obs_p = [obs_probs[i] for i in boot_idx]
            b_labels = [labels[i] for i in boot_idx]

            b_self_auc = compute_auroc2(b_self_p, b_labels)
            b_obs_auc = compute_auroc2(b_obs_p, b_labels)

            if b_self_auc is not None and b_obs_auc is not None:
                boot_deltas.append(b_self_auc - b_obs_auc)

        if boot_deltas:
            ci_lower = float(np.percentile(boot_deltas, 2.5))
            ci_upper = float(np.percentile(boot_deltas, 97.5))
            equivalent_within_sesoi = bool(ci_lower >= -sesoi and ci_upper <= sesoi)
            no_positive_advantage = bool(ci_upper <= sesoi)
        else:
            ci_lower, ci_upper = None, None
            equivalent_within_sesoi = None
            no_positive_advantage = None

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
            "forecast_classification_accuracy": obs_pred_acc,
            "observer_binary_accuracy": obs_pred_acc,
            "self_discrimination": self_disc,
            "observer_discrimination": obs_disc,
            "status": "valid",
        }

    # Compute Joint Intersection PAI (across Visible-Answer-Only, Reconstruction, and Input-Only)
    benchmark_names = [
        k for k in ["observer_visible_answer_only", "observer_reconstruction", "observer_input_only"]
        if k in observer_item_maps
    ]
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
        j_self_auc = compute_auroc2(j_self_p, j_labels)

        j_obs_aucs = []
        for b in benchmark_names:
            b_p = [float(observer_item_maps[b][k][0]) for k in joint_keys]
            b_auc = compute_auroc2(b_p, j_labels)
            j_obs_aucs.append(b_auc)

        if j_self_auc is None or any(a is None for a in j_obs_aucs):
            joint_pai_summary = {
                "joint_shared_items_count": n_joint,
                "point_pai": None,
                "self_auroc2": None,
                "max_benchmark_observer_auroc2": None,
                "ci_95_lower": None,
                "ci_95_upper": None,
                "sesoi_margin": sesoi,
                "status": "undefined_single_class",
            }
        else:
            max_bench_auc = max(j_obs_aucs)
            point_pai = float(j_self_auc - max_bench_auc)

            # Stratified Joint Bootstrap
            joint_boot_pais: List[float] = []
            for _ in range(n_bootstraps):
                boot_idx = _stratified_paired_bootstrap_indices(j_labels, rng)
                b_self_p = [j_self_p[i] for i in boot_idx]
                b_labels = [j_labels[i] for i in boot_idx]
                b_self_auc = compute_auroc2(b_self_p, b_labels)

                b_obs_aucs = []
                for b in benchmark_names:
                    b_p = [float(observer_item_maps[b][joint_keys[i]][0]) for i in boot_idx]
                    b_auc = compute_auroc2(b_p, b_labels)
                    b_obs_aucs.append(b_auc)

                if b_self_auc is not None and all(a is not None for a in b_obs_aucs):
                    b_max = max(b_obs_aucs)
                    joint_boot_pais.append(b_self_auc - b_max)

            if joint_boot_pais:
                ci_l = float(np.percentile(joint_boot_pais, 2.5))
                ci_u = float(np.percentile(joint_boot_pais, 97.5))
            else:
                ci_l, ci_u = None, None

            joint_pai_summary = {
                "joint_shared_items_count": n_joint,
                "point_pai": point_pai,
                "self_auroc2": j_self_auc,
                "max_benchmark_observer_auroc2": max_bench_auc,
                "ci_95_lower": ci_l,
                "ci_95_upper": ci_u,
                "sesoi_margin": sesoi,
                "status": "valid",
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
        "point_pai": joint.get("point_pai"),
        "self_auroc2": joint.get("self_auroc2"),
        "max_benchmark_observer_auroc2": joint.get("max_benchmark_observer_auroc2"),
        "ci_95_lower": joint.get("ci_95_lower"),
        "ci_95_upper": joint.get("ci_95_upper"),
        "contrasts": res.get("contrasts", {}),
    }
