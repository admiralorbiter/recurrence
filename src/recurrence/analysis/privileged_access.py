"""Privileged Access Index (PAI) and Observer calibration analysis."""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from recurrence.analysis.calibration import compute_post_decision_discrimination_from_pairs


def compute_brier_score_from_predictions(predictions: List[Tuple[Optional[bool], bool]]) -> Optional[float]:
    """Compute Brier score on binary forecast predictions vs actual correctness.
    
    predictions: List of (predicted_correct_bool, actual_correct_bool)
    """
    valid = [p for p in predictions if p[0] is not None]
    if not valid:
        return None
    squared_errors = [((1.0 if pred else 0.0) - (1.0 if actual else 0.0)) ** 2 for pred, actual in valid]
    return float(np.mean(squared_errors))


def compute_privileged_access_index(
    self_pairs: List[Tuple[Optional[int], bool]],
    observer_dict: Dict[str, List[Tuple[Optional[int], bool]]],
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute Privileged Access Index (PAI) with paired bootstrap confidence intervals.
    
    PAI = AUROC2_self - max(AUROC2_visible, AUROC2_reconstruction)
    
    Parameters
    ----------
    self_pairs : List[Tuple[Optional[int], bool]]
        Paired (confidence, correct) for the target self-report.
    observer_dict : Dict[str, List[Tuple[Optional[int], bool]]]
        Mapping from observer name to paired (confidence, correct).
    n_bootstraps : int
        Number of paired bootstrap resamples.
    seed : int
        Random seed for reproducibility.
    """
    rng = np.random.RandomState(seed)
    n = len(self_pairs)

    self_disc = compute_post_decision_discrimination_from_pairs(self_pairs)
    self_auroc = self_disc["auroc2"] if self_disc["auroc2"] is not None else 0.5

    obs_disc_results: Dict[str, Dict[str, Any]] = {}
    obs_aurocs: Dict[str, float] = {}

    for obs_name, pairs in observer_dict.items():
        disc = compute_post_decision_discrimination_from_pairs(pairs)
        obs_disc_results[obs_name] = disc
        obs_aurocs[obs_name] = disc["auroc2"] if disc["auroc2"] is not None else 0.5

    # Benchmark observers for PAI are visible evidence and reconstruction
    benchmark_aurocs = [
        obs_aurocs[k] for k in ["observer_visible", "observer_reconstruction"]
        if k in obs_aurocs
    ]
    max_benchmark_auroc = max(benchmark_aurocs) if benchmark_aurocs else 0.5
    point_pai = float(self_auroc - max_benchmark_auroc)

    # Paired Bootstrap CI calculation
    bootstrap_pais: List[float] = []
    if n > 0 and benchmark_aurocs:
        for _ in range(n_bootstraps):
            indices = rng.choice(n, size=n, replace=True)
            boot_self = [self_pairs[i] for i in indices]
            boot_self_disc = compute_post_decision_discrimination_from_pairs(boot_self)
            boot_self_auroc = boot_self_disc["auroc2"] if boot_self_disc["auroc2"] is not None else 0.5

            boot_obs_aurocs = []
            for obs_name in ["observer_visible", "observer_reconstruction"]:
                if obs_name in observer_dict:
                    boot_pairs = [observer_dict[obs_name][i] for i in indices]
                    boot_obs_disc = compute_post_decision_discrimination_from_pairs(boot_pairs)
                    val = boot_obs_disc["auroc2"] if boot_obs_disc["auroc2"] is not None else 0.5
                    boot_obs_aurocs.append(val)

            boot_max_obs = max(boot_obs_aurocs) if boot_obs_aurocs else 0.5
            bootstrap_pais.append(boot_self_auroc - boot_max_obs)

    if bootstrap_pais:
        ci_lower = float(np.percentile(bootstrap_pais, 2.5))
        ci_upper = float(np.percentile(bootstrap_pais, 97.5))
        # Two-sided empirical p-value for H0: PAI == 0
        p_val = float(np.mean([np.abs(b) >= np.abs(point_pai) for b in bootstrap_pais]))
    else:
        ci_lower = point_pai
        ci_upper = point_pai
        p_val = 1.0

    return {
        "point_pai": point_pai,
        "self_auroc2": self_auroc,
        "max_benchmark_observer_auroc2": max_benchmark_auroc,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "bootstrap_p_value": p_val,
        "n_bootstraps": n_bootstraps,
        "self_discrimination": self_disc,
        "observer_discrimination": obs_disc_results,
    }
