"""Maximum Likelihood Estimation of Type-2 Sensitivity (meta-d') and M-ratio.

Implements the Maniscalco & Lau (2012, 2014) signal detection theory model for
Type-2 sensitivity (meta-d') in 2-Alternative Forced Choice (2AFC) tasks.

References:
- Maniscalco, B., & Lau, H. (2012). A signal detection theoretic approach to
  measuring metacognitive sensitivity from confidence ratings. Consciousness and
  Cognition, 21(1), 422-430.
- Fleming, S. M., & Lau, H. C. (2014). How to measure metacognition. Frontiers in
  Human Neuroscience, 8, 443.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from recurrence.analysis.psychophysics import compute_sdt_indices


# Fixed preregistered confidence bin edges for probability ratings in [0, 100] or [0.0, 1.0]
# 4 ordered categories: [0, 65), [65, 80), [80, 95), [95, 100]
DEFAULT_CONFIDENCE_BINS_100 = [0.0, 65.0, 80.0, 95.0, 100.0]
DEFAULT_CONFIDENCE_BINS_UNIT = [0.0, 0.65, 0.80, 0.95, 1.00]


def discretize_confidence_ratings(
    confidences: np.ndarray,
    bin_edges: Optional[List[float]] = None,
) -> np.ndarray:
    """Discretize continuous confidence / probability ratings into ordered integer bins (1..K).
    
    Uses fixed, preregistered threshold binning covering the entire legal probability space [0, 100].
    """
    confs = np.asarray(confidences, dtype=float)
    max_val = np.nanmax(confs) if len(confs) > 0 else 1.0
    
    if bin_edges is None:
        edges = DEFAULT_CONFIDENCE_BINS_100 if max_val > 1.5 else DEFAULT_CONFIDENCE_BINS_UNIT
    else:
        edges = bin_edges

    # Assign each value into a bin 1..K (np.digitize on inner thresholds)
    # [0, 65) -> 1, [65, 80) -> 2, [80, 95) -> 3, [95, 100] -> 4
    bins = np.digitize(confs, edges[1:-1]) + 1
    k_max = len(edges) - 1
    return np.clip(bins, 1, k_max)



def build_type2_contingency_table(
    records: List[Dict[str, Any]],
    n_bins: int = 4,
    signal_target: str = "A",
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Construct the 2K response-confidence frequency vectors n_S1 and n_S2 for Maniscalco-Lau meta-d' fitting.
    
    Vector format:
    Indices 0..(K-1):   Chose Option A (R=1), from Highest Confidence (K) down to Lowest Confidence (1)
    Indices K..(2K-1):  Chose Option B (R=2), from Lowest Confidence (1) up to Highest Confidence (K)
    
    Returns:
    (n_s1, n_s2, K_actual)
    """
    valid_recs = []
    for r in records:
        gt = str(r.get("ground_truth", "")).upper()
        ans = str(r.get("parsed_answer", "")).upper()
        prob = r.get("probability")
        if gt in ["A", "B"] and ans in ["A", "B"] and prob is not None:
            valid_recs.append(r)

    if not valid_recs:
        return np.zeros(2 * n_bins), np.zeros(2 * n_bins), n_bins

    raw_confs = np.array([float(r["probability"]) for r in valid_recs])
    binned_confs = discretize_confidence_ratings(raw_confs)
    k_max = int(np.max(binned_confs)) if len(binned_confs) > 0 else n_bins
    K = max(2, k_max)

    n_s1 = np.zeros(2 * K, dtype=float)
    n_s2 = np.zeros(2 * K, dtype=float)

    for r, b_conf in zip(valid_recs, binned_confs):
        gt = str(r["ground_truth"]).upper()
        ans = str(r["parsed_answer"]).upper()
        conf_idx = int(np.clip(b_conf, 1, K))

        if ans == signal_target:  # Chose A (R=1)
            # Higher confidence -> smaller index (0 = max conf A, K-1 = min conf A)
            vec_idx = K - conf_idx
        else:  # Chose B (R=2)
            # Lower confidence -> smaller index (K = min conf B, 2K-1 = max conf B)
            vec_idx = K + (conf_idx - 1)

        if gt == signal_target:
            n_s1[vec_idx] += 1.0
        else:
            n_s2[vec_idx] += 1.0

    return n_s1, n_s2, K


def _meta_d_log_likelihood(
    params: np.ndarray,
    n_s1: np.ndarray,
    n_s2: np.ndarray,
    K: int,
    c_type1: float,
    d_type1: float,
) -> float:
    """Negative log-likelihood of Type-2 response counts given meta-d' and Type-2 criteria."""
    meta_d = params[0]
    # Parameter vector layout: [meta_d, tau_L_1..tau_L_K-1, tau_R_1..tau_R_K-1]
    # For R=1 (A): criteria tau_L are below c2
    # For R=2 (B): criteria tau_R are above c2
    tau_L = params[1:K]
    tau_R = params[K:]

    # Relative criterion preservation: c2 = c1 * (meta_d / d1) if d1 > 0 else c1
    c2 = c_type1 * (meta_d / d_type1) if abs(d_type1) > 1e-4 else c_type1

    # Check monotonicity of criteria
    # tau_L must be sorted: -inf < tau_L[0] < tau_L[1] < ... < tau_L[K-2] < c2
    # tau_R must be sorted: c2 < tau_R[0] < tau_R[1] < ... < tau_R[K-2] < +inf
    all_bounds_L = np.concatenate([[-np.inf], tau_L, [c2]])
    all_bounds_R = np.concatenate([[c2], tau_R, [np.inf]])

    if np.any(np.diff(all_bounds_L) <= 0) or np.any(np.diff(all_bounds_R) <= 0):
        return 1e10

    # Probabilities under S1 (mean = -meta_d/2, std = 1)
    mu_s1 = -meta_d / 2.0
    # Probabilities under S2 (mean = +meta_d/2, std = 1)
    mu_s2 = +meta_d / 2.0

    # Cumulative probabilities for L (R=1, indices 0..K-1)
    # Note: index 0 is highest conf A (most negative bound), index K-1 is lowest conf A (boundary near c2)
    p_s1_L = np.diff(norm.cdf(all_bounds_L - mu_s1))
    p_s2_L = np.diff(norm.cdf(all_bounds_L - mu_s2))

    # Cumulative probabilities for R (R=2, indices K..2K-1)
    p_s1_R = np.diff(norm.cdf(all_bounds_R - mu_s1))
    p_s2_R = np.diff(norm.cdf(all_bounds_R - mu_s2))

    p_s1 = np.concatenate([p_s1_L, p_s1_R])
    p_s2 = np.concatenate([p_s2_L, p_s2_R])

    # Avoid log(0)
    eps = 1e-12
    p_s1 = np.clip(p_s1, eps, 1.0 - eps)
    p_s2 = np.clip(p_s2, eps, 1.0 - eps)

    # Normalize within distribution
    p_s1 = p_s1 / np.sum(p_s1)
    p_s2 = p_s2 / np.sum(p_s2)

    log_lik = np.sum(n_s1 * np.log(p_s1)) + np.sum(n_s2 * np.log(p_s2))
    return -float(log_lik)


def fit_meta_d_mle(
    records: List[Dict[str, Any]],
    n_bins: int = 4,
    signal_target: str = "A",
) -> Dict[str, Any]:
    """Fit Type-2 meta-d' and M-ratio using Maniscalco & Lau (2012) Maximum Likelihood Estimation.
    
    Status Codes:
    - 'fit_success': Non-degenerate confidence; MLE optimization converged successfully.
    - 'confidence_degenerate': Constant confidence rating (std < 1e-4); Type-2 criteria undefined.
    - 'insufficient_class_counts': Accuracy is 0% or 100%; negative log-likelihood undefined.
    - 'insufficient_data': Fewer than 10 valid trials.
    - 'fit_failure': Numerical optimization failed to converge.
    """
    valid_recs = [
        r for r in records
        if r.get("ground_truth") in ["A", "B"]
        and r.get("parsed_answer") in ["A", "B"]
        and r.get("probability") is not None
    ]

    if len(valid_recs) < 10:
        return {
            "meta_d_status": "insufficient_data",
            "meta_d_prime": None,
            "m_ratio": None,
            "type1_d_prime": None,
            "type1_criterion_c": None,
            "meta_criterion_c2": None,
            "log_likelihood": None,
        }

    confidences = np.array([float(r["probability"]) for r in valid_recs])
    corrects = np.array([1 if r.get("correct", False) else 0 for r in valid_recs])
    n_corr = int(np.sum(corrects))
    n_inc = len(corrects) - n_corr

    if n_corr == 0 or n_inc == 0:
        return {
            "meta_d_status": "insufficient_class_counts",
            "meta_d_prime": None,
            "m_ratio": None,
            "type1_d_prime": None,
            "type1_criterion_c": None,
            "meta_criterion_c2": None,
            "log_likelihood": None,
        }

    # Detect degenerate confidence
    unique_confs = np.unique(confidences)
    if len(unique_confs) <= 1 or np.std(confidences) < 1e-4:
        return {
            "meta_d_status": "confidence_degenerate",
            "meta_d_prime": None,
            "m_ratio": None,
            "type1_d_prime": None,
            "type1_criterion_c": None,
            "meta_criterion_c2": None,
            "log_likelihood": None,
        }

    # Type-1 SDT parameters
    sdt_t1 = compute_sdt_indices(valid_recs, signal_target=signal_target)
    d1 = sdt_t1.get("d_prime")
    c1 = sdt_t1.get("criterion_c")

    if d1 is None or c1 is None:
        return {
            "meta_d_status": "fit_failure",
            "meta_d_prime": None,
            "m_ratio": None,
            "type1_d_prime": None,
            "type1_criterion_c": None,
            "meta_criterion_c2": None,
            "log_likelihood": None,
        }

    # Build contingency counts
    n_s1, n_s2, K = build_type2_contingency_table(valid_recs, n_bins=n_bins, signal_target=signal_target)

    # Initial parameter guess
    init_meta_d = max(0.1, float(d1))
    init_tau_L = np.linspace(c1 - 1.5, c1 - 0.2, K - 1)
    init_tau_R = np.linspace(c1 + 0.2, c1 + 1.5, K - 1)
    init_params = np.concatenate([[init_meta_d], init_tau_L, init_tau_R])

    # Bounds: meta_d >= 0
    bounds = [(0.0, 10.0)] + [(None, None)] * (2 * (K - 1))

    try:
        res = minimize(
            _meta_d_log_likelihood,
            init_params,
            args=(n_s1, n_s2, K, c1, d1),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-8},
        )

        if not res.success or res.fun >= 1e9:
            # Fallback to Nelder-Mead
            res = minimize(
                _meta_d_log_likelihood,
                init_params,
                args=(n_s1, n_s2, K, c1, d1),
                method="Nelder-Mead",
                options={"maxiter": 2000},
            )

        if not res.success or res.fun >= 1e9:
            return {
                "meta_d_status": "fit_failure",
                "meta_d_prime": None,
                "m_ratio": None,
                "type1_d_prime": d1,
                "type1_criterion_c": c1,
                "meta_criterion_c2": None,
                "log_likelihood": None,
            }

        fitted_meta_d = float(max(0.0, res.x[0]))
        c2 = c1 * (fitted_meta_d / d1) if abs(d1) > 1e-4 else c1
        m_ratio = float(fitted_meta_d / d1) if (d1 and d1 > 0.05) else None

        return {
            "meta_d_status": "fit_success",
            "meta_d_prime": fitted_meta_d,
            "m_ratio": m_ratio,
            "type1_d_prime": d1,
            "type1_criterion_c": c1,
            "meta_criterion_c2": float(c2),
            "log_likelihood": float(-res.fun),
            "k_confidence_bins": K,
        }

    except Exception:
        return {
            "meta_d_status": "fit_failure",
            "meta_d_prime": None,
            "m_ratio": None,
            "type1_d_prime": d1,
            "type1_criterion_c": c1,
            "meta_criterion_c2": None,
            "log_likelihood": None,
        }
