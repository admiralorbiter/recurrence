"""Maximum Likelihood Estimation of Type-2 Sensitivity (meta-d') and M-ratio.

Implements the exact Maniscalco & Lau (2012, 2014) conditional signal detection
theory likelihood for Type-2 sensitivity (meta-d') in 2-Alternative Forced Choice (2AFC).

References:
- Maniscalco, B., & Lau, H. (2012). A signal detection theoretic approach to
  measuring metacognitive sensitivity from confidence ratings. Consciousness and
  Cognition, 21(1), 422-430.
- Fleming, S. M., & Lau, H. C. (2014). How to measure metacognition. Frontiers in
  Human Neuroscience, 8, 443.
- Maniscalco, B., & Lau, H. (2014). Signal detection theory analysis of Type 1 and
  Type 2 data: meta-d', response bias, and the unequal variance model.
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
    
    K is strictly fixed at n_bins (default: 4) regardless of whether individual bins are sparse.
    
    Vector format:
    Indices 0..(K-1):   Chose Option A (R=1), from Highest Confidence (K) down to Lowest Confidence (1)
    Indices K..(2K-1):  Chose Option B (R=2), from Lowest Confidence (1) up to Highest Confidence (K)
    
    Returns:
    (n_s1, n_s2, K)
    """
    K = int(n_bins)
    n_s1 = np.zeros(2 * K, dtype=float)
    n_s2 = np.zeros(2 * K, dtype=float)

    valid_recs = []
    for r in records:
        gt = str(r.get("ground_truth", "")).upper()
        ans = str(r.get("parsed_answer", "")).upper()
        prob = r.get("probability")
        if gt in ["A", "B"] and ans in ["A", "B"] and prob is not None:
            valid_recs.append(r)

    if not valid_recs:
        return n_s1, n_s2, K

    raw_confs = np.array([float(r["probability"]) for r in valid_recs])
    binned_confs = discretize_confidence_ratings(raw_confs)

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
    """Negative log-likelihood of Type-2 response counts conditional on Type-1 responses.
    
    Implements the exact conditional likelihood formulation of Maniscalco & Lau (2012).
    Allows signed meta-d' in [-10.0, +10.0].
    """
    meta_d = params[0]
    # Parameter vector layout: [meta_d, tau_L_1..tau_L_K-1, tau_R_1..tau_R_K-1]
    # tau_L are K-1 criteria below c2 for R=1 (sorted)
    # tau_R are K-1 criteria above c2 for R=2 (sorted)
    tau_L = params[1:K]
    tau_R = params[K:]

    # Relative criterion preservation: c2 = c1 * (meta_d / d1) if |d1| > 1e-4 else c1
    c2 = c_type1 * (meta_d / d_type1) if abs(d_type1) > 1e-4 else c_type1

    # Monotonicity checks:
    # -inf < tau_L[0] < tau_L[1] < ... < tau_L[K-2] < c2
    # c2 < tau_R[0] < tau_R[1] < ... < tau_R[K-2] < +inf
    all_bounds_L = np.concatenate([[-np.inf], tau_L, [c2]])
    all_bounds_R = np.concatenate([[c2], tau_R, [np.inf]])

    if np.any(np.diff(all_bounds_L) <= 0) or np.any(np.diff(all_bounds_R) <= 0):
        return 1e10

    # Type-1 Response Areas under (meta_d, c2)
    # Under S1 (mean = -meta_d/2, std = 1):
    # P(R=1 | S1) = Phi(c2 + meta_d/2)
    # P(R=2 | S1) = 1 - Phi(c2 + meta_d/2)
    area_R1_S1 = norm.cdf(c2 + meta_d / 2.0)
    area_R2_S1 = 1.0 - area_R1_S1

    # Under S2 (mean = +meta_d/2, std = 1):
    # P(R=1 | S2) = Phi(c2 - meta_d/2)
    # P(R=2 | S2) = 1 - Phi(c2 - meta_d/2)
    area_R1_S2 = norm.cdf(c2 - meta_d / 2.0)
    area_R2_S2 = 1.0 - area_R1_S2

    eps = 1e-12
    area_R1_S1 = max(eps, area_R1_S1)
    area_R2_S1 = max(eps, area_R2_S1)
    area_R1_S2 = max(eps, area_R1_S2)
    area_R2_S2 = max(eps, area_R2_S2)

    # Unconditional interval probabilities for R=1 (bounds L)
    prob_L_S1 = np.diff(norm.cdf(all_bounds_L + meta_d / 2.0))
    prob_L_S2 = np.diff(norm.cdf(all_bounds_L - meta_d / 2.0))

    # Unconditional interval probabilities for R=2 (bounds R)
    prob_R_S1 = np.diff(norm.cdf(all_bounds_R + meta_d / 2.0))
    prob_R_S2 = np.diff(norm.cdf(all_bounds_R - meta_d / 2.0))

    # Conditional Type-2 probabilities: P(Conf | Resp, Stimulus) = P(Conf & Resp | Stimulus) / P(Resp | Stimulus)
    cond_L_S1 = np.clip(prob_L_S1 / area_R1_S1, eps, 1.0)
    cond_L_S2 = np.clip(prob_L_S2 / area_R1_S2, eps, 1.0)
    cond_R_S1 = np.clip(prob_R_S1 / area_R2_S1, eps, 1.0)
    cond_R_S2 = np.clip(prob_R_S2 / area_R2_S2, eps, 1.0)

    # Slice observed counts
    # n_s1: indices 0..K-1 are R=1 (from conf K down to 1), indices K..2K-1 are R=2 (from conf 1 up to K)
    n_L_S1 = n_s1[0:K]
    n_R_S1 = n_s1[K:2*K]
    n_L_S2 = n_s2[0:K]
    n_R_S2 = n_s2[K:2*K]

    # Sum conditional log likelihood
    log_lik = (
        np.sum(n_L_S1 * np.log(cond_L_S1)) +
        np.sum(n_L_S2 * np.log(cond_L_S2)) +
        np.sum(n_R_S1 * np.log(cond_R_S1)) +
        np.sum(n_R_S2 * np.log(cond_R_S2))
    )

    if np.isnan(log_lik) or np.isinf(log_lik):
        return 1e10

    return -float(log_lik)


def fit_meta_d_from_counts(
    n_s1: np.ndarray,
    n_s2: np.ndarray,
    d1: float,
    c1: float,
    K: int = 4,
) -> Dict[str, Any]:
    """Fit Maniscalco & Lau MLE meta-d' directly from contingency count vectors n_s1 and n_s2."""
    # Check total counts
    if np.sum(n_s1) < 5 or np.sum(n_s2) < 5:
        return {
            "meta_d_status": "insufficient_data",
            "meta_d_prime": None,
            "m_ratio": None,
            "type1_d_prime": d1,
            "type1_criterion_c": c1,
            "meta_criterion_c2": None,
            "log_likelihood": None,
        }

    # Initial parameter guesses
    init_meta_d = float(d1)
    init_tau_L = np.linspace(c1 - 1.8, c1 - 0.2, K - 1)
    init_tau_R = np.linspace(c1 + 0.2, c1 + 1.8, K - 1)
    init_params = np.concatenate([[init_meta_d], init_tau_L, init_tau_R])

    # Parameter bounds: signed meta_d in [-10.0, 10.0]
    bounds = [(-10.0, 10.0)] + [(-15.0, 15.0)] * (2 * (K - 1))

    try:
        res = minimize(
            _meta_d_log_likelihood,
            init_params,
            args=(n_s1, n_s2, K, c1, d1),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-9},
        )

        if not res.success or res.fun >= 1e9:
            # Fallback to Nelder-Mead
            res = minimize(
                _meta_d_log_likelihood,
                init_params,
                args=(n_s1, n_s2, K, c1, d1),
                method="Nelder-Mead",
                options={"maxiter": 3000},
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

        fitted_meta_d = float(res.x[0])
        c2 = c1 * (fitted_meta_d / d1) if abs(d1) > 1e-4 else c1
        m_ratio = float(fitted_meta_d / d1) if (d1 and abs(d1) > 0.05) else None

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


def fit_meta_d_mle(
    records: List[Dict[str, Any]],
    n_bins: int = 4,
    signal_target: str = "A",
) -> Dict[str, Any]:
    """Fit Type-2 meta-d' and M-ratio using Maniscalco & Lau (2012) Maximum Likelihood Estimation."""
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

    # Build contingency counts (fixed K=n_bins)
    n_s1, n_s2, K = build_type2_contingency_table(valid_recs, n_bins=n_bins, signal_target=signal_target)
    return fit_meta_d_from_counts(n_s1, n_s2, d1=float(d1), c1=float(c1), K=K)
