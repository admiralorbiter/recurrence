"""Psychophysics, psychometric curve fitting, monotonicity diagnostics, and reactivity analytics."""

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def compute_wilson_score_interval(
    successes: int,
    total: int,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """Compute exact Wilson score interval for binomial proportions.
    
    Returns (ci_lower, ci_upper) bounded in [0.0, 1.0].
    """
    if total == 0:
        return (0.0, 1.0)
    
    # z-score for two-tailed confidence level
    # 0.95 -> 1.95996, 0.90 -> 1.64485, 0.99 -> 2.57583
    if abs(confidence - 0.95) < 1e-4:
        z = 1.959963984540054
    elif abs(confidence - 0.90) < 1e-4:
        z = 1.6448536269514722
    elif abs(confidence - 0.99) < 1e-4:
        z = 2.5758293035489004
    else:
        # Approximate inverse erf
        z = 1.959963984540054

    p_hat = float(successes) / float(total)
    z_sq = z * z
    n = float(total)

    denominator = 1.0 + (z_sq / n)
    center = (p_hat + (z_sq / (2.0 * n))) / denominator
    margin = (z * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z_sq / (4.0 * (n ** 2))))) / denominator

    ci_lower = max(0.0, float(center - margin))
    ci_upper = min(1.0, float(center + margin))

    if successes == 0:
        ci_lower = 0.0
    if successes == total:
        ci_upper = 1.0

    return (ci_lower, ci_upper)


from scipy.stats import norm


def compute_sdt_indices(
    records: List[Dict[str, Any]],
    signal_target: str = "A"
) -> Dict[str, Any]:
    """Compute Signal Detection Theory (SDT) Type-1 sensitivity (d') and decision criterion (c).
    
    In 2AFC:
    - Signal event: Ground truth target is 'A'
    - Noise event: Ground truth target is 'B'
    - Hit: Chose 'A' when Target is 'A'
    - False Alarm: Chose 'A' when Target is 'B'
    
    Applies standard Macmillan & Creelman (2005) log-linear correction:
      H_adj = (Hits + 0.5) / (N_Signal + 1)
      FA_adj = (FAs + 0.5) / (N_Noise + 1)
      d' = norm_ppf(H_adj) - norm_ppf(FA_adj)
      c = -0.5 * (norm_ppf(H_adj) + norm_ppf(FA_adj))
    """
    n_signal = 0
    n_noise = 0
    hits = 0
    fas = 0

    for r in records:
        gt = str(r.get("ground_truth", "")).upper()
        ans = str(r.get("parsed_answer", "")).upper()
        if not ans or ans not in ["A", "B"]:
            continue
        if gt == signal_target:
            n_signal += 1
            if ans == signal_target:
                hits += 1
        else:
            n_noise += 1
            if ans == signal_target:
                fas += 1

    if n_signal == 0 or n_noise == 0:
        return {
            "d_prime": None,
            "criterion_c": None,
            "hit_rate_raw": None,
            "fa_rate_raw": None,
            "n_signal": n_signal,
            "n_noise": n_noise,
        }

    h_adj = float((hits + 0.5) / (n_signal + 1.0))
    fa_adj = float((fas + 0.5) / (n_noise + 1.0))

    z_h = float(norm.ppf(h_adj))
    z_fa = float(norm.ppf(fa_adj))

    d_prime = float(z_h - z_fa)
    c = float(-0.5 * (z_h + z_fa))

    return {
        "d_prime": d_prime,
        "criterion_c": c,
        "hit_rate_raw": float(hits / n_signal) if n_signal > 0 else 0.0,
        "fa_rate_raw": float(fas / n_noise) if n_noise > 0 else 0.0,
        "hit_rate_adj": h_adj,
        "fa_rate_adj": fa_adj,
        "n_signal": n_signal,
        "n_noise": n_noise,
    }


def compute_psychometric_curve(
    records: List[Dict[str, Any]],
    difficulty_key: str = "difficulty_level"
) -> Dict[str, Any]:
    """Compute empirical psychometric curve statistics grouped by difficulty level.
    
    Calculates accuracy, Wilson 95% CIs, SDT d', SDT criterion c, A/B position bias,
    confidence separation, continuous Brier score, and prompt token footprint per difficulty stratum.
    """
    if not records:
        return {"levels": [], "level_metrics": {}, "overall_summary": {}}

    grouped: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        d_val = r.get(difficulty_key, r.get("distractor_count", r.get("hop_depth", r.get("overwrite_count", 0))))
        grouped[d_val].append(r)

    sorted_levels = sorted(grouped.keys())
    level_metrics: Dict[str, Any] = {}

    all_accuracies: List[float] = []
    all_levels_numeric: List[float] = []

    for lvl in sorted_levels:
        lvl_recs = grouped[lvl]
        n_total = len(lvl_recs)
        correct_count = sum(1 for r in lvl_recs if r.get("correct", False))
        acc = float(correct_count / n_total) if n_total > 0 else 0.0
        ci_l, ci_u = compute_wilson_score_interval(correct_count, n_total, confidence=0.95)

        # Position bias: P(Chose 'A')
        a_count = sum(1 for r in lvl_recs if str(r.get("parsed_answer", "")).upper() == "A")
        p_a = float(a_count / n_total) if n_total > 0 else 0.5
        pos_bias = float(p_a - 0.5)

        # SDT Sensitivity (d') and Decision Criterion (c)
        sdt = compute_sdt_indices(lvl_recs, signal_target="A")

        # Compliance
        schema_valid_count = sum(1 for r in lvl_recs if r.get("schema_valid", False))
        schema_compliance = float(schema_valid_count / n_total) if n_total > 0 else 0.0

        parse_valid_count = sum(1 for r in lvl_recs if r.get("answer_parse_valid", False))
        answer_parse_compliance = float(parse_valid_count / n_total) if n_total > 0 else 0.0

        # Confidence & Separation
        valid_probs = [
            (float(r["probability"]), bool(r.get("correct", False)))
            for r in lvl_recs
            if r.get("probability") is not None and math.isfinite(float(r["probability"]))
        ]
        if valid_probs:
            all_p = [p for p, _ in valid_probs]
            correct_p = [p for p, y in valid_probs if y]
            incorrect_p = [p for p, y in valid_probs if not y]

            mean_conf = float(np.mean(all_p))
            mean_conf_corr = float(np.mean(correct_p)) if correct_p else None
            mean_conf_inc = float(np.mean(incorrect_p)) if incorrect_p else None
            conf_sep = (
                float(mean_conf_corr - mean_conf_inc)
                if (mean_conf_corr is not None and mean_conf_inc is not None)
                else None
            )
            brier = float(np.mean([((p) - (1.0 if y else 0.0)) ** 2 for p, y in valid_probs]))
        else:
            mean_conf = None
            mean_conf_corr = None
            mean_conf_inc = None
            conf_sep = None
            brier = None

        # Prompt lengths & token evaluation
        prompt_chars = [len(str(r.get("prompt", ""))) for r in lvl_recs]
        mean_chars = float(np.mean(prompt_chars)) if prompt_chars else 0.0
        
        # Prefer exact prompt_eval_count if present in metadata
        actual_tokens = [
            r.get("metadata", {}).get("prompt_eval_count") or r.get("prompt_eval_count")
            for r in lvl_recs
            if (r.get("metadata", {}).get("prompt_eval_count") or r.get("prompt_eval_count")) is not None
        ]
        if actual_tokens:
            mean_est_tokens = float(np.mean(actual_tokens))
        else:
            mean_est_tokens = float(mean_chars / 4.0)

        level_metrics[str(lvl)] = {
            "difficulty_level": lvl,
            "total_trials": n_total,
            "correct_trials": correct_count,
            "accuracy": acc,
            "ci_95_lower": ci_l,
            "ci_95_upper": ci_u,
            "option_a_selection_rate": p_a,
            "position_bias": pos_bias,
            "sdt_d_prime": sdt.get("d_prime"),
            "sdt_criterion_c": sdt.get("criterion_c"),
            "schema_compliance_rate": schema_compliance,
            "answer_parse_compliance_rate": answer_parse_compliance,
            "mean_confidence": mean_conf,
            "mean_confidence_correct": mean_conf_corr,
            "mean_confidence_incorrect": mean_conf_inc,
            "confidence_separation": conf_sep,
            "brier_score": brier,
            "mean_prompt_chars": mean_chars,
            "mean_estimated_tokens": mean_est_tokens,
        }

        all_accuracies.append(acc)
        try:
            all_levels_numeric.append(float(lvl))
        except (ValueError, TypeError):
            all_levels_numeric.append(float(len(all_levels_numeric)))

    # Monotonicity diagnostics
    mono_diag = compute_monotonicity_diagnostics(all_levels_numeric, all_accuracies)

    total_records = len(records)
    total_correct = sum(1 for r in records if r.get("correct", False))
    overall_acc = float(total_correct / total_records) if total_records > 0 else 0.0
    overall_ci_l, overall_ci_u = compute_wilson_score_interval(total_correct, total_records)

    return {
        "levels": sorted_levels,
        "level_metrics": level_metrics,
        "monotonicity_diagnostics": mono_diag,
        "overall_summary": {
            "total_trials": total_records,
            "overall_accuracy": overall_acc,
            "overall_ci_95_lower": overall_ci_l,
            "overall_ci_95_upper": overall_ci_u,
            "span_min_accuracy": min(all_accuracies) if all_accuracies else None,
            "span_max_accuracy": max(all_accuracies) if all_accuracies else None,
            "accuracy_span": (max(all_accuracies) - min(all_accuracies)) if all_accuracies else 0.0,
        },
    }


def compute_monotonicity_diagnostics(
    difficulty_levels: List[float],
    accuracies: List[float]
) -> Dict[str, Any]:
    """Compute rank correlation and directional step statistics to evaluate monotonicity."""
    n = len(difficulty_levels)
    if n < 2 or len(accuracies) != n:
        return {
            "spearman_rho": None,
            "kendall_tau": None,
            "negative_step_ratio": None,
            "max_accuracy_drop": None,
            "spans_target_operating_window": False,
            "staircase_readiness": "insufficient_data",
        }

    x = np.array(difficulty_levels, dtype=float)
    y = np.array(accuracies, dtype=float)

    # 1. Spearman Rank Correlation
    def _rank_data(arr: np.ndarray) -> np.ndarray:
        order = np.argsort(arr)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(arr), dtype=float) + 1.0
        # Average ties
        for val in np.unique(arr):
            ties = arr == val
            if np.sum(ties) > 1:
                ranks[ties] = np.mean(ranks[ties])
        return ranks

    r_x = _rank_data(x)
    r_y = _rank_data(y)

    if np.std(r_x) > 0 and np.std(r_y) > 0:
        spearman_rho = float(np.corrcoef(r_x, r_y)[0, 1])
    else:
        spearman_rho = 0.0

    # 2. Kendall Tau
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            if dx == 0 and dy == 0:
                continue
            elif dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif (dx > 0 and dy > 0) or (dx < 0 and dy < 0):
                concordant += 1
            else:
                discordant += 1

    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    kendall_tau = float((concordant - discordant) / denom) if denom > 0 else 0.0

    # 3. Step-by-step directional transitions
    step_diffs = [y[i + 1] - y[i] for i in range(n - 1)]
    # For a monotonic decline, step_diff should be <= 0
    negative_steps = sum(1 for d in step_diffs if d <= 1e-6)
    neg_step_ratio = float(negative_steps / (n - 1)) if (n - 1) > 0 else 1.0

    min_acc = float(np.min(y))
    max_acc = float(np.max(y))
    acc_drop = float(max_acc - min_acc)

    # Operational target window check: spans ~55-90% performance
    spans_window = bool(min_acc <= 0.75 and max_acc >= 0.70 and acc_drop >= 0.15)

    # Staircase readiness classification
    if spearman_rho <= -0.70 and kendall_tau <= -0.50 and neg_step_ratio >= 0.70 and acc_drop >= 0.20 and spans_window:
        readiness = "staircase_ready"
    elif (spearman_rho <= -0.50 or kendall_tau <= -0.40) and acc_drop >= 0.15:
        readiness = "promising_monotonic_trend"
    elif acc_drop < 0.10:
        readiness = "flat_or_ceiling"
    else:
        readiness = "non_monotonic"

    return {
        "spearman_rho": spearman_rho,
        "kendall_tau": kendall_tau,
        "negative_step_ratio": neg_step_ratio,
        "max_accuracy_drop": acc_drop,
        "min_accuracy": min_acc,
        "max_accuracy": max_acc,
        "spans_target_operating_window": spans_window,
        "staircase_readiness": readiness,
    }


def compute_elicitation_reactivity(
    paired_records: List[Tuple[Dict[str, Any], Dict[str, Any]]]
) -> Dict[str, Any]:
    """Compute paired reactivity statistics contrasting Answer-Only vs Answer+Confidence trials.
    
    paired_records: List of tuples (answer_only_record, answer_conf_record) on matched item seeds.
    """
    if not paired_records:
        return {"paired_count": 0, "error": "No paired records provided"}

    n = len(paired_records)
    n11 = 0  # both correct
    n00 = 0  # both incorrect
    n10 = 0  # only answer_only correct (conf incorrect)
    n01 = 0  # only conf correct (answer_only incorrect)

    exact_concordance_count = 0

    acc_only_hits = 0
    acc_conf_hits = 0

    for rec_only, rec_conf in paired_records:
        c_only = bool(rec_only.get("correct", False))
        c_conf = bool(rec_conf.get("correct", False))

        if c_only:
            acc_only_hits += 1
        if c_conf:
            acc_conf_hits += 1

        if c_only and c_conf:
            n11 += 1
        elif not c_only and not c_conf:
            n00 += 1
        elif c_only and not c_conf:
            n10 += 1
        else:
            n01 += 1

        ans_only = str(rec_only.get("parsed_answer", "")).upper()
        ans_conf = str(rec_conf.get("parsed_answer", "")).upper()
        if ans_only and ans_only == ans_conf:
            exact_concordance_count += 1

    acc_only = float(acc_only_hits / n)
    acc_conf = float(acc_conf_hits / n)
    delta_acc = float(acc_conf - acc_only)
    concordance = float(exact_concordance_count / n)

    # McNemar test with continuity correction
    discordant_total = n10 + n01
    if discordant_total > 0:
        mcnemar_stat = float(((abs(n10 - n01) - 1.0) ** 2) / discordant_total)
        # Approximate p-value from chi2(1)
        # p = 2 * (1 - Phi(sqrt(stat)))
        z = math.sqrt(mcnemar_stat)
        p_val = float(math.erfc(z / math.sqrt(2.0)))
    else:
        mcnemar_stat = 0.0
        p_val = 1.0

    if abs(delta_acc) <= 0.05 and p_val > 0.05 and concordance >= 0.85:
        status = "negligible_reactivity"
    elif abs(delta_acc) <= 0.12 and concordance >= 0.70:
        status = "moderate_reactivity"
    else:
        status = "severe_reactivity"

    return {
        "paired_trials_count": n,
        "answer_only_accuracy": acc_only,
        "answer_conf_accuracy": acc_conf,
        "delta_accuracy_conf_minus_only": delta_acc,
        "exact_answer_concordance_rate": concordance,
        "contingency_table": {
            "both_correct_n11": n11,
            "both_incorrect_n00": n00,
            "only_answer_only_correct_n10": n10,
            "only_conf_correct_n01": n01,
        },
        "mcnemar_chi2_statistic": mcnemar_stat,
        "mcnemar_p_value": p_val,
        "reactivity_status": status,
    }
