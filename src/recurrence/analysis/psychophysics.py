"""Psychophysics, psychometric curve fitting, monotonicity diagnostics, and reactivity analytics."""

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import binom


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


def compute_sdt_bootstrap_ci(
    records: List[Dict[str, Any]],
    signal_target: str = "A",
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Optional[float]]:
    """Compute stratified bootstrap confidence intervals for SDT d' and criterion c."""
    sig_records = [r for r in records if str(r.get("ground_truth", "")).upper() == signal_target]
    noise_records = [r for r in records if str(r.get("ground_truth", "")).upper() != signal_target]
    n_sig = len(sig_records)
    n_noise = len(noise_records)

    if n_sig < 2 or n_noise < 2:
        return {
            "d_prime_ci_lower": None,
            "d_prime_ci_upper": None,
            "criterion_c_ci_lower": None,
            "criterion_c_ci_upper": None,
        }

    rng = np.random.RandomState(seed)
    boot_d_primes = []
    boot_cs = []

    alpha = (1.0 - ci) / 2.0
    for _ in range(n_bootstraps):
        boot_sig = [sig_records[idx] for idx in rng.choice(n_sig, size=n_sig, replace=True)]
        boot_noise = [noise_records[idx] for idx in rng.choice(n_noise, size=n_noise, replace=True)]
        boot_sdt = compute_sdt_indices(boot_sig + boot_noise, signal_target=signal_target)
        if boot_sdt["d_prime"] is not None and boot_sdt["criterion_c"] is not None:
            boot_d_primes.append(boot_sdt["d_prime"])
            boot_cs.append(boot_sdt["criterion_c"])

    if not boot_d_primes:
        return {
            "d_prime_ci_lower": None,
            "d_prime_ci_upper": None,
            "criterion_c_ci_lower": None,
            "criterion_c_ci_upper": None,
        }

    return {
        "d_prime_ci_lower": float(np.percentile(boot_d_primes, alpha * 100)),
        "d_prime_ci_upper": float(np.percentile(boot_d_primes, (1.0 - alpha) * 100)),
        "criterion_c_ci_lower": float(np.percentile(boot_cs, alpha * 100)),
        "criterion_c_ci_upper": float(np.percentile(boot_cs, (1.0 - alpha) * 100)),
    }


def compute_nested_paired_transitions(
    records: List[Dict[str, Any]],
    difficulty_key: str = "difficulty_level"
) -> List[Dict[str, Any]]:
    """Compute adjacent-level paired transitions (correct->wrong vs wrong->correct) across difficulty levels.
    
    Identifies item identity by metadata seed / target_key to evaluate within-item stability.
    """
    grouped_by_level: Dict[Any, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    
    for idx, r in enumerate(records):
        lvl = r.get(difficulty_key, r.get("distractor_count", r.get("hop_depth", r.get("overwrite_count", 0))))
        t_key = r.get("target_key") or (r.get("metadata", {}).get("target_key") if isinstance(r.get("metadata"), dict) else None)
        if t_key:
            item_key = str(t_key)
        else:
            item_id = str(r.get("item_id", idx))
            if "_s" in item_id:
                item_key = "s" + item_id.split("_s")[-1]
            else:
                item_key = item_id
        grouped_by_level[lvl][item_key] = r

    sorted_levels = sorted(grouped_by_level.keys())
    transitions = []

    for i in range(len(sorted_levels) - 1):
        lvl1 = sorted_levels[i]
        lvl2 = sorted_levels[i + 1]
        recs1 = grouped_by_level[lvl1]
        recs2 = grouped_by_level[lvl2]

        common_keys = set(recs1.keys()) & set(recs2.keys())
        if not common_keys:
            continue

        both_correct = 0
        degraded_1_to_0 = 0
        persisted_wrong_0_to_0 = 0
        rebounded_0_to_1 = 0

        for k in common_keys:
            c1 = bool(recs1[k].get("correct", False))
            c2 = bool(recs2[k].get("correct", False))
            if c1 and c2:
                both_correct += 1
            elif c1 and not c2:
                degraded_1_to_0 += 1
            elif not c1 and not c2:
                persisted_wrong_0_to_0 += 1
            else:  # not c1 and c2
                rebounded_0_to_1 += 1

        n_pairs = len(common_keys)
        transitions.append({
            "from_level": lvl1,
            "to_level": lvl2,
            "paired_items_count": n_pairs,
            "retained_correct_1_to_1": both_correct,
            "degraded_1_to_0": degraded_1_to_0,
            "persisted_wrong_0_to_0": persisted_wrong_0_to_0,
            "rebounded_0_to_1": rebounded_0_to_1,
            "net_accuracy_delta": (rebounded_0_to_1 - degraded_1_to_0) / n_pairs if n_pairs > 0 else 0.0,
            "degradation_rate": degraded_1_to_0 / (both_correct + degraded_1_to_0) if (both_correct + degraded_1_to_0) > 0 else 0.0,
            "rebound_rate": rebounded_0_to_1 / (persisted_wrong_0_to_0 + rebounded_0_to_1) if (persisted_wrong_0_to_0 + rebounded_0_to_1) > 0 else 0.0,
        })

    return transitions


def compute_psychometric_curve(
    records: List[Dict[str, Any]],
    difficulty_key: str = "difficulty_level"
) -> Dict[str, Any]:
    """Compute empirical psychometric curve statistics grouped by difficulty level.
    
    Calculates accuracy, Wilson 95% CIs, SDT d', SDT criterion c with bootstrap CIs,
    A/B position bias, confidence separation, continuous Brier score, and prompt token footprint.
    """
    if not records:
        return {"levels": [], "level_metrics": {}, "overall_summary": {}}

    grouped: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        d_val = r.get(difficulty_key, r.get("distractor_count", r.get("hop_depth", r.get("overwrite_count", 0))))
        grouped[d_val].append(r)

    sorted_levels = sorted(grouped.keys())
    level_metrics: Dict[str, Dict[str, Any]] = {}
    all_accuracies: List[float] = []
    all_levels_numeric: List[float] = []

    for lvl in sorted_levels:
        lvl_recs = grouped[lvl]
        n_total = len(lvl_recs)
        correct_count = sum(1 for r in lvl_recs if r.get("correct", False))
        acc = float(correct_count / n_total) if n_total > 0 else 0.0
        ci_l, ci_u = compute_wilson_score_interval(correct_count, n_total)

        # Position bias: P(A chosen)
        a_count = sum(1 for r in lvl_recs if str(r.get("parsed_answer", "")).upper() == "A")
        p_a = float(a_count / n_total) if n_total > 0 else 0.5
        pos_bias = float(p_a - 0.5)

        # SDT sensitivity and criterion
        sdt = compute_sdt_indices(lvl_recs, signal_target="A")
        sdt_ci = compute_sdt_bootstrap_ci(lvl_recs, signal_target="A")

        # Schema & parsing compliance
        schema_compliance = float(sum(1 for r in lvl_recs if r.get("schema_valid", False)) / n_total) if n_total > 0 else 0.0
        answer_parse_compliance = float(sum(1 for r in lvl_recs if r.get("answer_parse_valid", True)) / n_total) if n_total > 0 else 0.0

        # Confidence and Brier
        conf_vals = [r.get("probability") for r in lvl_recs if r.get("probability") is not None]
        mean_conf = float(np.mean(conf_vals)) if conf_vals else None

        conf_corr = [r.get("probability") for r in lvl_recs if r.get("correct", False) and r.get("probability") is not None]
        conf_inc = [r.get("probability") for r in lvl_recs if not r.get("correct", False) and r.get("probability") is not None]
        mean_conf_corr = float(np.mean(conf_corr)) if conf_corr else None
        mean_conf_inc = float(np.mean(conf_inc)) if conf_inc else None
        conf_sep = (mean_conf_corr - mean_conf_inc) if (mean_conf_corr is not None and mean_conf_inc is not None) else None

        # Continuous Brier score
        brier_sq_diffs = []
        for r in lvl_recs:
            if r.get("probability") is not None:
                p_c = float(r["probability"])
                o_c = 1.0 if r.get("correct", False) else 0.0
                brier_sq_diffs.append((p_c - o_c) ** 2)
        brier = float(np.mean(brier_sq_diffs)) if brier_sq_diffs else None

        # Token footprint
        mean_chars = float(np.mean([len(r.get("prompt", "")) for r in lvl_recs])) if lvl_recs else 0.0
        actual_tokens = [
            r.get("metadata", {}).get("prompt_eval_count") or r.get("prompt_eval_count")
            for r in lvl_recs
            if (r.get("metadata", {}).get("prompt_eval_count") or r.get("prompt_eval_count")) is not None
        ]
        mean_est_tokens = float(np.mean(actual_tokens)) if actual_tokens else float(mean_chars / 4.0)

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
            "sdt_d_prime_ci_lower": sdt_ci.get("d_prime_ci_lower"),
            "sdt_d_prime_ci_upper": sdt_ci.get("d_prime_ci_upper"),
            "sdt_criterion_c": sdt.get("criterion_c"),
            "sdt_criterion_c_ci_lower": sdt_ci.get("criterion_c_ci_lower"),
            "sdt_criterion_c_ci_upper": sdt_ci.get("criterion_c_ci_upper"),
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

    # Monotonicity diagnostics & paired transitions
    mono_diag = compute_monotonicity_diagnostics(all_levels_numeric, all_accuracies)
    paired_transitions = compute_nested_paired_transitions(records, difficulty_key=difficulty_key)

    total_records = len(records)
    total_correct = sum(1 for r in records if r.get("correct", False))
    overall_acc = float(total_correct / total_records) if total_records > 0 else 0.0
    overall_ci_l, overall_ci_u = compute_wilson_score_interval(total_correct, total_records)

    return {
        "levels": sorted_levels,
        "level_metrics": level_metrics,
        "monotonicity_diagnostics": mono_diag,
        "paired_transitions": paired_transitions,
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

    # McNemar test with continuity correction and exact binomial test
    discordant_total = n10 + n01
    if discordant_total > 0:
        mcnemar_stat = float(((abs(n10 - n01) - 1.0) ** 2) / discordant_total)
        z = math.sqrt(mcnemar_stat)
        p_val_asymptotic = float(math.erfc(z / math.sqrt(2.0)))
        # Exact two-tailed binomial p-value
        k = min(n10, n01)
        p_val_exact = min(1.0, 2.0 * float(binom.cdf(k, discordant_total, 0.5)))
    else:
        mcnemar_stat = 0.0
        p_val_asymptotic = 1.0
        p_val_exact = 1.0

    if abs(delta_acc) <= 0.05 and p_val_exact > 0.05 and concordance >= 0.85:
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
        "mcnemar_p_value": p_val_asymptotic,
        "exact_mcnemar_p_value": p_val_exact,
        "reactivity_status": status,
    }


def compute_exact_mcnemar_p_value(b: int, c: int) -> float:
    """Compute exact two-tailed binomial p-value for discordant pairs in paired 2x2 data."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * float(binom.cdf(k, n, 0.5)))


def evaluate_calibration_gate(
    level_metrics: Dict[str, Any],
    target_d_prime_range: Tuple[float, float] = (0.9, 1.4),
    target_accuracy_range: Tuple[float, float] = (0.60, 0.80),
    max_absolute_criterion: float = 0.50,
    min_compliance_rate: float = 0.95,
) -> Dict[str, Any]:
    """Evaluate whether a mini-block / difficulty stratum satisfies the multi-criteria calibration gate.
    
    Gate Requirements:
    1. Sensitivity d' in target operating window (default: ~1.0 to 1.3).
    2. Accuracy in target operating window (default: ~60% to 80%).
    3. Low response bias: |c| < 0.50 (no category collapse / extreme letter preference).
    4. Near-perfect schema compliance: >= 95%.
    """
    d_prime = level_metrics.get("sdt_d_prime")
    criterion_c = level_metrics.get("sdt_criterion_c")
    acc = level_metrics.get("accuracy", 0.0)
    compliance = level_metrics.get("schema_compliance_rate", 1.0)

    d_prime_pass = bool(d_prime is not None and (target_d_prime_range[0] <= d_prime <= target_d_prime_range[1]))
    acc_pass = bool(target_accuracy_range[0] <= acc <= target_accuracy_range[1])
    criterion_pass = bool(criterion_c is not None and (abs(criterion_c) <= max_absolute_criterion))
    compliance_pass = bool(compliance >= min_compliance_rate)

    gate_passed = bool(d_prime_pass and acc_pass and criterion_pass and compliance_pass)

    return {
        "gate_passed": gate_passed,
        "d_prime_pass": d_prime_pass,
        "accuracy_pass": acc_pass,
        "criterion_pass": criterion_pass,
        "compliance_pass": compliance_pass,
        "observed_d_prime": d_prime,
        "observed_criterion_c": criterion_c,
        "observed_accuracy": acc,
        "observed_compliance": compliance,
    }


def compute_type2_sdt_metrics(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute Type-2 Signal Detection metrics (AUROC2, meta-d' proxy, Brier) with degenerate confidence detection.
    
    Status Codes:
    - 'estimable': Non-degenerate confidence variation across correct/incorrect items; Type-2 metrics estimable.
    - 'confidence_degenerate': Invariant confidence ratings (e.g. 100% or 50% on all items); Type-2 criteria undefined.
    - 'insufficient_class_counts': No errors (100% acc) or no correct trials (0% acc).
    - 'insufficient_data': Fewer than 5 valid trials.
    """
    valid_recs = [r for r in records if r.get("probability") is not None and r.get("correct") is not None]
    if len(valid_recs) < 5:
        return {
            "meta_d_status": "insufficient_data",
            "auroc2": None,
            "meta_d_prime": None,
            "m_ratio": None,
            "brier_score": None,
            "unique_confidence_levels": 0,
        }

    confidences = np.array([r["probability"] for r in valid_recs], dtype=float)
    corrects = np.array([1 if r["correct"] else 0 for r in valid_recs], dtype=int)
    n_corr = int(np.sum(corrects))
    n_inc = len(corrects) - n_corr

    brier = float(np.mean((confidences - corrects) ** 2))
    unique_confs = np.unique(confidences)
    n_unique = len(unique_confs)

    if n_corr == 0 or n_inc == 0:
        return {
            "meta_d_status": "insufficient_class_counts",
            "auroc2": None,
            "meta_d_prime": None,
            "m_ratio": None,
            "brier_score": brier,
            "unique_confidence_levels": n_unique,
            "mean_confidence": float(np.mean(confidences)),
        }

    # Degenerate confidence (invariant rating across all items)
    if n_unique <= 1 or np.std(confidences) < 1e-4:
        return {
            "meta_d_status": "confidence_degenerate",
            "auroc2": 0.50,  # Constant ranking yields chance AUROC2
            "meta_d_prime": None,
            "m_ratio": None,
            "brier_score": brier,
            "unique_confidence_levels": n_unique,
            "mean_confidence": float(np.mean(confidences)),
        }

    # Non-parametric AUROC2 via Mann-Whitney U
    r_corr = confidences[corrects == 1]
    r_inc = confidences[corrects == 0]
    concordant = sum(1.0 for c in r_corr for i in r_inc if c > i)
    ties = sum(0.5 for c in r_corr for i in r_inc if c == i)
    auroc2 = float((concordant + ties) / (n_corr * n_inc))

    sdt_t1 = compute_sdt_indices(valid_recs)
    d1 = sdt_t1.get("d_prime")

    return {
        "meta_d_status": "estimable",
        "auroc2": auroc2,
        "meta_d_prime": (d1 * auroc2 * 2.0) if d1 is not None else None,
        "m_ratio": (auroc2 * 2.0) if (d1 and d1 > 0) else None,
        "brier_score": brier,
        "unique_confidence_levels": n_unique,
        "mean_confidence": float(np.mean(confidences)),
        "mean_conf_correct": float(np.mean(r_corr)),
        "mean_conf_incorrect": float(np.mean(r_inc)),
        "confidence_separation": float(np.mean(r_corr) - np.mean(r_inc)),
    }


