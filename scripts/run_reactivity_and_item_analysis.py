"""Comprehensive Horizon 0 Reactivity Control & Offline Item-Level Panel Analysis.

1. Runs paired Answer-Only (no confidence elicited) across 6 models on the exact 40 items.
2. Performs offline item-level panel analysis across all 6 existing Level-0 runs:
   - Item consensus vs disagreement subsets
   - Ashuach et al. (2026) disagreement subset discrimination test
   - Option-letter bias & distribution
   - Moran & Whiting (2025) Shared Difficulty Axis test
   - Pairwise model answer & error correlations
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(".").resolve()))

from recurrence.backends.ollama import OllamaBackend
from recurrence.tasks.kv_retrieval import KVRetrievalTask
from recurrence.core.schemas import TARGET_ANSWER_ONLY_SCHEMA
from recurrence.analysis.calibration import compute_auroc2, compute_post_decision_discrimination_from_pairs
from recurrence.analysis.privileged_access import compute_continuous_brier_score

MODELS = [
    {"name": "qwen2.5:1.5b", "run_id": "run_e02_obs_qwen1_5b_001", "role": "1.5B"},
    {"name": "qwen2.5:3b", "run_id": "run_e02_obs_005", "role": "3B (Ref)"},
    {"name": "qwen2.5:7b", "run_id": "run_e02_obs_qwen7b_001", "role": "7B"},
    {"name": "qwen2.5:14b", "run_id": "run_e02_obs_qwen14b_001", "role": "14B"},
    {"name": "llama3.2:3b", "run_id": "run_e02_obs_llama3_2_3b_001", "role": "Llama-3.2 3B"},
    {"name": "mistral:latest", "run_id": "run_e02_obs_mistral7b_001", "role": "Mistral 7B"},
]


def run_reactivity_benchmark(seed: int = 42) -> Dict[str, Any]:
    """Run pure Answer-Only evaluation across all 6 models on the identical 40 counterbalanced items."""
    print("=" * 60)
    print("STEP 1: RUNNING PAIRED CONFIDENCE-REACTIVITY BENCHMARK")
    print("=" * 60)

    # 1. Generate exact identical 40 items under ask_confidence=False
    task_semantic = KVRetrievalTask(identifier_type="semantic", mode="forced_choice", ask_confidence=False, confidence_format="probability")
    raw_semantic = task_semantic.generate_raw_pairs(count=20, distractor_count=5, identifier_type="semantic", seed=seed)
    items_semantic = task_semantic.generate_items_from_raw(raw_semantic, seed=seed)

    task_opaque = KVRetrievalTask(identifier_type="opaque", mode="forced_choice", ask_confidence=False, confidence_format="probability")
    raw_opaque = task_opaque.generate_raw_pairs(count=20, distractor_count=5, identifier_type="opaque", seed=seed + 1000)
    items_opaque = task_opaque.generate_items_from_raw(raw_opaque, seed=seed + 1000)

    all_items = [(task_semantic, it) for it in items_semantic] + [(task_opaque, it) for it in items_opaque]

    reactivity_results = {}

    for m in MODELS:
        model_name = m["name"]
        run_id = m["run_id"]
        print(f"\nEvaluating Answer-Only on {model_name}...")

        backend = OllamaBackend(model_name=model_name, seed=seed)
        model_digest = backend.get_digest()

        ans_only_records = []
        semantic_corr = 0
        opaque_corr = 0

        for i, (task, item) in enumerate(all_items):
            messages = [{"role": "user", "content": item.prompt}]
            resp, meta = backend.chat(
                messages=messages,
                temperature=0.0,
                seed=seed,
                format=TARGET_ANSWER_ONLY_SCHEMA,
            )
            score = task.score_response(item, resp)
            is_corr = score["correct"]
            if is_corr:
                if item.metadata["identifier_type"] == "semantic":
                    semantic_corr += 1
                else:
                    opaque_corr += 1

            ans_only_records.append({
                "item_id": item.item_id,
                "ground_truth": item.ground_truth,
                "parsed_answer": score.get("parsed_answer"),
                "correct": is_corr,
                "schema_valid": score.get("schema_valid"),
                "raw_response": resp,
            })

        acc_sem = semantic_corr / 20.0
        acc_op = opaque_corr / 20.0
        acc_tot = (semantic_corr + opaque_corr) / 40.0

        # Load matched baseline with confidence
        base_trials_file = Path("results/e02_observer") / run_id / "trials.jsonl"
        base_trials = []
        with open(base_trials_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    base_trials.append(json.loads(line))

        base_corr = [t["target_correct"] for t in base_trials]
        base_answers = [t["target_parsed_answer"] for t in base_trials]
        base_acc = np.mean(base_corr)

        # Paired comparison
        same_answers = sum(1 for a, b in zip([r["parsed_answer"] for r in ans_only_records], base_answers) if a == b)
        # Contingency table for McNemar's test
        # b: correct in ans-only but incorrect in ans+conf
        # c: incorrect in ans-only but correct in ans+conf
        n_b = sum(1 for a, b in zip([r["correct"] for r in ans_only_records], base_corr) if a and not b)
        n_c = sum(1 for a, b in zip([r["correct"] for r in ans_only_records], base_corr) if not a and b)

        # Exact binomial test for McNemar (2-sided)
        if n_b + n_c > 0:
            mcnemar_p = stats.binomtest(n_b, n_b + n_c, 0.5).pvalue
        else:
            mcnemar_p = 1.0

        delta_acc = float(base_acc - acc_tot)

        reactivity_results[model_name] = {
            "model_name": model_name,
            "role": m["role"],
            "model_digest": model_digest,
            "answer_only_accuracy_overall": acc_tot,
            "answer_only_accuracy_semantic": acc_sem,
            "answer_only_accuracy_opaque": acc_op,
            "answer_plus_conf_accuracy_overall": float(base_acc),
            "delta_accuracy_conf_minus_ansonly": delta_acc,
            "answer_concordance_rate": same_answers / 40.0,
            "contingency_table": {"b_only_ansonly_correct": n_b, "c_only_ansconf_correct": n_c},
            "mcnemar_p_value": float(mcnemar_p),
            "trials": ans_only_records,
        }

        print(f"  -> Ans-Only Acc: {acc_tot:.1%} (Semantic: {acc_sem:.1%}, Opaque: {acc_op:.1%})")
        print(f"  -> Ans+Conf Acc: {base_acc:.1%} | Delta: {delta_acc:+.1%} (McNemar p={mcnemar_p:.4f})")
        print(f"  -> Answer Concordance: {same_answers}/40 ({same_answers/40.0:.1%})")

    # Save reactivity results
    out_dir = Path("results/reactivity_control")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "reactivity_summary.json", "w", encoding="utf-8") as f:
        # Exclude raw trials from summary json for brevity
        clean_summary = {k: {sk: sv for sk, sv in v.items() if sk != "trials"} for k, v in reactivity_results.items()}
        json.dump(clean_summary, f, indent=2)

    with open(out_dir / "reactivity_trials.json", "w", encoding="utf-8") as f:
        json.dump(reactivity_results, f, indent=2)

    return reactivity_results


def run_item_level_panel_analysis() -> Dict[str, Any]:
    """Perform offline item-level panel analysis across all 6 existing Level-0 runs."""
    print("\n" + "=" * 60)
    print("STEP 2: OFFLINE ITEM-LEVEL PANEL ANALYSIS (0 MODEL CALLS)")
    print("=" * 60)

    # 1. Load trials across all 6 models
    model_trials: Dict[str, List[Dict[str, Any]]] = {}
    for m in MODELS:
        run_id = m["run_id"]
        t_file = Path("results/e02_observer") / run_id / "trials.jsonl"
        trials = []
        with open(t_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    trials.append(json.loads(line))
        model_trials[m["name"]] = trials

    item_ids = [t["item_id"] for t in model_trials[MODELS[0]["name"]]]
    ground_truths = [t["ground_truth"] for t in model_trials[MODELS[0]["name"]]]
    n_items = len(item_ids)

    # 2. Item Consensus Matrix
    # Matrix of correctness: shape (6, 40)
    correct_matrix = np.array([[t["target_correct"] for t in model_trials[m["name"]]] for m in MODELS], dtype=int)
    # Matrix of self confidences: shape (6, 40)
    conf_matrix = np.array([[t["self_probability"] if t["self_probability"] is not None else np.nan for t in model_trials[m["name"]]] for m in MODELS])
    # Matrix of answers: shape (6, 40)
    answer_matrix = np.array([[t["target_parsed_answer"] for t in model_trials[m["name"]]] for m in MODELS])

    # Item difficulty / pass rate across panel
    item_pass_rates = np.mean(correct_matrix, axis=0)
    consensus_correct_items = np.where(item_pass_rates == 1.0)[0].tolist()
    consensus_incorrect_items = np.where(item_pass_rates == 0.0)[0].tolist()
    disagreement_items = np.where((item_pass_rates > 0.0) & (item_pass_rates < 1.0))[0].tolist()

    print(f"\nPanel Consensus Spectrum (N=40 total items):")
    print(f"  - Consensus Correct (all 6 models correct): {len(consensus_correct_items)} items ({len(consensus_correct_items)/40.0:.1%})")
    print(f"  - Consensus Incorrect (all 6 models failed): {len(consensus_incorrect_items)} items ({len(consensus_incorrect_items)/40.0:.1%})")
    print(f"  - Disagreement Items (1 to 5 models correct): {len(disagreement_items)} items ({len(disagreement_items)/40.0:.1%})")

    # 3. Ashuach et al. (2026) Disagreement Subset Discrimination Test
    print("\n--- Ashuach et al. Disagreement Subset Discrimination ---")
    disagreement_results = {}
    for m in MODELS:
        m_name = m["name"]
        m_confs = conf_matrix[MODELS.index(m), disagreement_items]
        m_corrs = correct_matrix[MODELS.index(m), disagreement_items]
        
        valid_pairs = [(float(c), bool(y)) for c, y in zip(m_confs, m_corrs) if not np.isnan(c)]
        disc = compute_post_decision_discrimination_from_pairs(valid_pairs)
        disagreement_results[m_name] = {
            "disagreement_n": len(valid_pairs),
            "accuracy_on_disagreement": float(np.mean(m_corrs)) if len(m_corrs) > 0 else None,
            "auroc2_on_disagreement": disc.get("auroc2"),
            "mean_conf_correct": disc.get("mean_confidence_correct"),
            "mean_conf_incorrect": disc.get("mean_confidence_incorrect"),
            "separation": disc.get("confidence_separation"),
        }
        print(f"  {m['role']:15s}: Acc = {np.mean(m_corrs):.1%}, AUROC2 on Disagreement = {disc.get('auroc2')}")

    # 4. Moran & Whiting (2025) Shared Difficulty Axis Correlation
    print("\n--- Moran & Whiting Shared Difficulty Axis Analysis ---")
    difficulty_correlations = {}
    for m in MODELS:
        m_confs = conf_matrix[MODELS.index(m)]
        m_corrs = correct_matrix[MODELS.index(m)]
        
        valid_idx = [i for i in range(n_items) if not np.isnan(m_confs[i])]
        if len(valid_idx) > 5 and np.std(m_confs[valid_idx]) > 1e-6:
            r_shared, p_shared = stats.pearsonr(m_confs[valid_idx], item_pass_rates[valid_idx])
            rho_shared, p_rho = stats.spearmanr(m_confs[valid_idx], item_pass_rates[valid_idx])
            # Model specific correctness correlation
            r_indiv, p_indiv = stats.pointbiserialr(m_corrs[valid_idx], m_confs[valid_idx])
        else:
            r_shared, rho_shared, r_indiv = None, None, None

        difficulty_correlations[m["name"]] = {
            "pearson_r_with_shared_difficulty": float(r_shared) if r_shared is not None else None,
            "spearman_rho_with_shared_difficulty": float(rho_shared) if rho_shared is not None else None,
            "point_biserial_r_with_own_correctness": float(r_indiv) if r_indiv is not None else None,
        }
        print(f"  {m['role']:15s}: r(Conf, Shared Pass-Rate) = {r_shared}, r(Conf, Own Correctness) = {r_indiv}")

    # 5. Option-Letter Bias Analysis
    print("\n--- Option Letter Selection Distribution (Ground truth is 25% each) ---")
    option_biases = {}
    options = ["A", "B", "C", "D"]
    for m in MODELS:
        answers = answer_matrix[MODELS.index(m)]
        counts = {opt: int(np.sum(answers == opt)) for opt in options}
        props = {opt: float(counts[opt] / n_items) for opt in options}
        # Chi-square test against uniform 10 per option
        chi2, p_val = stats.chisquare(list(counts.values()), [10, 10, 10, 10])
        option_biases[m["name"]] = {
            "counts": counts,
            "proportions": props,
            "chi2_stat": float(chi2),
            "p_value": float(p_val),
            "has_significant_position_bias": bool(p_val < 0.05),
        }
        print(f"  {m['role']:15s}: A={counts['A']:2d}, B={counts['B']:2d}, C={counts['C']:2d}, D={counts['D']:2d} | Chi2={chi2:.2f} (p={p_val:.4f})")

    # 6. Pairwise Model Answer Concordance
    pairwise_concordance = {}
    for i, m1 in enumerate(MODELS):
        for j, m2 in enumerate(MODELS):
            if i <= j:
                same = int(np.sum(answer_matrix[i] == answer_matrix[j]))
                pairwise_concordance[f"{m1['name']}__vs__{m2['name']}"] = {
                    "model_1": m1["name"],
                    "model_2": m2["name"],
                    "identical_answers": same,
                    "concordance_rate": float(same / n_items),
                }

    analysis_summary = {
        "panel_consensus": {
            "total_items": n_items,
            "consensus_correct_count": len(consensus_correct_items),
            "consensus_incorrect_count": len(consensus_incorrect_items),
            "disagreement_items_count": len(disagreement_items),
            "consensus_correct_indices": consensus_correct_items,
            "consensus_incorrect_indices": consensus_incorrect_items,
            "disagreement_indices": disagreement_items,
        },
        "ashuach_disagreement_subsets": disagreement_results,
        "moran_whiting_difficulty_axis": difficulty_correlations,
        "option_letter_biases": option_biases,
        "pairwise_concordance": pairwise_concordance,
    }

    out_file = Path("results/item_level_panel_analysis.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(analysis_summary, f, indent=2)

    return analysis_summary


if __name__ == "__main__":
    reactivity_res = run_reactivity_benchmark(seed=42)
    item_analysis_res = run_item_level_panel_analysis()
