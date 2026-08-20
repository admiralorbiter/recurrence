"""Sprint S14 Stage S14.0A Runner: Secret Internal-Intervention Temporal Localization.

Usage:
  python experiments/e14_latent_metacognition/run_s14_0a_temporal_localization.py --stage a1
  python experiments/e14_latent_metacognition/run_s14_0a_temporal_localization.py --stage a2
  python experiments/e14_latent_metacognition/run_s14_0a_temporal_localization.py --stage a3
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from recurrence.models.recurrent_gemma_adapter import RecurrentGemmaAdapter
from recurrence.tasks.impulse_stimuli import get_filler_tokens_for_regime, build_audited_vocabulary_pool
from recurrence.tasks.specificity_microscope import build_microscope_pairs, MicroscopePair
from recurrence.tasks.temporal_localization import (
    TEMPORAL_LOCALIZATION_OPTIONS,
    LocalizationTrialResult,
    generate_neutral_intervals,
    execute_temporal_localization_trial,
)


def select_scout_pairs(all_pairs: List[MicroscopePair]) -> List[MicroscopePair]:
    family_firsts = {}
    for p in all_pairs:
        if p.family_id not in family_firsts:
            family_firsts[p.family_id] = p
    return list(family_firsts.values())


def print_confusion_matrix(trials: List[LocalizationTrialResult], label: str = "Target M1"):
    categories = ["int1", "int2", "int3", "int4", "none"]
    matrix = {true_c: {pred_c: 0 for pred_c in categories} for true_c in categories}
    totals = {true_c: 0 for true_c in categories}

    for t in trials:
        true_c = t.target_interval
        pred_c = t.target_m1_pred if "M1" in label else t.target_m2_pred
        matrix[true_c][pred_c] += 1
        totals[true_c] += 1

    print(f"\n{label} Empirical 5x5 Confusion Matrix (True Interval -> Predicted Interval):")
    print(f"{'True Class':<12} | {'int1':<8} | {'int2':<8} | {'int3':<8} | {'int4':<8} | {'none':<8} | {'Total':<6}")
    print("-" * 75)
    for true_c in categories:
        row_str = " | ".join(f"{matrix[true_c][pred_c]:<8}" for pred_c in categories)
        print(f"{true_c:<12} | {row_str} | {totals[true_c]:<6}")


def analyze_stage_results(trials: List[LocalizationTrialResult], stage_name: str) -> Dict[str, Any]:
    n_total = len(trials)
    target_m1_acc = sum(1 for t in trials if t.target_m1_acc) / n_total
    target_m2_acc = sum(1 for t in trials if t.target_m2_acc) / n_total
    observer_m1_acc = sum(1 for t in trials if t.observer_m1_acc) / n_total
    observer_m2_acc = sum(1 for t in trials if t.observer_m2_acc) / n_total

    mean_target_js = float(np.mean([t.target_js_div for t in trials]))
    mean_observer_js = float(np.mean([t.observer_js_div for t in trials]))
    target_agreement = sum(1 for t in trials if t.target_semantic_agreement) / n_total

    mean_pai_m1 = float(np.mean([t.pai_m1 for t in trials]))
    mean_pai_m2 = float(np.mean([t.pai_m2 for t in trials]))

    # Anomaly Detection (Perturbed vs Sham)
    # Target correctly predicted non-"none" when true != "none", or predicted "none" when true == "none"
    det_acc_m1 = sum(1 for t in trials if (t.target_interval != "none" and t.target_m1_pred != "none") or (t.target_interval == "none" and t.target_m1_pred == "none")) / n_total

    print("\n" + "=" * 95)
    print(f"STAGE {stage_name.upper()} RESULTS SUMMARY (N = {n_total} trials)")
    print("=" * 95)
    print(f"Target Accuracy (M1):              {target_m1_acc*100:6.2f}% (Chance baseline: 20.0%)")
    print(f"Target Accuracy (M2):              {target_m2_acc*100:6.2f}%")
    print(f"Public Replay Observer Acc (M1):   {observer_m1_acc*100:6.2f}%")
    print(f"Public Replay Observer Acc (M2):   {observer_m2_acc*100:6.2f}%")
    print(f"Privileged Access Index PAI (M1):  {mean_pai_m1:+6.4f} (Target - Observer)")
    print(f"Privileged Access Index PAI (M2):  {mean_pai_m2:+6.4f}")
    print(f"Semantic Agreement (M1 vs M2):     {target_agreement*100:6.2f}%")
    print(f"Mean Target JS Divergence:         {mean_target_js:6.4f}")
    print(f"Mean Observer JS Divergence:       {mean_observer_js:6.4f}")
    print(f"Anomaly Detection (Any vs Sham):   {det_acc_m1*100:6.2f}%")
    print("=" * 95)

    print_confusion_matrix(trials, label="Target M1")

    return {
        "n_total": n_total,
        "target_m1_acc": target_m1_acc,
        "target_m2_acc": target_m2_acc,
        "observer_m1_acc": observer_m1_acc,
        "observer_m2_acc": observer_m2_acc,
        "pai_m1": mean_pai_m1,
        "pai_m2": mean_pai_m2,
        "target_agreement": target_agreement,
        "mean_target_js": mean_target_js,
        "det_acc_m1": det_acc_m1,
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Run S14.0A Temporal Localization Experiments.")
    parser.add_argument("--stage", type=str, default="a1", choices=["a0", "a1", "a2", "a3"], help="Experiment stage (a1=whole_state 1-pair, a2=rglru 1-pair, a3=4-pair scout)")
    parser.add_argument("--model_id", type=str, default="google/recurrentgemma-2b-it", help="Model ID (default: google/recurrentgemma-2b-it)")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device (cuda/cpu)")
    parser.add_argument("--repeats_per_cell", type=int, default=4, help="Number of repetitions per condition cell (default: 4)")
    args = parser.parse_args()

    out_dir = Path("results") / "e14_latent_metacognition" / "s14_0a_localization"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[S14.0A Runner] Initializing Stage '{args.stage}' on {args.model_id} (device={args.device})...", flush=True)
    t_start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(args.device)
    adapter = RecurrentGemmaAdapter(model=model, tokenizer=tokenizer, device=args.device, dtype=torch.bfloat16)
    print(f"Model loaded and adapted in {time.perf_counter() - t_start:.2f}s", flush=True)

    audited_pool, _ = build_audited_vocabulary_pool(tokenizer)
    all_pairs = build_microscope_pairs()
    scout_pairs = select_scout_pairs(all_pairs)

    if args.stage in ("a0", "a1", "a2"):
        target_pairs = [scout_pairs[0]]  # Single canonical pair: marked_object_p01
    else:
        target_pairs = scout_pairs  # All 4 canonical pairs

    condition_type = "whole_state" if args.stage in ("a0", "a1") else ("rglru_only" if args.stage == "a2" else "rglru_only")
    conditions = ["int1", "int2", "int3", "int4", "none"]

    trials_list: List[LocalizationTrialResult] = []

    for p_idx, pair in enumerate(target_pairs):
        global_pair_idx = all_pairs.index(pair)
        cur_seed = 42 + global_pair_idx * 100
        print(f"\n[S14.0A] Processing Pair {p_idx+1}/{len(target_pairs)} ({pair.pair_id}, Family: {pair.family_id}, Seed: {cur_seed})...", flush=True)

        toks_prompt_a = tokenizer.encode(pair.prompt_a, add_special_tokens=False)
        toks_prompt_b = tokenizer.encode(pair.prompt_b, add_special_tokens=False)
        tok_a_id = tokenizer.encode(pair.target_a, add_special_tokens=False)[-1]
        tok_b_id = tokenizer.encode(pair.target_b, add_special_tokens=False)[-1]

        pair_excluded = set(toks_prompt_a + toks_prompt_b + [tok_a_id, tok_b_id])

        # Canonical B=1 4096-token Common Origin Preparation
        print(f"  Preparing canonical B=1 4,096-token origin states for Recipient (A) and Donor (B)...", flush=True)
        _, s_rec_0 = adapter.encode_sequence(toks_prompt_a, step_by_step=False, return_logits=False)
        _, s_don_0 = adapter.encode_sequence(toks_prompt_b, step_by_step=False, return_logits=False)

        filler = get_filler_tokens_for_regime("random", length=4096, seed=cur_seed, audited_pool=audited_pool, tokenizer=tokenizer, excluded_token_ids=pair_excluded)
        for i in range(0, 4096, 512):
            chunk = filler[i : i + 512]
            _, s_rec_0 = adapter.encode_sequence(chunk, initial_snapshot=s_rec_0, step_by_step=False, return_logits=False)
            _, s_don_0 = adapter.encode_sequence(chunk, initial_snapshot=s_don_0, step_by_step=False, return_logits=False)

        # Run Trials across conditions x repeats
        total_cells = len(conditions) * args.repeats_per_cell
        cell_count = 0

        for cond_target in conditions:
            cond_label = "sham" if cond_target == "none" else condition_type
            for rep in range(args.repeats_per_cell):
                trial_seed = cur_seed + 1000 + cell_count * 10
                cell_count += 1

                # Generate 4 neutral intervals of 64 tokens each
                intervals = generate_neutral_intervals(
                    tokenizer=tokenizer,
                    audited_pool=audited_pool,
                    seed=trial_seed,
                    num_intervals=4,
                    tokens_per_interval=64,
                    excluded_token_ids=pair_excluded,
                )

                res = execute_temporal_localization_trial(
                    adapter=adapter,
                    s_recipient_0=s_rec_0,
                    s_donor_0=s_don_0,
                    intervals=intervals,
                    condition=cond_label,
                    target_interval=cond_target,
                    pair=pair,
                    seed_mapping_1=trial_seed + 1,
                    seed_mapping_2=trial_seed + 2,
                    use_chat_template=True,
                )

                trials_list.append(res)
                print(f"  Trial {cell_count:02d}/{total_cells} (True: {cond_target:<4}) | Target M1: {res.target_m1_pred:<4} (Acc: {res.target_m1_acc}) | Obs M1: {res.observer_m1_pred:<4} (Acc: {res.observer_m1_acc}) | PAI: {res.pai_m1:+4.1f} | JS: {res.target_js_div:.4f}", flush=True)

    # Analyze and Output Summary
    summary = analyze_stage_results(trials_list, stage_name=args.stage)

    out_file = out_dir / f"s14_0a_{args.stage}_results.json"
    serializable_trials = [
        {
            "pair_id": t.pair_id,
            "family_id": t.family_id,
            "condition": t.condition,
            "target_interval": t.target_interval,
            "channels": t.channels,
            "target_m1_pred": t.target_m1_pred,
            "target_m2_pred": t.target_m2_pred,
            "target_m1_acc": t.target_m1_acc,
            "target_m2_acc": t.target_m2_acc,
            "target_semantic_agreement": t.target_semantic_agreement,
            "target_js_div": t.target_js_div,
            "observer_m1_pred": t.observer_m1_pred,
            "observer_m2_pred": t.observer_m2_pred,
            "observer_m1_acc": t.observer_m1_acc,
            "observer_m2_acc": t.observer_m2_acc,
            "pai_m1": t.pai_m1,
            "pai_m2": t.pai_m2,
        }
        for t in trials_list
    ]

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "trials": serializable_trials}, f, indent=2)

    print(f"\n[S14.0A] Results successfully saved to {out_file}", flush=True)


if __name__ == "__main__":
    main()
