"""Sprint S14: Contextual Calibration on Visible Controls.

Tests whether subtractive contextual calibration (z_evidence - z_content_free)
salvages arbitrary-label forced-choice readout on 20 visible deterministic items.

Evaluates:
1. Raw vs. Calibrated visible forced-choice accuracy.
2. Mapping equivariance and semantic agreement across independent label permutations M1 and M2.
3. Jensen-Shannon divergence D_JS (M1 || M2).
4. Fixed-letter bias rate before and after calibration.
"""

import time
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from recurrence.models.recurrent_gemma_adapter import RecurrentGemmaAdapter
from recurrence.tasks.forced_choice_probes import (
    SemanticOption,
    create_forced_choice_mapping,
    compute_js_divergence,
)


PINNED_IT_REVISION = "2766eb5d4264c6c0357803990791f9ab9cd50f8e"

VISIBLE_OPTIONS = [
    SemanticOption("int1", "Interval 1"),
    SemanticOption("int2", "Interval 2"),
    SemanticOption("int3", "Interval 3"),
    SemanticOption("int4", "Interval 4"),
    SemanticOption("none", "No anomaly occurred"),
]


def build_20_visible_items() -> List[Dict[str, Any]]:
    """Build 20 deterministic visible test items (4 per class)."""
    items = []
    classes = ["int1", "int2", "int3", "int4", "none"]
    descriptors = [
        ("anomaly detected", "normal"),
        ("sensor disturbance flagged", "clear"),
        ("state divergence observed", "baseline"),
        ("irregular transmission noted", "standard"),
    ]

    for true_c in classes:
        for var_idx, (flag_str, normal_str) in enumerate(descriptors):
            intervals = []
            for i in range(1, 5):
                c_key = f"int{i}"
                if true_c == c_key:
                    intervals.append(f"[INTERVAL {i}] {flag_str}")
                else:
                    intervals.append(f"[INTERVAL {i}] {normal_str}")
            record_text = "\n".join(intervals)

            items.append({
                "item_id": f"{true_c}_var{var_idx+1}",
                "true_key": true_c,
                "record_text": record_text,
                "flag_str": flag_str,
                "normal_str": normal_str,
            })
    return items


def format_evidence_prompt(mapping, record_text: str, tokenizer) -> str:
    lines = [
        "Task: Review the sequential processing record and identify which interval contained the anomaly.",
        "",
        "Record:",
        record_text.strip(),
        "",
        "Which interval contained the anomaly?",
    ]
    for letter in sorted(mapping.letters):
        key = mapping.label_to_key[letter]
        opt = next(o for o in mapping.options if o.key == key)
        lines.append(f"{letter}) {opt.description}")
    lines.append("")
    lines.append("Answer with only the single option letter:")
    raw = "\n".join(lines)
    msgs = [{"role": "user", "content": raw}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def format_content_free_prompt(mapping, tokenizer) -> str:
    lines = [
        "Task: Review the sequential processing record and identify which interval contained the anomaly.",
        "",
        "Record:",
        "[INTERVAL 1] neutral stream\n[INTERVAL 2] neutral stream\n[INTERVAL 3] neutral stream\n[INTERVAL 4] neutral stream",
        "",
        "No diagnostic evidence about which interval was modified is provided.",
        "",
        "Which interval contained the anomaly?",
    ]
    for letter in sorted(mapping.letters):
        key = mapping.label_to_key[letter]
        opt = next(o for o in mapping.options if o.key == key)
        lines.append(f"{letter}) {opt.description}")
    lines.append("")
    lines.append("Answer with only the single option letter:")
    raw = "\n".join(lines)
    msgs = [{"role": "user", "content": raw}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def evaluate_single_mapping(adapter, mapping, evidence_prompt, content_free_prompt):
    # 1. Evidence logits
    toks_ev = adapter.tokenizer.encode(evidence_prompt, add_special_tokens=False)
    out_ev, _ = adapter.encode_sequence(toks_ev, step_by_step=False, return_logits=True, logits_to_keep=1)
    lg_ev_full = out_ev[0].float()
    lg_ev = torch.stack([lg_ev_full[mapping.label_token_ids[l]] for l in mapping.letters])

    # 2. Content-free calibration logits
    toks_cf = adapter.tokenizer.encode(content_free_prompt, add_special_tokens=False)
    out_cf, _ = adapter.encode_sequence(toks_cf, step_by_step=False, return_logits=True, logits_to_keep=1)
    lg_cf_full = out_cf[0].float()
    lg_cf = torch.stack([lg_cf_full[mapping.label_token_ids[l]] for l in mapping.letters])

    # Raw scoring
    p_raw = F.softmax(lg_ev, dim=-1).tolist()
    raw_best_l = max(mapping.letters, key=lambda l: p_raw[mapping.letters.index(l)])
    raw_best_k = mapping.label_to_key[raw_best_l]
    raw_sem_p = {mapping.label_to_key[l]: p for l, p in zip(mapping.letters, p_raw)}

    # Subtractive calibration: z_cal = z_ev - z_cf
    lg_cal = lg_ev - lg_cf
    p_cal = F.softmax(lg_cal, dim=-1).tolist()
    cal_best_l = max(mapping.letters, key=lambda l: p_cal[mapping.letters.index(l)])
    cal_best_k = mapping.label_to_key[cal_best_l]
    cal_sem_p = {mapping.label_to_key[l]: p for l, p in zip(mapping.letters, p_cal)}

    return {
        "raw_best_letter": raw_best_l,
        "raw_best_key": raw_best_k,
        "raw_sem_probs": raw_sem_p,
        "cal_best_letter": cal_best_l,
        "cal_best_key": cal_best_k,
        "cal_sem_probs": cal_sem_p,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "google/recurrentgemma-2b-it"
    print(f"\n" + "=" * 95)
    print(f"CONTEXTUAL CALIBRATION ON VISIBLE CONTROLS: {model_id}")
    print("=" * 95)

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=PINNED_IT_REVISION)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=PINNED_IT_REVISION, torch_dtype=torch.bfloat16).to(device)
    adapter = RecurrentGemmaAdapter(model=model, tokenizer=tokenizer, device=device, dtype=torch.bfloat16)
    print(f"Model loaded and verified in {time.perf_counter() - t0:.2f}s")

    items = build_20_visible_items()
    print(f"Loaded {len(items)} deterministic visible test items across 5 classes.")

    results = []

    for idx, item in enumerate(items):
        true_k = item["true_key"]

        # Mapping 1 (seed = 100 + idx)
        map1 = create_forced_choice_mapping(VISIBLE_OPTIONS, tokenizer, seed=100 + idx)
        ev_prompt_1 = format_evidence_prompt(map1, item["record_text"], tokenizer)
        cf_prompt_1 = format_content_free_prompt(map1, tokenizer)
        eval_1 = evaluate_single_mapping(adapter, map1, ev_prompt_1, cf_prompt_1)

        # Mapping 2 (seed = 200 + idx)
        map2 = create_forced_choice_mapping(VISIBLE_OPTIONS, tokenizer, seed=200 + idx)
        ev_prompt_2 = format_evidence_prompt(map2, item["record_text"], tokenizer)
        cf_prompt_2 = format_content_free_prompt(map2, tokenizer)
        eval_2 = evaluate_single_mapping(adapter, map2, ev_prompt_2, cf_prompt_2)

        # Metrics
        raw_acc_1 = (eval_1["raw_best_key"] == true_k)
        raw_acc_2 = (eval_2["raw_best_key"] == true_k)
        cal_acc_1 = (eval_1["cal_best_key"] == true_k)
        cal_acc_2 = (eval_2["cal_best_key"] == true_k)

        raw_agree = (eval_1["raw_best_key"] == eval_2["raw_best_key"])
        cal_agree = (eval_1["cal_best_key"] == eval_2["cal_best_key"])

        raw_js = compute_js_divergence(eval_1["raw_sem_probs"], eval_2["raw_sem_probs"])
        cal_js = compute_js_divergence(eval_1["cal_sem_probs"], eval_2["cal_sem_probs"])

        rec = {
            "item_id": item["item_id"],
            "true_key": true_k,
            "raw_m1_pred": eval_1["raw_best_key"],
            "raw_m2_pred": eval_2["raw_best_key"],
            "raw_m1_letter": eval_1["raw_best_letter"],
            "raw_m2_letter": eval_2["raw_best_letter"],
            "raw_acc_1": raw_acc_1,
            "raw_acc_2": raw_acc_2,
            "raw_agree": raw_agree,
            "raw_js": raw_js,
            "cal_m1_pred": eval_1["cal_best_key"],
            "cal_m2_pred": eval_2["cal_best_key"],
            "cal_m1_letter": eval_1["cal_best_letter"],
            "cal_m2_letter": eval_2["cal_best_letter"],
            "cal_acc_1": cal_acc_1,
            "cal_acc_2": cal_acc_2,
            "cal_agree": cal_agree,
            "cal_js": cal_js,
        }
        results.append(rec)
        print(f"Item {idx+1:02d} ({true_k:<5}) | Raw M1: {eval_1['raw_best_key']:<5} (Acc={raw_acc_1!s:<5}) | Cal M1: {eval_1['cal_best_key']:<5} (Acc={cal_acc_1!s:<5}) | Cal M2: {eval_2['cal_best_key']:<5} | Cal JS: {cal_js:.4f}", flush=True)

    # Compute aggregate summary
    n = len(results)
    raw_acc1_mean = sum(1 for r in results if r["raw_acc_1"]) / n
    raw_acc2_mean = sum(1 for r in results if r["raw_acc_2"]) / n
    cal_acc1_mean = sum(1 for r in results if r["cal_acc_1"]) / n
    cal_acc2_mean = sum(1 for r in results if r["cal_acc_2"]) / n

    raw_agree_mean = sum(1 for r in results if r["raw_agree"]) / n
    cal_agree_mean = sum(1 for r in results if r["cal_agree"]) / n

    raw_js_mean = sum(r["raw_js"] for r in results) / n
    cal_js_mean = sum(r["cal_js"] for r in results) / n

    raw_fixed_letter_rate = sum(1 for r in results if r["raw_m1_letter"] == r["raw_m2_letter"]) / n
    cal_fixed_letter_rate = sum(1 for r in results if r["cal_m1_letter"] == r["cal_m2_letter"]) / n

    print("\n" + "=" * 95)
    print("CONTEXTUAL CALIBRATION SUMMARY (N = 20 Visible Test Items)")
    print("=" * 95)
    print(f"{'Metric':<36} | {'Raw (Uncalibrated)':<22} | {'Calibrated (z_ev - z_cf)':<24}")
    print("-" * 95)
    print(f"{'M1 Visible Accuracy':<36} | {raw_acc1_mean*100:6.2f}%                 | {cal_acc1_mean*100:6.2f}%")
    print(f"{'M2 Visible Accuracy':<36} | {raw_acc2_mean*100:6.2f}%                 | {cal_acc2_mean*100:6.2f}%")
    print(f"{'Semantic Agreement (M1 <-> M2)':<36} | {raw_agree_mean*100:6.2f}%                 | {cal_agree_mean*100:6.2f}%")
    print(f"{'Fixed-Letter Response Rate':<36} | {raw_fixed_letter_rate*100:6.2f}%                 | {cal_fixed_letter_rate*100:6.2f}%")
    print(f"{'Mean Jensen-Shannon Divergence':<36} | {raw_js_mean:6.4f}                 | {cal_js_mean:6.4f}")
    print("=" * 95)

    # Check Gate
    print("\nREADOUT CALIBRATION GATE VERDICT:")
    if cal_acc1_mean >= 0.80 and cal_js_mean <= 0.15:
        print("  [PASSED] Subtractive contextual calibration salvages arbitrary-label forced-choice readout.")
    else:
        print(f"  [FAILED] Calibrated visible accuracy ({cal_acc1_mean*100:.1f}%) and JS divergence ({cal_js_mean:.4f}) fail the measurement-validity gate.")
        print("  [RECOMMENDATION] Formally retire arbitrary-label temporal localization; pivot to direct-word prior-intention (S14.0C).")
    print("=" * 95)

    # Save artifact
    out_dir = Path("results") / "e14_latent_metacognition" / "readout_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "visible_contextual_calibration_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "model_id": model_id,
            "pinned_revision": PINNED_IT_REVISION,
            "raw_acc_m1": raw_acc1_mean,
            "raw_acc_m2": raw_acc2_mean,
            "cal_acc_m1": cal_acc1_mean,
            "cal_acc_m2": cal_acc2_mean,
            "raw_agree": raw_agree_mean,
            "cal_agree": cal_agree_mean,
            "raw_js": raw_js_mean,
            "cal_js": cal_js_mean,
            "raw_fixed_letter_rate": raw_fixed_letter_rate,
            "cal_fixed_letter_rate": cal_fixed_letter_rate,
            "items": results,
        }, f, indent=2)
    print(f"Saved calibration artifact to {out_file}\n")


if __name__ == "__main__":
    main()
