"""Sprint S14 Stage S14.0: Preflight Substrate Bring-Up & Forced-Choice Verification.

Tests both `google/recurrentgemma-2b` (base) and `google/recurrentgemma-2b-it` (instruction-tuned):
1. Snapshot / restore determinism invariants.
2. Token-clock single-token forced-choice compliance.
3. Randomized answer-order mapping stability and mapping equivariance (D_JS).
4. Fixed-letter bias detection.
"""

import time
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

from recurrence.models.recurrent_gemma_adapter import RecurrentGemmaAdapter
from recurrence.tasks.forced_choice_probes import (
    SemanticOption,
    create_forced_choice_mapping,
    format_forced_choice_prompt,
    score_forced_choice_prompt,
    evaluate_mapping_equivariance,
)


def build_synthetic_preflight_items():
    """Build simple visible forced-choice items to verify prompt understanding and label tracking."""
    return [
        {
            "preamble": "You are a precise diagnostic system analyzing sequential intervals.",
            "question": "Which of the following colors corresponds to cobalt?",
            "options": [
                SemanticOption("amber", "The color amber (yellow-orange)"),
                SemanticOption("cobalt", "The color cobalt (deep blue)"),
                SemanticOption("garnet", "The color garnet (deep red)"),
                SemanticOption("zircon", "The color zircon (light blue/grey)"),
                SemanticOption("none", "None of the above"),
            ],
            "true_key": "cobalt",
        },
        {
            "preamble": "Review the following interval record: [Interval 1: normal] [Interval 2: anomaly detected] [Interval 3: normal] [Interval 4: normal].",
            "question": "Which interval contained the anomaly?",
            "options": [
                SemanticOption("int1", "Interval 1"),
                SemanticOption("int2", "Interval 2"),
                SemanticOption("int3", "Interval 3"),
                SemanticOption("int4", "Interval 4"),
                SemanticOption("none", "No anomaly occurred"),
            ],
            "true_key": "int2",
        },
        {
            "preamble": "Review the following interval record: [Interval 1: normal] [Interval 2: normal] [Interval 3: normal] [Interval 4: normal].",
            "question": "Which interval was flagged as irregular?",
            "options": [
                SemanticOption("int1", "Interval 1"),
                SemanticOption("int2", "Interval 2"),
                SemanticOption("int3", "Interval 3"),
                SemanticOption("int4", "Interval 4"),
                SemanticOption("none", "No irregular interval was flagged"),
            ],
            "true_key": "none",
        },
        {
            "preamble": "The sensor sequence was recorded as: Segment 1 (clear), Segment 2 (clear), Segment 3 (clear), Segment 4 (disturbed).",
            "question": "Which segment experienced a disturbance?",
            "options": [
                SemanticOption("int1", "Segment 1"),
                SemanticOption("int2", "Segment 2"),
                SemanticOption("int3", "Segment 3"),
                SemanticOption("int4", "Segment 4"),
                SemanticOption("none", "No disturbance observed"),
            ],
            "true_key": "int4",
        },
    ]


@torch.inference_mode()
def run_preflight_for_model(model_id: str, device: str = "cuda") -> dict:
    print(f"\n" + "=" * 90)
    print(f"RUNNING S14.0 PREFLIGHT FOR: {model_id}")
    print("=" * 90)

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    adapter = RecurrentGemmaAdapter(model=model, tokenizer=tokenizer, device=device, dtype=torch.bfloat16)
    print(f"Model loaded and adapted in {time.perf_counter() - t0:.2f}s")

    items = build_synthetic_preflight_items()
    results = []

    for idx, item in enumerate(items):
        options = item["options"]
        true_key = item["true_key"]

        # Run with initial zero state
        s0 = adapter.create_canonical_initial_state(batch_size=1)

        # Test mapping equivariance with two seeds (100 and 200)
        res = evaluate_mapping_equivariance(
            adapter=adapter,
            snapshot=s0,
            options=options,
            preamble=item["preamble"],
            question=item["question"],
            seed_1=100 + idx,
            seed_2=200 + idx,
            true_key=true_key,
        )

        results.append(res)
        print(f"Item {idx+1} ({true_key:<7}) | M1: {res['m1_predicted_key']:<7} (L: {res['m1_predicted_letter']}) | M2: {res['m2_predicted_key']:<7} (L: {res['m2_predicted_letter']}) | JS Div: {res['js_divergence']:.4f} | M1 Acc: {res['m1_acc']} | M2 Acc: {res['m2_acc']}")

    m1_acc = sum(1 for r in results if r["m1_acc"]) / len(results)
    m2_acc = sum(1 for r in results if r["m2_acc"]) / len(results)
    mean_js = sum(r["js_divergence"] for r in results) / len(results)
    semantic_agreement_rate = sum(1 for r in results if r["semantic_agreement"]) / len(results)
    fixed_letter_rate = sum(1 for r in results if r["fixed_letter_chosen"]) / len(results)

    print("-" * 90)
    print(f"Summary for {model_id}:")
    print(f"  Accuracy (M1): {m1_acc*100:.1f}% | Accuracy (M2): {m2_acc*100:.1f}%")
    print(f"  Semantic Agreement Across Remappings: {semantic_agreement_rate*100:.1f}%")
    print(f"  Fixed-Letter Bias Rate: {fixed_letter_rate*100:.1f}%")
    print(f"  Mean Mapping Jensen-Shannon Divergence: {mean_js:.4f}")
    print("=" * 90)

    # Free memory before next model
    del adapter
    del model
    torch.cuda.empty_cache()

    return {
        "model_id": model_id,
        "m1_acc": m1_acc,
        "m2_acc": m2_acc,
        "semantic_agreement_rate": semantic_agreement_rate,
        "fixed_letter_rate": fixed_letter_rate,
        "mean_js": mean_js,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[S14.0 Preflight Bring-Up] Running on device={device} (bfloat16)...")

    # 1. Base Model Check
    res_base = run_preflight_for_model("google/recurrentgemma-2b", device=device)

    # 2. Instruction-Tuned Model Check
    res_it = run_preflight_for_model("google/recurrentgemma-2b-it", device=device)

    print("\n" + "=" * 90)
    print("PREFLIGHT DECISION GATE COMPARISON")
    print("=" * 90)
    print(f"{'Model Substrate':<32} | {'M1 Accuracy':<12} | {'M2 Accuracy':<12} | {'Semantic Agreement':<20} | {'Mean JS Div':<12}")
    print("-" * 90)
    print(f"{res_base['model_id']:<32} | {res_base['m1_acc']*100:<11.1f}% | {res_base['m2_acc']*100:<11.1f}% | {res_base['semantic_agreement_rate']*100:<19.1f}% | {res_base['mean_js']:<12.4f}")
    print(f"{res_it['model_id']:<32} | {res_it['m1_acc']*100:<11.1f}% | {res_it['m2_acc']*100:<11.1f}% | {res_it['semantic_agreement_rate']*100:<19.1f}% | {res_it['mean_js']:<12.4f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
