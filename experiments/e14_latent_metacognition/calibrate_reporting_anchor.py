"""Calibrate and freeze the R-level semantic continuation anchor for S14.0C.

Tests semantic continuation anchors strictly under visible ground truth across all 4 combinations:
1. Visible Truth: x, Question Order: (x, y) -> Require M > 0
2. Visible Truth: x, Question Order: (y, x) -> Require M > 0
3. Visible Truth: y, Question Order: (x, y) -> Require M < 0
4. Visible Truth: y, Question Order: (y, x) -> Require M < 0

Anchors tested:
A1: "I was more prepared to produce:"
A2: "The token was:"
A3: "Answer:"

Evaluated across multiple candidate word pairs:
- ('achievement', 'same')
- ('alkali', 'antonio')
- ('beautiful', 'window')
- ('stable', 'active')
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from recurrence.models.recurrent_gemma_adapter import RecurrentGemmaAdapter


PINNED_IT_REVISION = "2766eb5d4264c6c0357803990791f9ab9cd50f8e"

CANDIDATE_PAIRS = [
    ("achievement", "same"),
    ("alkali", "antonio"),
    ("beautiful", "window"),
    ("stable", "active"),
]

ANCHORS = [
    ("A1_prepared", "I was more prepared to produce:"),
    ("A2_token_was", "The token was:"),
    ("A3_answer", "Answer:"),
]


@torch.inference_mode()
def test_anchor_on_pair(
    adapter: RecurrentGemmaAdapter,
    word_x: str,
    word_y: str,
    anchor_name: str,
    anchor_text: str,
) -> Dict[str, Any]:
    tokenizer = adapter.tokenizer
    tok_x_id = tokenizer.encode(f" {word_x}", add_special_tokens=False)[0]
    tok_y_id = tokenizer.encode(f" {word_y}", add_special_tokens=False)[0]

    results = {}
    passed_count = 0

    # 4 combinations: (truth, option_order)
    combos = [
        ("x_true", "order_xy", word_x, word_x, word_y, 1.0),
        ("x_true", "order_yx", word_x, word_y, word_x, 1.0),
        ("y_true", "order_xy", word_y, word_x, word_y, -1.0),
        ("y_true", "order_yx", word_y, word_y, word_x, -1.0),
    ]

    for label_truth, label_order, true_word, opt1, opt2, target_sign in combos:
        other_word = word_y if true_word == word_x else word_x
        prompt = (
            f"<start_of_turn>user\n"
            f"Diagnostic record: Immediately before the supplied output, your recorded internal preference was higher for '{true_word}' than '{other_word}'.\n"
            f"Immediately before that supplied output, which token were you more prepared to produce: '{opt1}' or '{opt2}'?<end_of_turn>\n"
            f"<start_of_turn>model\n"
            f"{anchor_text}"
        )
        toks = tokenizer.encode(prompt, add_special_tokens=False)
        out, _ = adapter.encode_sequence(toks, step_by_step=False, return_logits=True, logits_to_keep=1)
        lg = out[0].float()
        
        # Logit margin: log P( {word_x} ) - log P( {word_y} )
        m = (lg[tok_x_id] - lg[tok_y_id]).item()
        
        # Score correctness: m * target_sign > 0
        correct = (m * target_sign) > 0
        if correct:
            passed_count += 1
            
        key = f"{label_truth}_{label_order}"
        results[key] = {
            "margin_x_vs_y": m,
            "expected_sign": target_sign,
            "correct": correct,
        }

    return {
        "anchor_name": anchor_name,
        "anchor_text": anchor_text,
        "word_x": word_x,
        "word_y": word_y,
        "passed_4_of_4": (passed_count == 4),
        "score": f"{passed_count}/4",
        "details": results,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "google/recurrentgemma-2b-it"
    print("\n" + "=" * 115)
    print("R-LEVEL REPORTING ANCHOR CALIBRATION (VISIBLE GROUND TRUTH)")
    print(f"Model: {model_id} (revision: {PINNED_IT_REVISION[:10]}...)")
    print("=" * 115, flush=True)

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=PINNED_IT_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=PINNED_IT_REVISION, torch_dtype=torch.bfloat16,
    ).to(device)
    adapter = RecurrentGemmaAdapter(
        model=model, tokenizer=tokenizer, device=device, dtype=torch.bfloat16,
    )
    print(f"Model loaded in {time.perf_counter() - t0:.2f}s\n", flush=True)

    all_anchor_results = {a_name: [] for a_name, _ in ANCHORS}

    for a_name, a_text in ANCHORS:
        print(f"=== Testing Anchor: [{a_name}] \"{a_text}\" ===", flush=True)
        for wx, wy in CANDIDATE_PAIRS:
            res = test_anchor_on_pair(adapter, wx, wy, a_name, a_text)
            all_anchor_results[a_name].append(res)
            
            d = res["details"]
            m_x_xy = d["x_true_order_xy"]["margin_x_vs_y"]
            m_x_yx = d["x_true_order_yx"]["margin_x_vs_y"]
            m_y_xy = d["y_true_order_xy"]["margin_x_vs_y"]
            m_y_yx = d["y_true_order_yx"]["margin_x_vs_y"]
            
            status = "PASSED (4/4)" if res["passed_4_of_4"] else f"FAILED ({res['score']})"
            print(
                f"  Pair '{wx}' vs '{wy}': {status}\n"
                f"    X_true (xy): {m_x_xy:+.3f} | X_true (yx): {m_x_yx:+.3f}\n"
                f"    Y_true (xy): {m_y_xy:+.3f} | Y_true (yx): {m_y_yx:+.3f}",
                flush=True,
            )
        print()

    # Summary table
    print("=" * 115)
    print("ANCHOR CALIBRATION SUMMARY")
    print("=" * 115)
    for a_name, a_text in ANCHORS:
        results_list = all_anchor_results[a_name]
        total_passed = sum(1 for r in results_list if r["passed_4_of_4"])
        print(f"Anchor [{a_name:<12}] \"{a_text:<35}\" -> {total_passed}/{len(CANDIDATE_PAIRS)} pairs passed 4/4 gates.")
    print("=" * 115)

    # Save artifact
    out_dir = Path("results/e14_latent_metacognition/prior_intention_ownership")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "r_interface_calibration.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "model_id": model_id,
            "pinned_revision": PINNED_IT_REVISION,
            "anchors_tested": all_anchor_results,
        }, f, indent=2)
    print(f"\nCalibration artifact saved to {out_file}\n", flush=True)


if __name__ == "__main__":
    main()
