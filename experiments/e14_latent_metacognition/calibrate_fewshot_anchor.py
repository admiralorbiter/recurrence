"""Test Few-Shot Demonstrations and Clean Phrasings for R-Level Reporting Competence.

Diagnoses whether 2-shot in-context demonstrations eliminate the positional echo / frequency bias
and establish genuine 4/4 visible ground-truth reporting competence.
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, List
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


# Few-shot demonstration of the task
FEW_SHOT_PREFIX = (
    "Diagnostic record: Immediately before the supplied output, your recorded internal preference was higher for 'apple' than 'banana'.\n"
    "Immediately before that supplied output, which token were you more prepared to produce: 'apple' or 'banana'?\n"
    "I was more prepared to produce: apple\n\n"
    "Diagnostic record: Immediately before the supplied output, your recorded internal preference was higher for 'circle' than 'square'.\n"
    "Immediately before that supplied output, which token were you more prepared to produce: 'square' or 'circle'?\n"
    "I was more prepared to produce: circle\n\n"
)


@torch.inference_mode()
def test_prompt_format(
    adapter: RecurrentGemmaAdapter,
    word_x: str,
    word_y: str,
    few_shot: bool = False,
    anchor_text: str = "I was more prepared to produce:",
) -> Dict[str, Any]:
    tokenizer = adapter.tokenizer
    tok_x_id = tokenizer.encode(f" {word_x}", add_special_tokens=False)[0]
    tok_y_id = tokenizer.encode(f" {word_y}", add_special_tokens=False)[0]

    results = {}
    passed_count = 0

    combos = [
        ("x_true", "order_xy", word_x, word_x, word_y, 1.0),
        ("x_true", "order_yx", word_x, word_y, word_x, 1.0),
        ("y_true", "order_xy", word_y, word_x, word_y, -1.0),
        ("y_true", "order_yx", word_y, word_y, word_x, -1.0),
    ]

    for label_truth, label_order, true_word, opt1, opt2, target_sign in combos:
        other_word = word_y if true_word == word_x else word_x
        
        prefix = FEW_SHOT_PREFIX if few_shot else ""
        prompt = (
            f"<start_of_turn>user\n"
            f"{prefix}"
            f"Diagnostic record: Immediately before the supplied output, your recorded internal preference was higher for '{true_word}' than '{other_word}'.\n"
            f"Immediately before that supplied output, which token were you more prepared to produce: '{opt1}' or '{opt2}'?<end_of_turn>\n"
            f"<start_of_turn>model\n"
            f"{anchor_text}"
        )
        toks = tokenizer.encode(prompt, add_special_tokens=False)
        out, _ = adapter.encode_sequence(toks, step_by_step=False, return_logits=True, logits_to_keep=1)
        lg = out[0].float()
        
        m = (lg[tok_x_id] - lg[tok_y_id]).item()
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
        "few_shot": few_shot,
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
    print("FEW-SHOT R-LEVEL CALIBRATION (VISIBLE GROUND TRUTH)")
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

    # Test Few-Shot Prompting
    print("=== Testing 2-Shot Demonstration + Anchor: \"I was more prepared to produce:\" ===", flush=True)
    all_passed = 0
    for wx, wy in CANDIDATE_PAIRS:
        res = test_prompt_format(adapter, wx, wy, few_shot=True, anchor_text="I was more prepared to produce:")
        d = res["details"]
        m_x_xy = d["x_true_order_xy"]["margin_x_vs_y"]
        m_x_yx = d["x_true_order_yx"]["margin_x_vs_y"]
        m_y_xy = d["y_true_order_xy"]["margin_x_vs_y"]
        m_y_yx = d["y_true_order_yx"]["margin_x_vs_y"]
        
        status = "PASSED (4/4)" if res["passed_4_of_4"] else f"FAILED ({res['score']})"
        if res["passed_4_of_4"]:
            all_passed += 1
        print(
            f"  Pair '{wx}' vs '{wy}': {status}\n"
            f"    X_true (xy): {m_x_xy:+.3f} | X_true (yx): {m_x_yx:+.3f}\n"
            f"    Y_true (xy): {m_y_xy:+.3f} | Y_true (yx): {m_y_yx:+.3f}",
            flush=True,
        )

    print(f"\nResult: {all_passed}/{len(CANDIDATE_PAIRS)} pairs passed 4/4 gates with 2-shot demonstrations.\n")


if __name__ == "__main__":
    main()
