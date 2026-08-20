"""Verify Balanced Order Permutation (BOP) for R-level calibration.

Tests whether averaging logit margins over forward (x, y) and reversed (y, x) question orderings
completely cancels out the positional bias and achieves 100% visible reporting accuracy.

Formula:
  M_calibrated = (M(order xy) - M(order yx)) / 2
  Target: M_calibrated > 0 when X is true, M_calibrated < 0 when Y is true.
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from recurrence.models.recurrent_gemma_adapter import RecurrentGemmaAdapter


PINNED_IT_REVISION = "2766eb5d4264c6c0357803990791f9ab9cd50f8e"

PAIRS_TO_TEST = [
    ("alkali", "antonio"),
    ("achievement", "effort"),
    ("crystal", "granite"),
    ("copper", "silver"),
    ("amber", "cobalt"),
]

TEMPLATE_STR = (
    "<start_of_turn>user\n"
    "Record: Prior token preference was '{true_word}' (not '{other_word}').\n"
    "Which token was preferred: '{opt1}' or '{opt2}'?<end_of_turn>\n"
    "<start_of_turn>model\n"
    "Preferred token:"
)


@torch.inference_mode()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "google/recurrentgemma-2b-it"
    print("\n" + "=" * 115)
    print("BALANCED ORDER PERMUTATION (BOP) R-LEVEL CALIBRATION")
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

    all_passed = 0

    for wx, wy in PAIRS_TO_TEST:
        tok_x_id = tokenizer.encode(f" {wx}", add_special_tokens=False)[0]
        tok_y_id = tokenizer.encode(f" {wy}", add_special_tokens=False)[0]

        # 1. When X is true: evaluate both (xy) and (yx) orderings
        prompt_x_xy = TEMPLATE_STR.format(true_word=wx, other_word=wy, opt1=wx, opt2=wy)
        prompt_x_yx = TEMPLATE_STR.format(true_word=wx, other_word=wy, opt1=wy, opt2=wx)

        out_x_xy, _ = adapter.encode_sequence(tokenizer.encode(prompt_x_xy, add_special_tokens=False), step_by_step=False, return_logits=True, logits_to_keep=1)
        out_x_yx, _ = adapter.encode_sequence(tokenizer.encode(prompt_x_yx, add_special_tokens=False), step_by_step=False, return_logits=True, logits_to_keep=1)

        m_x_xy = (out_x_xy[0].float()[tok_x_id] - out_x_xy[0].float()[tok_y_id]).item()
        m_x_yx = (out_x_yx[0].float()[tok_x_id] - out_x_yx[0].float()[tok_y_id]).item()
        m_x_calibrated = (m_x_xy + m_x_yx) / 2.0

        # 2. When Y is true: evaluate both (xy) and (yx) orderings
        prompt_y_xy = TEMPLATE_STR.format(true_word=wy, other_word=wx, opt1=wx, opt2=wy)
        prompt_y_yx = TEMPLATE_STR.format(true_word=wy, other_word=wx, opt1=wy, opt2=wx)

        out_y_xy, _ = adapter.encode_sequence(tokenizer.encode(prompt_y_xy, add_special_tokens=False), step_by_step=False, return_logits=True, logits_to_keep=1)
        out_y_yx, _ = adapter.encode_sequence(tokenizer.encode(prompt_y_yx, add_special_tokens=False), step_by_step=False, return_logits=True, logits_to_keep=1)

        m_y_xy = (out_y_xy[0].float()[tok_x_id] - out_y_xy[0].float()[tok_y_id]).item()
        m_y_yx = (out_y_yx[0].float()[tok_x_id] - out_y_yx[0].float()[tok_y_id]).item()
        m_y_calibrated = (m_y_xy + m_y_yx) / 2.0

        correct_x = m_x_calibrated > 0
        correct_y = m_y_calibrated < 0
        passed = correct_x and correct_y

        if passed:
            all_passed += 1
            status = "PASSED"
        else:
            status = "FAILED"

        print(
            f"  Pair '{wx}' vs '{wy}': [{status}]\n"
            f"    When X is True -> Calibrated Margin: {m_x_calibrated:+.3f} (Raw: xy={m_x_xy:+.2f}, yx={m_x_yx:+.2f}) -> Correct: {correct_x}\n"
            f"    When Y is True -> Calibrated Margin: {m_y_calibrated:+.3f} (Raw: xy={m_y_xy:+.2f}, yx={m_y_yx:+.2f}) -> Correct: {correct_y}",
            flush=True,
        )

    print("\n" + "=" * 115)
    print(f"BOP CALIBRATION RESULT: {all_passed}/{len(PAIRS_TO_TEST)} pairs passed 100% visible accuracy.")
    print("=" * 115)


if __name__ == "__main__":
    main()
