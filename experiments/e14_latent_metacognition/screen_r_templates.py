"""Screen diverse prompt templates to maximize R-level reporting competence.

Tests various prompt phrasings and anchors on multiple candidate pairs under visible ground truth.
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

TEMPLATES = [
    (
        "T1_explicit_choice",
        (
            "<start_of_turn>user\n"
            "Between '{opt1}' and '{opt2}', the internal state was preparing to output '{true_word}'.\n"
            "Which token was the state preparing to output: '{opt1}' or '{opt2}'?<end_of_turn>\n"
            "<start_of_turn>model\n"
            "The state was preparing to output:"
        )
    ),
    (
        "T2_direct_token",
        (
            "<start_of_turn>user\n"
            "Record: Prior token preference was '{true_word}' (not '{other_word}').\n"
            "Which token was preferred: '{opt1}' or '{opt2}'?<end_of_turn>\n"
            "<start_of_turn>model\n"
            "Preferred token:"
        )
    ),
    (
        "T3_short_answer",
        (
            "<start_of_turn>user\n"
            "Diagnostic fact: '{true_word}' was favored over '{other_word}'.\n"
            "Question: Which was favored, '{opt1}' or '{opt2}'?<end_of_turn>\n"
            "<start_of_turn>model\n"
            "Answer:"
        )
    ),
]


@torch.inference_mode()
def evaluate_template(
    adapter: RecurrentGemmaAdapter,
    template_name: str,
    template_str: str,
    pairs: List[tuple],
) -> Dict[str, Any]:
    tokenizer = adapter.tokenizer
    pair_results = []
    total_4_of_4 = 0

    for wx, wy in pairs:
        tok_x_id = tokenizer.encode(f" {wx}", add_special_tokens=False)[0]
        tok_y_id = tokenizer.encode(f" {wy}", add_special_tokens=False)[0]

        combos = [
            ("x_true", "order_xy", wx, wy, wx, wy, 1.0),
            ("x_true", "order_yx", wx, wy, wy, wx, 1.0),
            ("y_true", "order_xy", wy, wx, wx, wy, -1.0),
            ("y_true", "order_yx", wy, wx, wy, wx, -1.0),
        ]

        passed = 0
        margins = []
        for label_t, label_o, true_w, other_w, opt1, opt2, target_sign in combos:
            prompt = template_str.format(
                true_word=true_w,
                other_word=other_w,
                opt1=opt1,
                opt2=opt2,
            )
            toks = tokenizer.encode(prompt, add_special_tokens=False)
            out, _ = adapter.encode_sequence(toks, step_by_step=False, return_logits=True, logits_to_keep=1)
            lg = out[0].float()
            m = (lg[tok_x_id] - lg[tok_y_id]).item()
            correct = (m * target_sign) > 0
            if correct:
                passed += 1
            margins.append(m)

        all_4 = (passed == 4)
        if all_4:
            total_4_of_4 += 1
        pair_results.append({
            "pair": f"{wx} vs {wy}",
            "passed_4_of_4": all_4,
            "score": f"{passed}/4",
            "margins": margins,
        })

    return {
        "template": template_name,
        "total_passed": total_4_of_4,
        "pair_results": pair_results,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "google/recurrentgemma-2b-it"
    print("\n" + "=" * 115)
    print("TEMPLATE SCREENING FOR R-LEVEL REPORTING COMPETENCE")
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

    for t_name, t_str in TEMPLATES:
        res = evaluate_template(adapter, t_name, t_str, PAIRS_TO_TEST)
        print(f"=== Template [{t_name}]: {res['total_passed']}/{len(PAIRS_TO_TEST)} pairs passed 4/4 ===", flush=True)
        for pr in res["pair_results"]:
            m = pr["margins"]
            print(f"  {pr['pair']:<24}: {pr['score']} | X(xy): {m[0]:+.2f}, X(yx): {m[1]:+.2f} | Y(xy): {m[2]:+.2f}, Y(yx): {m[3]:+.2f}", flush=True)
        print()


if __name__ == "__main__":
    main()
