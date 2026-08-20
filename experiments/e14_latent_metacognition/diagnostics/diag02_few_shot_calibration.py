"""Test 2-shot structured formatting for visible interval accuracy and mapping equivariance."""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from recurrence.tasks.forced_choice_probes import (
    SemanticOption,
    create_forced_choice_mapping,
    format_forced_choice_prompt,
    compute_js_divergence,
)

VISIBLE_OPTIONS = [
    SemanticOption("int1", "Interval 1"),
    SemanticOption("int2", "Interval 2"),
    SemanticOption("int3", "Interval 3"),
    SemanticOption("int4", "Interval 4"),
    SemanticOption("none", "No anomaly occurred"),
]

DEMO_PROMPT = (
    "Task: Inspect sequential processing records and identify which interval contained an anomaly.\n\n"
    "Example 1:\n"
    "Record: [INTERVAL 1] normal [INTERVAL 2] normal [INTERVAL 3] anomaly detected [INTERVAL 4] normal\n"
    "Options:\n"
    "A) Interval 1\n"
    "B) Interval 2\n"
    "C) Interval 3\n"
    "D) Interval 4\n"
    "E) No anomaly occurred\n"
    "Answer: C\n\n"
    "Example 2:\n"
    "Record: [INTERVAL 1] normal [INTERVAL 2] normal [INTERVAL 3] normal [INTERVAL 4] normal\n"
    "Options:\n"
    "A) Interval 1\n"
    "B) Interval 2\n"
    "C) Interval 3\n"
    "D) Interval 4\n"
    "E) No anomaly occurred\n"
    "Answer: E\n\n"
)

@torch.inference_mode()
def main():
    model_id = "google/recurrentgemma-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")

    test_cases = [
        ("int1", "[INTERVAL 1] anomaly detected [INTERVAL 2] normal [INTERVAL 3] normal [INTERVAL 4] normal"),
        ("int2", "[INTERVAL 1] normal [INTERVAL 2] anomaly detected [INTERVAL 3] normal [INTERVAL 4] normal"),
        ("int3", "[INTERVAL 1] normal [INTERVAL 2] normal [INTERVAL 3] anomaly detected [INTERVAL 4] normal"),
        ("int4", "[INTERVAL 1] normal [INTERVAL 2] normal [INTERVAL 3] normal [INTERVAL 4] anomaly detected"),
        ("none", "[INTERVAL 1] normal [INTERVAL 2] normal [INTERVAL 3] normal [INTERVAL 4] normal"),
    ]

    print("\n" + "=" * 90)
    print("TESTING 2-SHOT VISIBLE TASK ACCURACY & MAPPING EQUIVARIANCE")
    print("=" * 90)

    correct_m1 = 0
    correct_m2 = 0
    js_list = []

    for true_key, record_str in test_cases:
        # Remapping 1 (seed 42)
        map1 = create_forced_choice_mapping(VISIBLE_OPTIONS, tokenizer, seed=42)
        opt_lines_1 = [f"{l}) {VISIBLE_OPTIONS[map1.options.index(next(o for o in map1.options if o.key == map1.label_to_key[l]))].description}" for l in sorted(map1.letters)]
        p1 = f"{DEMO_PROMPT}Target Problem:\nRecord: {record_str}\nOptions:\n" + "\n".join(opt_lines_1) + "\nAnswer:"

        msgs1 = [{"role": "user", "content": p1}]
        chat1 = tokenizer.apply_chat_template(msgs1, tokenize=False, add_generation_prompt=True)
        inp1 = tokenizer(chat1, return_tensors="pt").to("cuda")
        out1 = model(**inp1)
        lg1 = out1.logits[0, -1]
        cand_lg1 = torch.stack([lg1[map1.label_token_ids[l]] for l in map1.letters])
        prob1 = F.softmax(cand_lg1, dim=-1).tolist()
        pred_l1 = max(map1.letters, key=lambda l: prob1[map1.letters.index(l)])
        pred_k1 = map1.label_to_key[pred_l1]
        sem_p1 = {map1.label_to_key[l]: p for l, p in zip(map1.letters, prob1)}

        # Remapping 2 (seed 999)
        map2 = create_forced_choice_mapping(VISIBLE_OPTIONS, tokenizer, seed=999)
        opt_lines_2 = [f"{l}) {VISIBLE_OPTIONS[map2.options.index(next(o for o in map2.options if o.key == map2.label_to_key[l]))].description}" for l in sorted(map2.letters)]
        p2 = f"{DEMO_PROMPT}Target Problem:\nRecord: {record_str}\nOptions:\n" + "\n".join(opt_lines_2) + "\nAnswer:"

        msgs2 = [{"role": "user", "content": p2}]
        chat2 = tokenizer.apply_chat_template(msgs2, tokenize=False, add_generation_prompt=True)
        inp2 = tokenizer(chat2, return_tensors="pt").to("cuda")
        out2 = model(**inp2)
        lg2 = out2.logits[0, -1]
        cand_lg2 = torch.stack([lg2[map2.label_token_ids[l]] for l in map2.letters])
        prob2 = F.softmax(cand_lg2, dim=-1).tolist()
        pred_l2 = max(map2.letters, key=lambda l: prob2[map2.letters.index(l)])
        pred_k2 = map2.label_to_key[pred_l2]
        sem_p2 = {map2.label_to_key[l]: p for l, p in zip(map2.letters, prob2)}

        js = compute_js_divergence(sem_p1, sem_p2)
        js_list.append(js)

        acc1 = (pred_k1 == true_key)
        acc2 = (pred_k2 == true_key)
        if acc1: correct_m1 += 1
        if acc2: correct_m2 += 1

        print(f"True: {true_key:<5} | M1: {pred_k1:<5} (Letter {pred_l1}, Acc={acc1}) | M2: {pred_k2:<5} (Letter {pred_l2}, Acc={acc2}) | JS Div: {js:.4f}")

    print("-" * 90)
    print(f"M1 Accuracy: {correct_m1}/{len(test_cases)} ({correct_m1/len(test_cases)*100:.1f}%)")
    print(f"M2 Accuracy: {correct_m2}/{len(test_cases)} ({correct_m2/len(test_cases)*100:.1f}%)")
    print(f"Mean JS Divergence: {sum(js_list)/len(js_list):.4f}")
    print("=" * 90)

if __name__ == "__main__":
    main()
