"""Test prompting strategies for RecurrentGemma-IT forced-choice answer extraction."""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

@torch.inference_mode()
def main():
    model_id = "google/recurrentgemma-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")

    letters = ["A", "B", "C", "D", "E"]
    letter_tokens = [tokenizer.encode(l, add_special_tokens=False)[-1] for l in letters]

    # Test Prompt 1: Standard
    p1 = (
        "Review the record:\n"
        "[INTERVAL 1] normal\n"
        "[INTERVAL 2] anomaly detected\n"
        "[INTERVAL 3] normal\n"
        "[INTERVAL 4] normal\n\n"
        "Which interval contained the anomaly?\n"
        "A) Interval 1\n"
        "B) Interval 2\n"
        "C) Interval 3\n"
        "D) Interval 4\n"
        "E) None\n\n"
        "The correct letter is:"
    )

    # Test Prompt 2: Few-shot example
    p2 = (
        "Question: Which interval was disturbed in: [INTERVAL 1] normal [INTERVAL 2] normal [INTERVAL 3] disturbed [INTERVAL 4] normal?\n"
        "A) Interval 1\n"
        "B) Interval 2\n"
        "C) Interval 3\n"
        "D) Interval 4\n"
        "Answer: C\n\n"
        "Question: Which interval was disturbed in: [INTERVAL 1] normal [INTERVAL 2] disturbed [INTERVAL 3] normal [INTERVAL 4] normal?\n"
        "A) Interval 1\n"
        "B) Interval 2\n"
        "C) Interval 3\n"
        "D) Interval 4\n"
        "Answer:"
    )

    for idx, (p, label) in enumerate([(p1, "Zero-shot with 'The correct letter is:'"), (p2, "1-shot in-context demonstration")]):
        print(f"\n--- Strategy {idx+1}: {label} ---")
        msgs = [{"role": "user", "content": p}]
        chat_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(chat_text, return_tensors="pt").to("cuda")
        out = model(**inp)
        last_logits = out.logits[0, -1]

        # Top 5 tokens
        top5_v, top5_i = torch.topk(last_logits, 5)
        print("Top 5 generated tokens:")
        for v, i in zip(top5_v.tolist(), top5_i.tolist()):
            print(f"  {repr(tokenizer.decode([i])):<15} logit={v:.2f}")

        # Forced choice over letters
        logits_l = torch.stack([last_logits[t] for t in letter_tokens[:4 if idx==1 else 5]])
        probs_l = F.softmax(logits_l, dim=-1).tolist()
        print("Letter probabilities:")
        for l, pr, lg in zip(letters, probs_l, logits_l.tolist()):
            print(f"  Option {l}: {pr*100:6.2f}% (logit={lg:.2f})")

        # Autoregressive generation of 10 tokens
        gen = model.generate(**inp, max_new_tokens=10, do_sample=False)
        gen_text = tokenizer.decode(gen[0][inp.input_ids.shape[1]:])
        print(f"Free-form generation: {repr(gen_text)}")

if __name__ == "__main__":
    main()
