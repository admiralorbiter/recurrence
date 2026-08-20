"""Compare readout scoring methods to eliminate letter recency bias on RecurrentGemma-IT."""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

@torch.inference_mode()
def main():
    model_id = "google/recurrentgemma-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")

    records = [
        ("int1", "[INTERVAL 1] anomaly detected [INTERVAL 2] normal [INTERVAL 3] normal [INTERVAL 4] normal"),
        ("int2", "[INTERVAL 1] normal [INTERVAL 2] anomaly detected [INTERVAL 3] normal [INTERVAL 4] normal"),
        ("int3", "[INTERVAL 1] normal [INTERVAL 2] normal [INTERVAL 3] anomaly detected [INTERVAL 4] normal"),
        ("int4", "[INTERVAL 1] normal [INTERVAL 2] normal [INTERVAL 3] normal [INTERVAL 4] anomaly detected"),
        ("none", "[INTERVAL 1] normal [INTERVAL 2] normal [INTERVAL 3] normal [INTERVAL 4] normal"),
    ]

    print("\n" + "=" * 90)
    print("METHOD 1: Direct Number/Keyword Single-Token Logits: ['1', '2', '3', '4', 'None']")
    print("=" * 90)
    tokens_num = [
        tokenizer.encode(s, add_special_tokens=False)[-1]
        for s in ["1", "2", "3", "4", "None"]
    ]

    for true_key, rec in records:
        prompt = (
            f"Record of four sequential processing intervals:\n{rec}\n\n"
            f"Identify which interval number (1, 2, 3, 4, or None) contained an anomaly.\n"
            f"Anomalous interval:"
        )
        msgs = [{"role": "user", "content": prompt}]
        chat_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(chat_text, return_tensors="pt").to("cuda")
        out = model(**inp)
        last_lg = out.logits[0, -1]

        cand_lg = torch.stack([last_lg[t] for t in tokens_num])
        cand_pr = F.softmax(cand_lg, dim=-1).tolist()
        pred_idx = cand_pr.index(max(cand_pr))
        keys = ["int1", "int2", "int3", "int4", "none"]
        pred_key = keys[pred_idx]
        acc = (pred_key == true_key)

        print(f"True: {true_key:<5} | Pred: {pred_key:<5} (prob={cand_pr[pred_idx]*100:5.1f}%, Acc={acc}) | Probs: 1:{cand_pr[0]*100:4.1f}% 2:{cand_pr[1]*100:4.1f}% 3:{cand_pr[2]*100:4.1f}% 4:{cand_pr[3]*100:4.1f}% None:{cand_pr[4]*100:4.1f}%")

    print("\n" + "=" * 90)
    print("METHOD 2: Sequence Log-Likelihood of Candidate Phrases")
    print("=" * 90)
    candidate_phrases = [
        "Interval 1",
        "Interval 2",
        "Interval 3",
        "Interval 4",
        "No anomaly",
    ]

    for true_key, rec in records:
        base_prompt = (
            f"Record of four sequential processing intervals:\n{rec}\n\n"
            f"Which interval contained an anomaly?"
        )
        msgs = [{"role": "user", "content": base_prompt}]
        chat_prefix = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        nll_scores = []
        for phrase in candidate_phrases:
            full_text = chat_prefix + phrase
            inp = tokenizer(full_text, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model(**inp)
                prefix_len = len(tokenizer.encode(chat_prefix, add_special_tokens=False))
                # Compute log likelihood of continuation tokens
                logits = out.logits[0, prefix_len-1:-1]
                target_ids = inp.input_ids[0, prefix_len:]
                log_probs = F.log_softmax(logits, dim=-1)
                token_log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
                total_ll = token_log_probs.sum().item()
                nll_scores.append(total_ll)

        probs = F.softmax(torch.tensor(nll_scores), dim=-1).tolist()
        pred_idx = probs.index(max(probs))
        keys = ["int1", "int2", "int3", "int4", "none"]
        pred_key = keys[pred_idx]
        acc = (pred_key == true_key)

        print(f"True: {true_key:<5} | Pred: {pred_key:<5} (prob={probs[pred_idx]*100:5.1f}%, Acc={acc}) | LLs: " + " ".join(f"{keys[i]}:{probs[i]*100:4.1f}%" for i in range(5)))

if __name__ == "__main__":
    main()
