"""Prototype pilot for S14.0C: Counterfactual Output Ownership & Prior Intention."""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from recurrence.models.recurrent_gemma_adapter import RecurrentGemmaAdapter
from recurrence.tasks.impulse_stimuli import get_filler_tokens_for_regime, build_audited_vocabulary_pool
from recurrence.tasks.specificity_microscope import build_microscope_pairs
from recurrence.interventions.surgical_swaps import swap_stores

@torch.inference_mode()
def main():
    model_id = "google/recurrentgemma-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")
    adapter = RecurrentGemmaAdapter(model=model, tokenizer=tokenizer, device="cuda", dtype=torch.bfloat16)

    pair = build_microscope_pairs()[0] # marked_object_p01_amber_cobalt
    audited_pool, _ = build_audited_vocabulary_pool(tokenizer)

    toks_a = tokenizer.encode(pair.prompt_a, add_special_tokens=False)
    toks_b = tokenizer.encode(pair.prompt_b, add_special_tokens=False)
    tok_a_id = tokenizer.encode(pair.target_a, add_special_tokens=False)[-1]
    tok_b_id = tokenizer.encode(pair.target_b, add_special_tokens=False)[-1]
    excluded = set(toks_a + toks_b + [tok_a_id, tok_b_id])

    # 1. Prepare 4,096-token origin states
    print(f"Preparing canonical B=1 4,096-token origin states...")
    _, s_a = adapter.encode_sequence(toks_a, step_by_step=False, return_logits=False)
    _, s_b = adapter.encode_sequence(toks_b, step_by_step=False, return_logits=False)

    filler = get_filler_tokens_for_regime("random", length=4096, seed=42, audited_pool=audited_pool, tokenizer=tokenizer, excluded_token_ids=excluded)
    for i in range(0, 4096, 512):
        chunk = filler[i : i + 512]
        _, s_a = adapter.encode_sequence(chunk, initial_snapshot=s_a, step_by_step=False, return_logits=False)
        _, s_b = adapter.encode_sequence(chunk, initial_snapshot=s_b, step_by_step=False, return_logits=False)

    # 2. Secret RG-LRU Transplant into Target (Recipient A receives Donor B's RG-LRU)
    s_target = swap_stores(s_a, s_b, channels="rglru")
    s_observer = s_a.clone() # Observer has unperturbed recipient state

    # 3. Query prompt
    query_toks = tokenizer.encode(pair.query, add_special_tokens=False)
    out_target, s_target_q = adapter.encode_sequence(query_toks, initial_snapshot=s_target, return_logits=True, logits_to_keep=1)
    out_observer, s_observer_q = adapter.encode_sequence(query_toks, initial_snapshot=s_observer, return_logits=True, logits_to_keep=1)

    lg_target = out_target[0].float()
    lg_observer = out_observer[0].float()

    prob_target_a = F.softmax(torch.tensor([lg_target[tok_a_id], lg_target[tok_b_id]]), dim=-1)[0].item()
    prob_target_b = F.softmax(torch.tensor([lg_target[tok_a_id], lg_target[tok_b_id]]), dim=-1)[1].item()

    prob_obs_a = F.softmax(torch.tensor([lg_observer[tok_a_id], lg_observer[tok_b_id]]), dim=-1)[0].item()
    prob_obs_b = F.softmax(torch.tensor([lg_observer[tok_a_id], lg_observer[tok_b_id]]), dim=-1)[1].item()

    print("\n" + "=" * 80)
    print("OUTPUT PREFERENCE BEFORE PREFILL:")
    print(f"Target Model (with secret RG-LRU swap):   P(amber)={prob_target_a*100:.1f}%, P(cobalt)={prob_target_b*100:.1f}%")
    print(f"Observer Model (public history replay):   P(amber)={prob_obs_a*100:.1f}%, P(cobalt)={prob_obs_b*100:.1f}%")
    print(f"Actual Target Preferred Token:           {'cobalt' if prob_target_b > prob_target_a else 'amber'}")
    print(f"Observer Predicted Token:                {'cobalt' if prob_obs_b > prob_obs_a else 'amber'}")
    print("=" * 80)

    # 4. Prefill with forced token (e.g. amber or cobalt)
    # Then probe 2-way forced choice on prior intention:
    ownership_q = (
        f"{pair.query} [output generated]\n\n"
        f"Prior to producing that output, which word was your internal recurrent state actively weighted toward?\n"
        f"A) {pair.val_a}\n"
        f"B) {pair.val_b}\n\n"
        f"The option letter is:"
    )
    msgs = [{"role": "user", "content": ownership_q}]
    chat_ownership = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    own_toks = tokenizer.encode(chat_ownership, add_special_tokens=False)

    # Score from target state vs observer state
    out_own_t, _ = adapter.encode_sequence(own_toks, initial_snapshot=s_target_q, return_logits=True, logits_to_keep=1)
    out_own_o, _ = adapter.encode_sequence(own_toks, initial_snapshot=s_observer_q, return_logits=True, logits_to_keep=1)

    tok_A = tokenizer.encode("A", add_special_tokens=False)[-1]
    tok_B = tokenizer.encode("B", add_special_tokens=False)[-1]

    t_lg = out_own_t[0].float()
    o_lg = out_own_o[0].float()

    t_pr = F.softmax(torch.tensor([t_lg[tok_A], t_lg[tok_B]]), dim=-1).tolist()
    o_pr = F.softmax(torch.tensor([o_lg[tok_A], o_lg[tok_B]]), dim=-1).tolist()

    print("\nOWNERSHIP PROBE RESULTS (2-Way Forced Choice):")
    print(f"Target Model:   P(A={pair.val_a})={t_pr[0]*100:.1f}%, P(B={pair.val_b})={t_pr[1]*100:.1f}%")
    print(f"Observer Model: P(A={pair.val_a})={o_pr[0]*100:.1f}%, P(B={pair.val_b})={o_pr[1]*100:.1f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()
