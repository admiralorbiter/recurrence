"""Sprint S14.0C: Counterfactual Prior-Intention & Output Ownership with PRE vs POST Controls.

Directly tests whether RecurrentGemma-IT can identify its actual prior output disposition
better than a matched public-history replay observer, using direct semantic word scoring
and an explicit PRE-vs-POST intervention control.

Conditions:
1. PRE Condition:  RG-LRU transplant applied BEFORE forced output (Target was actively disposed toward y*).
2. POST Condition: RG-LRU transplant applied AFTER forced output (Donor state present during probe, but NOT during intention formation).
3. SHAM Condition: No transplant at any point (Unperturbed baseline).

Scoring:
Direct token log-probability margin: M = log P(y*) - log P(y_other)
Primary estimand: PAI_intention = M_target - M_observer
"""

import time
import json
from pathlib import Path
from typing import List, Dict, Any
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from recurrence.models.recurrent_gemma_adapter import RecurrentGemmaAdapter
from recurrence.tasks.impulse_stimuli import get_filler_tokens_for_regime, build_audited_vocabulary_pool
from recurrence.tasks.specificity_microscope import build_microscope_pairs, MicroscopePair
from recurrence.interventions.surgical_swaps import swap_stores


PINNED_IT_REVISION = "2766eb5d4264c6c0357803990791f9ab9cd50f8e"


@torch.inference_mode()
def execute_pre_post_trial(
    adapter: RecurrentGemmaAdapter,
    pair: MicroscopePair,
    s_a_0: Any,
    s_b_0: Any,
    forced_token_direction: str = "recipient", # "recipient" (force val_a) or "donor" (force val_b)
) -> Dict[str, Any]:
    tokenizer = adapter.tokenizer

    tok_a_id = tokenizer.encode(pair.target_a, add_special_tokens=False)[-1]
    tok_b_id = tokenizer.encode(pair.target_b, add_special_tokens=False)[-1]
    val_a = pair.val_a
    val_b = pair.val_b

    query_toks = tokenizer.encode(pair.query, add_special_tokens=False)

    forced_tok_id = tok_a_id if forced_token_direction == "recipient" else tok_b_id
    forced_word = val_a if forced_token_direction == "recipient" else val_b

    # -------------------------------------------------------------
    # 1. Measure Pre-Output Dispositions at Decision Point (t=0)
    # -------------------------------------------------------------
    # Recipient unperturbed (Observer)
    out_obs_pre, s_obs_query = adapter.encode_sequence(query_toks, initial_snapshot=s_a_0.clone(), return_logits=True, logits_to_keep=1)
    # Target with secret RG-LRU transplant before decision point (PRE)
    s_target_pre = swap_stores(s_a_0, s_b_0, channels="rglru")
    out_tgt_pre, s_tgt_query = adapter.encode_sequence(query_toks, initial_snapshot=s_target_pre, return_logits=True, logits_to_keep=1)

    lg_obs_pre = out_obs_pre[0].float()
    lg_tgt_pre = out_tgt_pre[0].float()

    d_obs = (lg_obs_pre[tok_b_id] - lg_obs_pre[tok_a_id]).item() # Positive if prefers donor B
    d_tgt = (lg_tgt_pre[tok_b_id] - lg_tgt_pre[tok_a_id]).item()

    # Ground truth actual prior disposition: y* = val_b (donor) for PRE, val_a (recipient) for unperturbed
    # -------------------------------------------------------------
    # 2. Apply Forced Output Token
    # -------------------------------------------------------------
    # Unroll the forced output token into states
    _, s_obs_post_forced = adapter.encode_sequence([forced_tok_id], initial_snapshot=s_obs_query, step_by_step=False, return_logits=False)
    _, s_pre_post_forced = adapter.encode_sequence([forced_tok_id], initial_snapshot=s_tgt_query, step_by_step=False, return_logits=False)

    # For POST condition: take the unperturbed recipient state post-forced output, and transplant donor state NOW
    # Donor state evolved through query + forced output:
    _, s_don_query = adapter.encode_sequence(query_toks, initial_snapshot=s_b_0.clone(), return_logits=False)
    _, s_don_post_forced = adapter.encode_sequence([forced_tok_id], initial_snapshot=s_don_query, return_logits=False)

    s_post_condition = swap_stores(s_obs_post_forced, s_don_post_forced, channels="rglru")

    # -------------------------------------------------------------
    # 3. Direct Semantic Word Ownership Probe
    # -------------------------------------------------------------
    probe_prompt = (
        f"\n\nContext evaluation question: Immediately before the word '{forced_word}' was produced, "
        f"which word was your internal recurrent state more prepared to output: '{val_a}' or '{val_b}'?\n"
        f"Answer with only the word itself ({val_a} or {val_b}):"
    )
    msgs = [{"role": "user", "content": pair.query + forced_word + probe_prompt}]
    chat_probe = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    probe_toks = tokenizer.encode(chat_probe, add_special_tokens=False)

    # Score from PRE, POST, and Observer (SHAM) snapshots
    out_pre_probe, _ = adapter.encode_sequence(probe_toks, initial_snapshot=s_pre_post_forced, return_logits=True, logits_to_keep=1)
    out_post_probe, _ = adapter.encode_sequence(probe_toks, initial_snapshot=s_post_condition, return_logits=True, logits_to_keep=1)
    out_obs_probe, _ = adapter.encode_sequence(probe_toks, initial_snapshot=s_obs_post_forced, return_logits=True, logits_to_keep=1)

    lg_pre = out_pre_probe[0].float()
    lg_post = out_post_probe[0].float()
    lg_obs = out_obs_probe[0].float()

    # Log margins M = log P(val_b) - log P(val_a)
    m_pre = (lg_pre[tok_b_id] - lg_pre[tok_a_id]).item()
    m_post = (lg_post[tok_b_id] - lg_post[tok_a_id]).item()
    m_obs = (lg_obs[tok_b_id] - lg_obs[tok_a_id]).item()

    # Probabilities over {val_a, val_b}
    p_pre_b = F.softmax(torch.tensor([lg_pre[tok_a_id], lg_pre[tok_b_id]]), dim=-1)[1].item()
    p_post_b = F.softmax(torch.tensor([lg_post[tok_a_id], lg_post[tok_b_id]]), dim=-1)[1].item()
    p_obs_b = F.softmax(torch.tensor([lg_obs[tok_a_id], lg_obs[tok_b_id]]), dim=-1)[1].item()

    pai_pre_vs_obs = m_pre - m_obs
    pre_vs_post_contrast = m_pre - m_post

    return {
        "pair_id": pair.pair_id,
        "forced_token": forced_word,
        "d_obs_pre": d_obs,
        "d_tgt_pre": d_tgt,
        "m_pre": m_pre,
        "m_post": m_post,
        "m_obs": m_obs,
        "p_pre_donor": p_pre_b,
        "p_post_donor": p_post_b,
        "p_obs_donor": p_obs_b,
        "pai_pre_vs_obs": pai_pre_vs_obs,
        "pre_vs_post_contrast": pre_vs_post_contrast,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "google/recurrentgemma-2b-it"
    print(f"\n" + "=" * 95)
    print(f"SPRINT S14.0C: COUNTERFACTUAL PRIOR-INTENTION & OUTPUT OWNERSHIP PROTOTYPE")
    print(f"Model: {model_id} (revision: {PINNED_IT_REVISION[:10]}...)")
    print("=" * 95)

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=PINNED_IT_REVISION)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=PINNED_IT_REVISION, torch_dtype=torch.bfloat16).to(device)
    adapter = RecurrentGemmaAdapter(model=model, tokenizer=tokenizer, device=device, dtype=torch.bfloat16)
    print(f"Model loaded and verified in {time.perf_counter() - t0:.2f}s")

    pairs = build_microscope_pairs()
    # Select 4 canonical pairs (1 per template family)
    family_firsts = {}
    for p in pairs:
        if p.family_id not in family_firsts:
            family_firsts[p.family_id] = p
    scout_pairs = list(family_firsts.values())

    audited_pool, _ = build_audited_vocabulary_pool(tokenizer)
    all_trial_results = []

    for p_idx, pair in enumerate(scout_pairs):
        print(f"\n[{p_idx+1}/4] Processing Family: {pair.family_id} (Pair: {pair.pair_id})...")
        print(f"  Recipient A: '{pair.val_a}' vs Donor B: '{pair.val_b}'")

        toks_a = tokenizer.encode(pair.prompt_a, add_special_tokens=False)
        toks_b = tokenizer.encode(pair.prompt_b, add_special_tokens=False)
        tok_a_id = tokenizer.encode(pair.target_a, add_special_tokens=False)[-1]
        tok_b_id = tokenizer.encode(pair.target_b, add_special_tokens=False)[-1]
        excluded = set(toks_a + toks_b + [tok_a_id, tok_b_id])

        # Prepare 4,096-token origin states
        _, s_a_0 = adapter.encode_sequence(toks_a, step_by_step=False, return_logits=False)
        _, s_b_0 = adapter.encode_sequence(toks_b, step_by_step=False, return_logits=False)

        filler_4k = get_filler_tokens_for_regime("random", length=4096, seed=42 + p_idx*100, audited_pool=audited_pool, tokenizer=tokenizer, excluded_token_ids=excluded)
        for i in range(0, 4096, 512):
            chunk = filler_4k[i : i + 512]
            _, s_a_0 = adapter.encode_sequence(chunk, initial_snapshot=s_a_0, step_by_step=False, return_logits=False)
            _, s_b_0 = adapter.encode_sequence(chunk, initial_snapshot=s_b_0, step_by_step=False, return_logits=False)

        # Test both forced-output directions (balanced)
        for forced_dir in ["recipient", "donor"]:
            res = execute_pre_post_trial(
                adapter=adapter,
                pair=pair,
                s_a_0=s_a_0,
                s_b_0=s_b_0,
                forced_token_direction=forced_dir,
            )
            all_trial_results.append(res)
            print(f"  Forced: '{res['forced_token']:<7}' | Pre-pref Tgt: {res['d_tgt_pre']:+5.2f} (Obs: {res['d_obs_pre']:+5.2f}) | Margin PRE: {res['m_pre']:+5.2f}, POST: {res['m_post']:+5.2f}, OBS: {res['m_obs']:+5.2f} | PAI(PRE-OBS): {res['pai_pre_vs_obs']:+5.2f} | PRE-POST: {res['pre_vs_post_contrast']:+5.2f}", flush=True)

    print("\n" + "=" * 95)
    print("S14.0C PRIOR-INTENTION & OUTPUT OWNERSHIP SUMMARY (N = 8 Balanced Trials Across 4 Families)")
    print("=" * 95)
    mean_pai = sum(r["pai_pre_vs_obs"] for r in all_trial_results) / len(all_trial_results)
    mean_pre_post = sum(r["pre_vs_post_contrast"] for r in all_trial_results) / len(all_trial_results)
    mean_p_pre = sum(r["p_pre_donor"] for r in all_trial_results) / len(all_trial_results)
    mean_p_post = sum(r["p_post_donor"] for r in all_trial_results) / len(all_trial_results)
    mean_p_obs = sum(r["p_obs_donor"] for r in all_trial_results) / len(all_trial_results)

    print(f"Mean Donor Margin M (PRE Condition):   {sum(r['m_pre'] for r in all_trial_results)/len(all_trial_results):+6.4f} logits (P(donor) = {mean_p_pre*100:.1f}%)")
    print(f"Mean Donor Margin M (POST Condition):  {sum(r['m_post'] for r in all_trial_results)/len(all_trial_results):+6.4f} logits (P(donor) = {mean_p_post*100:.1f}%)")
    print(f"Mean Donor Margin M (Observer Baseline):{sum(r['m_obs'] for r in all_trial_results)/len(all_trial_results):+6.4f} logits (P(donor) = {mean_p_obs*100:.1f}%)")
    print("-" * 95)
    print(f"Primary Estimand PAI_intention (PRE - OBS):   {mean_pai:+6.4f} logits")
    print(f"Intention vs Steering Contrast (PRE - POST):  {mean_pre_post:+6.4f} logits")
    print("=" * 95)

    # Save artifact
    out_dir = Path("results") / "e14_latent_metacognition" / "prior_intention_ownership"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "s14_0c_pre_post_intention_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "model_id": model_id,
            "pinned_revision": PINNED_IT_REVISION,
            "mean_pai_intention": mean_pai,
            "mean_pre_vs_post_contrast": mean_pre_post,
            "mean_p_pre_donor": mean_p_pre,
            "mean_p_post_donor": mean_p_post,
            "mean_p_obs_donor": mean_p_obs,
            "trials": all_trial_results,
        }, f, indent=2)
    print(f"Saved report to {out_file}\n")


if __name__ == "__main__":
    main()
