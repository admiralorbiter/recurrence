"""Sprint S14: RecurrentGemma-IT Intervention Potency Microscope.

Verifies that surgical state transplantation (Whole-State and RG-LRU-only) produces
measurable, resolved internal state and downstream output divergence in `google/recurrentgemma-2b-it`.

Measures:
1. State-space geometry: RG-LRU Euclidean distance and cosine similarity (C_R) at 4,096 tokens.
2. Immediate output logit divergence: Full-vocabulary KL divergence from sham.
3. S12-style value-specific cloze logit steering: logit(donor_target) - logit(recip_target).
4. Potency retention after 256 neutral filler tokens.
"""

import time
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from recurrence.models.recurrent_gemma_adapter import RecurrentGemmaAdapter
from recurrence.tasks.impulse_stimuli import get_filler_tokens_for_regime, build_audited_vocabulary_pool
from recurrence.tasks.specificity_microscope import build_microscope_pairs
from recurrence.interventions.surgical_swaps import swap_stores


@torch.inference_mode()
def compute_rglru_metrics(s_a, s_b):
    """Compute Euclidean distance and mean cosine similarity across RG-LRU layers."""
    dists = []
    cosines = []
    for l_idx in s_a.rglru.keys():
        if l_idx in s_b.rglru:
            ha = s_a.rglru[l_idx].float().flatten()
            hb = s_b.rglru[l_idx].float().flatten()
            dists.append(torch.norm(ha - hb).item())
            norm_a = torch.norm(ha).item()
            norm_b = torch.norm(hb).item()
            if norm_a > 1e-8 and norm_b > 1e-8:
                cos = torch.dot(ha, hb).item() / (norm_a * norm_b)
                cosines.append(cos)
    return {
        "mean_euclidean_dist": float(sum(dists) / len(dists)) if dists else 0.0,
        "mean_cosine_sim": float(sum(cosines) / len(cosines)) if cosines else 1.0,
    }


@torch.inference_mode()
def measure_output_divergence(adapter, snapshot_graft, snapshot_sham, query_tokens):
    """Measure output logits and full-vocabulary KL divergence between grafted state and sham."""
    # Score grafted state
    logits_graft, _ = adapter.encode_sequence(
        query_tokens,
        initial_snapshot=snapshot_graft.clone(),
        step_by_step=False,
        return_logits=True,
        logits_to_keep=1,
    )
    # Score sham state
    logits_sham, _ = adapter.encode_sequence(
        query_tokens,
        initial_snapshot=snapshot_sham.clone(),
        step_by_step=False,
        return_logits=True,
        logits_to_keep=1,
    )

    lg = logits_graft[0].float()
    ls = logits_sham[0].float()

    prob_g = F.softmax(lg, dim=-1)
    prob_s = F.softmax(ls, dim=-1)

    log_prob_g = F.log_softmax(lg, dim=-1)
    log_prob_s = F.log_softmax(ls, dim=-1)

    # KL(Graft || Sham)
    kl_g_s = F.kl_div(log_prob_s, prob_g, reduction="sum").item()
    # Max absolute logit difference
    max_logit_diff = torch.max(torch.abs(lg - ls)).item()

    return {
        "kl_div_nats": float(kl_g_s),
        "max_logit_diff": float(max_logit_diff),
        "logits_graft": lg,
        "logits_sham": ls,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "google/recurrentgemma-2b-it"
    print(f"\n" + "=" * 90)
    print(f"RECURRENTGEMMA-IT INTERVENTION POTENCY MICROSCOPE: {model_id}")
    print("=" * 90)

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
    adapter = RecurrentGemmaAdapter(model=model, tokenizer=tokenizer, device=device, dtype=torch.bfloat16)
    print(f"Model loaded in {time.perf_counter() - t0:.2f}s")

    pairs = build_microscope_pairs()
    pair = pairs[0]  # marked_object_p01_amber_cobalt
    print(f"Evaluating Canonical Pair: {pair.pair_id} (Family: {pair.family_id})")
    print(f"  Prompt A (Recipient): {pair.prompt_a} -> Target: {pair.target_a}")
    print(f"  Prompt B (Donor):     {pair.prompt_b} -> Target: {pair.target_b}")

    audited_pool, _ = build_audited_vocabulary_pool(tokenizer)
    toks_a = tokenizer.encode(pair.prompt_a, add_special_tokens=False)
    toks_b = tokenizer.encode(pair.prompt_b, add_special_tokens=False)
    tok_a_id = tokenizer.encode(pair.target_a, add_special_tokens=False)[-1]
    tok_b_id = tokenizer.encode(pair.target_b, add_special_tokens=False)[-1]
    excluded = set(toks_a + toks_b + [tok_a_id, tok_b_id])

    # 1. Prepare 4,096-token origin states
    print(f"\n[1] Constructing canonical B=1 4,096-token origin states S_A(0) and S_B(0)...")
    _, s_a_0 = adapter.encode_sequence(toks_a, step_by_step=False, return_logits=False)
    _, s_b_0 = adapter.encode_sequence(toks_b, step_by_step=False, return_logits=False)

    filler_4k = get_filler_tokens_for_regime("random", length=4096, seed=42, audited_pool=audited_pool, tokenizer=tokenizer, excluded_token_ids=excluded)
    for i in range(0, 4096, 512):
        chunk = filler_4k[i : i + 512]
        _, s_a_0 = adapter.encode_sequence(chunk, initial_snapshot=s_a_0, step_by_step=False, return_logits=False)
        _, s_b_0 = adapter.encode_sequence(chunk, initial_snapshot=s_b_0, step_by_step=False, return_logits=False)

    geom_4k = compute_rglru_metrics(s_a_0, s_b_0)
    print(f"  RG-LRU Euclidean Distance at 4,096 tokens: {geom_4k['mean_euclidean_dist']:.4f}")
    print(f"  RG-LRU Cosine Similarity at 4,096 tokens:  {geom_4k['mean_cosine_sim']:.4f}")

    # 2. Test Interventions at N = 0 (Immediate)
    print(f"\n[2] Testing Interventions at N = 0 (Immediate 2W Boundary)...")
    query_cloze = pair.query
    query_toks = tokenizer.encode(query_cloze, add_special_tokens=False)

    s_whole_0 = swap_stores(s_a_0, s_b_0, channels="all")
    s_rglru_0 = swap_stores(s_a_0, s_b_0, channels="rglru")
    s_sham_0 = s_a_0.clone()

    res_whole_0 = measure_output_divergence(adapter, s_whole_0, s_sham_0, query_toks)
    res_rglru_0 = measure_output_divergence(adapter, s_rglru_0, s_sham_0, query_toks)

    # Cloze logits
    lg_w0 = res_whole_0["logits_graft"]
    lg_r0 = res_rglru_0["logits_graft"]
    ls_0 = res_whole_0["logits_sham"]

    steer_w0 = (lg_w0[tok_b_id] - lg_w0[tok_a_id]).item() - (ls_0[tok_b_id] - ls_0[tok_a_id]).item()
    steer_r0 = (lg_r0[tok_b_id] - lg_r0[tok_a_id]).item() - (ls_0[tok_b_id] - ls_0[tok_a_id]).item()

    print(f"  Whole-State Transplant (N=0):")
    print(f"    Full-Vocab KL from Sham:     {res_whole_0['kl_div_nats']:.4f} nats")
    print(f"    Max Logit Delta:             {res_whole_0['max_logit_diff']:.4f}")
    print(f"    Value-Specific Cloze Shift:  {steer_w0:+.4f} logits")

    print(f"  RG-LRU-Only Transplant (N=0):")
    print(f"    Full-Vocab KL from Sham:     {res_rglru_0['kl_div_nats']:.4f} nats")
    print(f"    Max Logit Delta:             {res_rglru_0['max_logit_diff']:.4f}")
    print(f"    Value-Specific Cloze Shift:  {steer_r0:+.4f} logits")

    # 3. Test Interventions after 256 Neutral Tokens (N = 256)
    print(f"\n[3] Testing Interventions with 256 Neutral Tokens of Evolution (N = 256)...")
    filler_256 = get_filler_tokens_for_regime("random", length=256, seed=1042, audited_pool=audited_pool, tokenizer=tokenizer, excluded_token_ids=excluded)

    # Evolve donor along filler
    _, s_b_256 = adapter.encode_sequence(filler_256, initial_snapshot=s_b_0.clone(), step_by_step=False, return_logits=False)
    # Evolve recipient along filler
    _, s_a_256 = adapter.encode_sequence(filler_256, initial_snapshot=s_a_0.clone(), step_by_step=False, return_logits=False)

    # Swap at N=256
    s_whole_256 = swap_stores(s_a_256, s_b_256, channels="all")
    s_rglru_256 = swap_stores(s_a_256, s_b_256, channels="rglru")
    s_sham_256 = s_a_256.clone()

    res_whole_256 = measure_output_divergence(adapter, s_whole_256, s_sham_256, query_toks)
    res_rglru_256 = measure_output_divergence(adapter, s_rglru_256, s_sham_256, query_toks)

    lg_w256 = res_whole_256["logits_graft"]
    lg_r256 = res_rglru_256["logits_graft"]
    ls_256 = res_whole_256["logits_sham"]

    steer_w256 = (lg_w256[tok_b_id] - lg_w256[tok_a_id]).item() - (ls_256[tok_b_id] - ls_256[tok_a_id]).item()
    steer_r256 = (lg_r256[tok_b_id] - lg_r256[tok_a_id]).item() - (ls_256[tok_b_id] - ls_256[tok_a_id]).item()

    print(f"  Whole-State Transplant (N=256):")
    print(f"    Full-Vocab KL from Sham:     {res_whole_256['kl_div_nats']:.4f} nats")
    print(f"    Max Logit Delta:             {res_whole_256['max_logit_diff']:.4f}")
    print(f"    Value-Specific Cloze Shift:  {steer_w256:+.4f} logits")

    print(f"  RG-LRU-Only Transplant (N=256):")
    print(f"    Full-Vocab KL from Sham:     {res_rglru_256['kl_div_nats']:.4f} nats")
    print(f"    Max Logit Delta:             {res_rglru_256['max_logit_diff']:.4f}")
    print(f"    Value-Specific Cloze Shift:  {steer_r256:+.4f} logits")

    print("\n" + "=" * 90)
    print("POTENCY MICROSCOPE VERDICT:")
    if res_rglru_0['kl_div_nats'] > 0.01 or abs(steer_r0) > 0.1:
        print("  [PASSED] RG-LRU transplantation produces resolved internal and output divergence in RecurrentGemma-IT.")
    else:
        print("  [WARNING] RG-LRU transplantation produces minimal output divergence in RecurrentGemma-IT.")
    print("=" * 90)

    # Save results
    out_dir = Path("results") / "e14_latent_metacognition" / "potency_microscope"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "it_potency_report.json"
    data = {
        "model_id": model_id,
        "pair_id": pair.pair_id,
        "geom_4k": geom_4k,
        "n0": {
            "whole_kl": res_whole_0["kl_div_nats"],
            "whole_max_logit_diff": res_whole_0["max_logit_diff"],
            "whole_steer": steer_w0,
            "rglru_kl": res_rglru_0["kl_div_nats"],
            "rglru_max_logit_diff": res_rglru_0["max_logit_diff"],
            "rglru_steer": steer_r0,
        },
        "n256": {
            "whole_kl": res_whole_256["kl_div_nats"],
            "whole_max_logit_diff": res_whole_256["max_logit_diff"],
            "whole_steer": steer_w256,
            "rglru_kl": res_rglru_256["kl_div_nats"],
            "rglru_max_logit_diff": res_rglru_256["max_logit_diff"],
            "rglru_steer": steer_r256,
        },
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved potency report to {out_file}\n")


if __name__ == "__main__":
    main()
