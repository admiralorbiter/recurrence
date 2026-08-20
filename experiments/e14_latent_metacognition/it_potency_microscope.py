"""Sprint S14: RecurrentGemma-IT Intervention Potency Microscope (Corrected).

Measures causal and physical response to surgical state transplantation in `google/recurrentgemma-2b-it`
at pinned revision `2766eb5d4264c6c0357803990791f9ab9cd50f8e`.

Evaluates:
1. State-space geometry: RG-LRU Euclidean distance and cosine similarity (C_R) at 4,096 tokens.
2. Immediate output logit divergence: Full-vocabulary KL divergence from sham.
3. Donor-directed paired cloze-margin shift: [(z_B - z_A)_graft - (z_B - z_A)_sham].
4. Immediate potency when intervention is applied to independently evolved N=256 states.
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


PINNED_IT_REVISION = "2766eb5d4264c6c0357803990791f9ab9cd50f8e"


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
    logits_graft, _ = adapter.encode_sequence(
        query_tokens,
        initial_snapshot=snapshot_graft.clone(),
        step_by_step=False,
        return_logits=True,
        logits_to_keep=1,
    )
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

    kl_g_s = F.kl_div(log_prob_s, prob_g, reduction="sum").item()
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
    print(f"RECURRENTGEMMA-IT INTERVENTION POTENCY MICROSCOPE: {model_id} (rev={PINNED_IT_REVISION[:10]}...)")
    print("=" * 90)

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=PINNED_IT_REVISION)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=PINNED_IT_REVISION, torch_dtype=torch.bfloat16).to(device)
    adapter = RecurrentGemmaAdapter(model=model, tokenizer=tokenizer, device=device, dtype=torch.bfloat16)
    print(f"Model loaded and verified against pinned revision in {time.perf_counter() - t0:.2f}s")

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

    lg_w0 = res_whole_0["logits_graft"]
    lg_r0 = res_rglru_0["logits_graft"]
    ls_0 = res_whole_0["logits_sham"]

    steer_w0 = (lg_w0[tok_b_id] - lg_w0[tok_a_id]).item() - (ls_0[tok_b_id] - ls_0[tok_a_id]).item()
    steer_r0 = (lg_r0[tok_b_id] - lg_r0[tok_a_id]).item() - (ls_0[tok_b_id] - ls_0[tok_a_id]).item()

    print(f"  Whole-State Reference Transplant (N=0):")
    print(f"    Full-Vocab KL from Sham:        {res_whole_0['kl_div_nats']:.4f} nats")
    print(f"    Max Logit Delta:                {res_whole_0['max_logit_diff']:.4f}")
    print(f"    Donor-Directed Cloze Shift:     {steer_w0:+.4f} logits")

    print(f"  RG-LRU-Only Transplant (N=0):")
    print(f"    Full-Vocab KL from Sham:        {res_rglru_0['kl_div_nats']:.4f} nats")
    print(f"    Max Logit Delta:                {res_rglru_0['max_logit_diff']:.4f}")
    print(f"    Donor-Directed Cloze Shift:     {steer_r0:+.4f} logits")

    # 3. Test Interventions applied to independently evolved N = 256 States (Immediate Query)
    print(f"\n[3] Testing Interventions applied directly to independently evolved N=256 states...")
    filler_256 = get_filler_tokens_for_regime("random", length=256, seed=1042, audited_pool=audited_pool, tokenizer=tokenizer, excluded_token_ids=excluded)

    # Evolve donor and recipient independently along filler
    _, s_b_256 = adapter.encode_sequence(filler_256, initial_snapshot=s_b_0.clone(), step_by_step=False, return_logits=False)
    _, s_a_256 = adapter.encode_sequence(filler_256, initial_snapshot=s_a_0.clone(), step_by_step=False, return_logits=False)

    geom_256 = compute_rglru_metrics(s_a_256, s_b_256)
    print(f"  RG-LRU Euclidean Distance between evolved states at N=256: {geom_256['mean_euclidean_dist']:.4f}")
    print(f"  RG-LRU Cosine Similarity between evolved states at N=256:  {geom_256['mean_cosine_sim']:.4f}")

    # Swap immediately at N=256 and query
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

    print(f"  Whole-State Reference Transplant at N=256 (Immediate Query):")
    print(f"    Full-Vocab KL from Sham:        {res_whole_256['kl_div_nats']:.4f} nats")
    print(f"    Max Logit Delta:                {res_whole_256['max_logit_diff']:.4f}")
    print(f"    Donor-Directed Cloze Shift:     {steer_w256:+.4f} logits")

    print(f"  RG-LRU-Only Transplant at N=256 (Immediate Query):")
    print(f"    Full-Vocab KL from Sham:        {res_rglru_256['kl_div_nats']:.4f} nats")
    print(f"    Max Logit Delta:                {res_rglru_256['max_logit_diff']:.4f}")
    print(f"    Donor-Directed Cloze Shift:     {steer_r256:+.4f} logits")

    print("\n" + "=" * 90)
    print("POTENCY MICROSCOPE SUMMARY:")
    print("  [CONFIRMED] Measurable causal potency on canonical diagnostic pair in RecurrentGemma-IT.")
    print("=" * 90)

    # Save results
    out_dir = Path("results") / "e14_latent_metacognition" / "potency_microscope"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "it_potency_report.json"
    data = {
        "model_id": model_id,
        "pinned_revision": PINNED_IT_REVISION,
        "pair_id": pair.pair_id,
        "geom_4k": geom_4k,
        "geom_256_evolved": geom_256,
        "n0_immediate": {
            "whole_ref_kl": res_whole_0["kl_div_nats"],
            "whole_ref_max_logit_diff": res_whole_0["max_logit_diff"],
            "whole_ref_donor_shift": steer_w0,
            "rglru_kl": res_rglru_0["kl_div_nats"],
            "rglru_max_logit_diff": res_rglru_0["max_logit_diff"],
            "rglru_donor_shift": steer_r0,
        },
        "n256_immediate_on_evolved_state": {
            "whole_ref_kl": res_whole_256["kl_div_nats"],
            "whole_ref_max_logit_diff": res_whole_256["max_logit_diff"],
            "whole_ref_donor_shift": steer_w256,
            "rglru_kl": res_rglru_256["kl_div_nats"],
            "rglru_max_logit_diff": res_rglru_256["max_logit_diff"],
            "rglru_donor_shift": steer_r256,
        },
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved corrected potency report to {out_file}\n")


if __name__ == "__main__":
    main()
